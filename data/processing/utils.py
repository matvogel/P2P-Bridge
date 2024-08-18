import fpsample
import numpy as np
import torch
from cuml.neighbors import NearestNeighbors
from loguru import logger
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from pytorch3d.ops import sample_farthest_points
from sklearn import neighbors


def optimize_assignments(A: ArrayLike, B: ArrayLike, closest_neighbors: ArrayLike) -> ArrayLike:
    """
    Optimize the assignments from A to B, maximizing unique mappings in B while minimizing total distance.

    Args:
        A (ArrayLike): Nx3 matrix representing points in 3D.
        B (ArrayLike): Nx3 matrix representing points in 3D.
        closest_neighbors (ArrayLike): Indices of the k closest points in B for each point in A.

    Returns:
        ArrayLike: Indices of the k closest points in B for each point in A
    """
    N = A.shape[0]
    assigned_B_indices = -1 * np.ones(N, dtype=int)  # Initialize with -1 (unassigned)
    available_B_points = set(range(B.shape[0]))  # Set of available points in B

    for i, neighbors in enumerate(closest_neighbors):
        # Try to assign to the closest available neighbor
        for neighbor in neighbors:
            if neighbor in available_B_points:
                assigned_B_indices[i] = neighbor
                available_B_points.remove(neighbor)
                break

        # If all neighbors are already assigned, assign to the closest regardless of uniqueness
        if assigned_B_indices[i] == -1:
            assigned_B_indices[i] = neighbors[0]

    return assigned_B_indices


def find_closest_neighbors_cuml(A: ArrayLike, B: ArrayLike, k: int = 5) -> ArrayLike:
    """
    For each point in A, efficiently find the k closest points in B using NearestNeighbors from scikit-learn.

    Args:
        A (ArrayLike): Nx3 matrix representing points in 3D.
        B (ArrayLike): Nx3 matrix representing points in 3D.
        k (int): Number of neighbors to find.

    Returns:
        ArrayLike: Indices of the k closest points in B for each point in A.
    """
    # Using NearestNeighbors to find k closest points
    neigh = NearestNeighbors(n_neighbors=k, metric="l2")
    neigh.fit(B)
    _, indices = neigh.kneighbors(A)

    return indices


@torch.no_grad()
def create_spherical_batches(
    pcd_clean: ArrayLike,
    pcd_noisy: ArrayLike,
    rgb_clean: ArrayLike,
    rgb_noisy: ArrayLike,
    features: ArrayLike,
    args: DictConfig,
) -> ArrayLike:
    """
    Create spherical batches of points from clean and noisy point clouds.

    Args:
        pcd_clean (ArrayLike): Nx3 matrix representing points in 3D.
        pcd_noisy (ArrayLike): Nx3 matrix representing points in 3D.
        rgb_clean (ArrayLike): Nx3 matrix representing RGB values.
        rgb_noisy (ArrayLike): Nx3 matrix representing RGB values.
        features (ArrayLike): Nx128 matrix representing features.
        args (DictConfig): Configuration.

    Returns:
        ArrayLike: List of dictionaries containing clean and noisy points, colors, features, center, and scale.

    Raises:
        AssertionError: If the number of batches is not equal to the number of indices.
    """
    tree_clean = neighbors.KDTree(pcd_clean, metric="l2")
    tree_noisy = neighbors.KDTree(pcd_noisy, metric="l2")

    # calculate number of center points
    n_batches = int(np.ceil(pcd_noisy.shape[0] / args.npoints))

    # get center points of batches
    idxs = fpsample.bucket_fps_kdline_sampling(pcd_noisy, n_batches, h=5)
    center_points = pcd_noisy[idxs]

    # first query points in radius
    idxs_clean = tree_clean.query_radius(center_points, r=args.r, return_distance=False)
    idxs_noisy = tree_noisy.query_radius(center_points, r=args.r, return_distance=False)

    assert len(idxs_clean) == len(idxs_noisy) == n_batches, "Number of batches is not equal to number of indices"

    data = []
    n_skipped = 0
    unique_assignments = 0.0
    avg_noisy_pts = 0.0

    for idx in range(len(idxs_noisy)):
        clean_batch_points = pcd_clean[idxs_clean[idx]]
        noisy_batch_points = pcd_noisy[idxs_noisy[idx]]
        clean_batch_colors = rgb_clean[idxs_clean[idx]]
        noisy_batch_colors = rgb_noisy[idxs_noisy[idx]]
        noisy_batch_features = features[idxs_noisy[idx]] if features is not None else None

        # skip if the batch is too small
        if len(clean_batch_points) < args.npoints:
            n_skipped += 1
            continue

        # skip if we would have to upsample the iphone batch by a factor bigger than 8
        if len(noisy_batch_points) < args.npoints // 8:
            n_skipped += 1
            continue

        avg_noisy_pts += len(noisy_batch_points)
        diff = args.npoints - len(noisy_batch_points)

        if diff > 0:
            rand_idx = np.random.randint(0, len(noisy_batch_points), diff)
            noisy_additional_xyz = noisy_batch_points[rand_idx]
            noisy_additional_rgb = noisy_batch_colors[rand_idx]
            noisy_additional_dino = noisy_batch_features[rand_idx] if features is not None else None

            # calculate bounding box diagonal of noisy batch points
            diagonal = np.linalg.norm(np.max(noisy_batch_points, axis=0) - np.min(noisy_batch_points, axis=0))
            noise_level = 1e-2 * diagonal

            noisy_additional_xyz += np.random.normal(0, noise_level, noisy_additional_xyz.shape)

            noisy_batch_points = np.concatenate([noisy_batch_points, noisy_additional_xyz])
            noisy_batch_colors = np.concatenate([noisy_batch_colors, noisy_additional_rgb])
            noisy_batch_features = (
                np.concatenate([noisy_batch_features, noisy_additional_dino]) if features is not None else None
            )

            # assign points using NN
            cn = find_closest_neighbors_cuml(noisy_batch_points, clean_batch_points, k=128)
            assignment = optimize_assignments(noisy_batch_points, clean_batch_points, cn)

            unique_assignments += len(np.unique(assignment)) / len(assignment)

            clean_batch_points_aligned = clean_batch_points[assignment]
            clean_batch_colors_aligned = clean_batch_colors[assignment]

            # center the points
            center = noisy_batch_points.mean(axis=0)
            clean_batch_points_aligned -= center
            noisy_batch_points -= center

            # scale the points
            scale = np.max(np.linalg.norm(noisy_batch_points, axis=1))
            clean_batch_points_aligned /= scale
            noisy_batch_points /= scale

            # create output data to save as npz later
            batch_data = {}
            batch_data["clean"] = np.concatenate([clean_batch_points_aligned, clean_batch_colors_aligned], axis=1)
            batch_data["noisy"] = np.concatenate([noisy_batch_points, noisy_batch_colors], axis=1)
            batch_data["idxs"] = np.concatenate([idxs_noisy[idx], idxs_noisy[idx][rand_idx]])

            if features is not None:
                batch_data["features"] = noisy_batch_features.astype(np.float16)

            batch_data["center"] = center
            batch_data["scale"] = scale
            data.append(batch_data)
        else:
            noisy_batch_points_torch = torch.from_numpy(noisy_batch_points).float().cuda().unsqueeze(0)

            with torch.no_grad():
                noisy_batch_points, fps_idxs = sample_farthest_points(noisy_batch_points_torch, K=args.npoints)
                noisy_batch_points = noisy_batch_points.squeeze().cpu().numpy()
                fps_idxs = fps_idxs.squeeze().cpu().numpy()

                noisy_batch_colors = noisy_batch_colors[fps_idxs]
                noisy_batch_features = noisy_batch_features[fps_idxs] if features is not None else None

            # assign points using NN
            cn = find_closest_neighbors_cuml(noisy_batch_points, clean_batch_points, k=128)
            assignment = optimize_assignments(noisy_batch_points, clean_batch_points, cn)
            unique_assignments += len(np.unique(assignment)) / len(assignment)

            clean_batch_points_aligned = clean_batch_points[assignment]
            clean_batch_colors_aligned = clean_batch_colors[assignment]

            # center the points
            center = noisy_batch_points.mean(axis=0)
            clean_batch_points_aligned -= center
            noisy_batch_points -= center

            # scale the points
            scale = np.max(np.linalg.norm(noisy_batch_points, axis=1))
            clean_batch_points_aligned /= scale
            noisy_batch_points /= scale

            # create output data to save as npz later
            batch_data = {}
            batch_data["clean"] = np.concatenate([clean_batch_points_aligned, clean_batch_colors_aligned], axis=1)
            batch_data["noisy"] = np.concatenate([noisy_batch_points, noisy_batch_colors], axis=1)
            batch_data["idxs"] = idxs_noisy[idx][fps_idxs]

            if features is not None:
                batch_data["features"] = noisy_batch_features.astype(np.float16)

            batch_data["center"] = center
            batch_data["scale"] = scale

            data.append(batch_data)

    logger.info(f"Skipped {n_skipped} batches out of {len(idxs_noisy)} batches")
    logger.info(f"Unique assignments: {unique_assignments / len(data)}")
    logger.info(f"Average number of points in noisy batches: {avg_noisy_pts / len(idxs_noisy)}")

    return data
