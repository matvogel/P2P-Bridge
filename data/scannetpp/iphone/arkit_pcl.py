"""
The script reads the iphone RGB, depth images, and the corresponding camera poses and intrinsics, and backproject them into a point cloud.s
"""

import argparse
import json
import os
from typing import Optional

import numpy as np
import open3d as o3d
from common.scene_release import ScannetppScene_Release
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, default="02455b3d20")
    parser.add_argument("--data_root", type=str, required=True, help="The root directory of the data.")
    parser.add_argument("--output", type=str, default="pcl.ply", help="The output filename (PLY format).")
    parser.add_argument("--sample_rate", type=int, default=10, help="Sample rate of the frames.")
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--grid_size", type=float, default=0.05, help="Grid size for voxel downsampling.")
    parser.add_argument("--n_outliers", type=int, default=50, help="Number of neighbors for outlier removal.")
    parser.add_argument("--outlier_radius", type=float, default=0.1, help="Radius for outlier removal.")
    parser.add_argument("--final_grid_size", type=float, default=0.05, help="Grid size for voxel downsampling.")
    parser.add_argument("--final_n_outliers", type=int, default=20, help="Number of neighbors for outlier removal.")
    parser.add_argument("--final_outlier_radius", type=float, default=0.05, help="Radius for outlier removal.")

    return parser.parse_args()


def filter_iphone_scan_fast(iphone_scan, iphone_colors, faro_scan, features=None, threshold=1.0):
    """
    Filter the iPhone scan to remove points that are too far from the Faro scan.
    This is done by calculating the nearest neighbor distances between the two scans,
    and then removing points that are more than the specified threshold times the std away from the mean.
    """
    import cudf
    import cuml
    import pandas as pd

    df = cudf.DataFrame()
    df["x"] = faro_scan[:, 0]
    df["y"] = faro_scan[:, 1]
    df["z"] = faro_scan[:, 2]
    knn = cuml.NearestNeighbors(n_neighbors=1)
    knn.fit(df)
    dists, inds = knn.kneighbors(cudf.from_pandas(pd.DataFrame(iphone_scan)))
    dists = np.array(dists.to_pandas().values).ravel()
    mean = dists.mean()
    std = dists.std()
    threshold = mean + threshold * std
    mask = dists < threshold
    return iphone_scan[mask], iphone_colors[mask], features[mask] if features is not None else None


def voxel_down_sample_with_features(xyz, rgb, voxel_size, feats):
    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Voxel downsampling
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # Initialize lists for downsampled data
    downsampled_xyz = []
    downsampled_rgb = []
    downsampled_feats = []

    voxels = voxel_grid.get_voxels()

    # Iterate through each voxel and aggregate features
    for voxel in voxels:
        indices = voxel.grid_index

        # Aggregate points, colors, and features
        voxel_xyz = np.mean(xyz[indices], axis=0)
        voxel_rgb = np.mean(rgb[indices], axis=0)
        voxel_feats = np.mean(feats[indices], axis=0)

        downsampled_xyz.append(voxel_xyz)
        downsampled_rgb.append(voxel_rgb)
        downsampled_feats.append(voxel_feats)

    return np.array(downsampled_xyz), np.array(downsampled_rgb), np.array(downsampled_feats)


def merge_gpu_pcds(pcd, pcd_new):
    if pcd.is_empty():
        return pcd_new
    pcd.point.positions = pcd.point.positions.append(pcd_new.point.positions)
    pcd.point.colors = pcd.point.colors.append(pcd_new.point.colors)
    return pcd


def create_gpu_pointcloud(xyz=None, rgb=None, pcd=None):
    if pcd is not None:
        xyz = np.array(pcd.points)
        rgb = np.array(pcd.colors)

    device = o3d.core.Device("CPU:0")

    # using open3d tensor operations
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(xyz, device=device, dtype=o3d.core.float32)
    if rgb is not None:
        pcd.point.colors = o3d.core.Tensor(rgb, device=device, dtype=o3d.core.float32)

    return pcd.cuda()


def outlier_removal_cuml(xyz, rgb, nb_points, radius, std_thresh=None, feats=None):
    # outlier removal by cuml
    # query neighbors and filter out if biggest radius is larger than threshold

    # create cuml dataframe
    import cudf
    import cuml

    df = cudf.DataFrame()
    df["x"] = xyz[:, 0]
    df["y"] = xyz[:, 1]
    df["z"] = xyz[:, 2]

    if std_thresh is not None:
        nb_points = 20

    # create cuml knn
    knn = cuml.NearestNeighbors(n_neighbors=nb_points)
    knn.fit(df)

    # query neighbors
    dists, inds = knn.kneighbors(df)
    dists = np.array(dists.to_pandas().values)

    # filter out
    if std_thresh is None:
        mask = dists[:, -1] < radius
    else:
        # calculate the mean and std of all the distances. Then filter out points that are not within 1 std of the mean.
        mean = np.mean(dists[:, -1])
        std = np.std(dists[:, -1])
        mask = (dists[:, -1] < mean + std * std_thresh) & (dists[:, -1] > mean - std * std_thresh)

    xyz = xyz[mask]
    rgb = rgb[mask]
    if feats is not None:
        feats = feats[mask]
    return xyz, rgb, feats


def lookup_features(uv, feature_map):
    """
    Look up features from a feature map using uv coordinates.

    Args:
    uv (np.ndarray): An array of shape (N, 2) containing uv coordinates.
    feature_map (np.ndarray): The feature map of shape (H, W, F).

    Returns:
    np.ndarray: An array of shape (N, F) containing the features corresponding to the uv coordinates.
    """
    H, W, F = feature_map.shape

    # Round coordinates and convert to integer
    uv_int = np.round(uv).astype(int)

    # Bound the coordinates to be within the feature map
    uv_int[:, 0] = np.clip(uv_int[:, 0], 0, W - 1)
    uv_int[:, 1] = np.clip(uv_int[:, 1], 0, H - 1)

    # Look up the features
    features = feature_map[uv_int[:, 1], uv_int[:, 0], :]

    return features


def project_to_image(
    xyz: np.ndarray, rgb: np.ndarray, image: np.ndarray, camera_to_world: np.ndarray, intrinsic: np.ndarray
):
    """
    Project the point cloud back to the image plane.

    Args:
    xyz (np.ndarray): The XYZ coordinates of the point cloud.
    rgb (np.ndarray): The RGB values corresponding to each point in the point cloud.
    camera_to_world (np.ndarray): The camera to world transformation matrix.
    intrinsic (np.ndarray): The camera intrinsic matrix.

    Returns:
    np.ndarray, np.ndarray: The pixel coordinates and the corresponding RGB values.
    """
    # Convert world coordinates to camera coordinates
    world_to_camera = np.linalg.inv(camera_to_world)
    xyz_camera = world_to_camera @ np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1).T

    # Project the points to the image plane
    uvz = intrinsic @ xyz_camera[:3, :]
    uv = uvz[:2, :] / uvz[2, :]
    uv = uv.T

    # Filter out points that are not within the image frame
    valid_indices = (uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])
    uv = uv[valid_indices]
    rgb = rgb[valid_indices]

    return uv, rgb


def backproject(
    image: np.ndarray,
    depth: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    use_point_subsample: bool = True,
    point_subsample_rate: int = 4,
    use_voxel_subsample: bool = True,
    voxel_grid_size: float = 0.02,
    n_outliers: int = 20,
    outlier_radius: float = 0.1,
    no_filtering: bool = False,
):
    """Backproject RGB-D image into a point cloud.
    The resolution of RGB and depth are not be the same (the aspect ratio should be the smae).
    Therefore, we need to scale the RGB image and the intrinsic matrix to match the depth.
    """

    # Scale the intrinsic matrix
    scale_factor = 256 / 1920
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= scale_factor
    intrinsic[1, 1] *= scale_factor
    intrinsic[0, 2] *= scale_factor
    intrinsic[1, 2] *= scale_factor

    yy, xx = np.meshgrid(np.arange(0, depth.shape[0]), np.arange(0, depth.shape[1]), indexing="ij")
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    z = depth[yy, xx]
    valid_mask = np.logical_not((z < min_depth) | (z > max_depth) | np.isnan(z) | np.isinf(z))
    x = xx[valid_mask]
    y = yy[valid_mask]
    uv_one = np.stack([x, y, np.ones_like(x)], axis=0)
    xyz = np.linalg.inv(intrinsic) @ uv_one * z[valid_mask]
    xyz_one = np.concatenate([xyz, np.ones_like(xyz[:1, :])], axis=0)
    xyz_one = camera_to_world @ xyz_one
    xyz = xyz_one[:3, :].T
    rgb = image[y, x]

    pcd = create_gpu_pointcloud(xyz, rgb)

    if not no_filtering:
        pcd, _ = pcd.remove_radius_outliers(nb_points=n_outliers, search_radius=outlier_radius)

    if use_point_subsample:
        raise NotImplementedError
    if use_voxel_subsample:
        pcd = pcd.voxel_down_sample(voxel_grid_size)

    return pcd


def save_point_cloud(
    filename: str,
    points: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    binary: bool = True,
    verbose: bool = True,
):
    """Save an RGB point cloud as a PLY file.
    Args:
        filename: The output filename.
        points: Nx3 matrix where each row is a point.
        rgb: Nx3 matrix where each row is the RGB value of the corresponding point. If not provided, use gray color for all the points.
        binary: Whether to save the PLY file in binary format.
        verbose: Whether to print the output filename.
    """
    if rgb is None:
        rgb = np.tile(np.array([128], dtype=np.uint8), (points.shape[0], 3))
    npy_types = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points.shape[0]):
            vertices.append(tuple(points[row_idx, :]) + tuple(rgb[row_idx, :]))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, "vertex")

        # Write
        PlyData([el]).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, "w") as f:
            f.write(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex %d\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n" % points.shape[0]
            )
            for row_idx in range(points.shape[0]):
                X, Y, Z = points[row_idx]
                R, G, B = rgb[row_idx]
                f.write("%f %f %f %d %d %d 0\n" % (X, Y, Z, R, G, B))
    if verbose is True:
        print("Saved point cloud to: %s" % filename)


def main():
    args = parse_args()
    scene = ScannetppScene_Release(args.scene_id, args.data_root)
    iphone_rgb_dir = scene.iphone_rgb_dir
    iphone_depth_dir = scene.iphone_depth_dir

    with open(scene.iphone_pose_intrinsic_imu_path, "r") as f:
        json_data = json.load(f)
    frame_data = [(frame_id, data) for frame_id, data in json_data.items()]
    frame_data.sort()

    all_xyz = []
    all_rgb = []

    for frame_id, data in tqdm(frame_data[:: args.sample_rate]):
        camera_to_world = np.array(data["aligned_pose"]).reshape(4, 4)
        intrinsic = np.array(data["intrinsic"]).reshape(3, 3)
        rgb = np.array(Image.open(os.path.join(iphone_rgb_dir, frame_id + ".jpg")), dtype=np.uint8)
        depth = np.array(Image.open(os.path.join(iphone_depth_dir, frame_id + ".png")), dtype=np.float32) / 1000.0

        xyz, rgb = backproject(
            rgb,
            depth,
            camera_to_world,
            intrinsic,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            use_point_subsample=False,
            use_voxel_subsample=True,
            voxel_grid_size=args.grid_size,
            outlier_radius=args.outlier_radius,
            n_outliers=args.n_outliers,
        )

        all_xyz.append(xyz)
        all_rgb.append(rgb)

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)

    # Voxel downsample again
    print("Final processing...")
    all_xyz, all_rgb, _ = outlier_removal_fast(
        all_xyz, all_rgb, nb_points=args.final_n_outliers, radius=args.final_outlier_radius
    )
    all_xyz, all_rgb = voxel_down_sample_gpu(all_xyz, all_rgb, voxel_size=args.grid_size)

    filename = f"sr={args.sample_rate}_min={args.min_depth}_max={args.max_depth}_grid={args.grid_size}_outlier={args.outlier_radius}_n={args.n_outliers}_final_grid={args.final_grid_size}_final_outlier={args.final_outlier_radius}_final_n={args.final_n_outliers}_icp={icp}.ply"

    save_point_cloud(
        filename=filename,
        points=all_xyz,
        rgb=all_rgb,
        binary=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
