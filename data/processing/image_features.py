import json
import os
from typing import Literal, Tuple

import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms
from einops import rearrange
from loguru import logger
from numba import njit
from numpy.typing import ArrayLike
from PIL import Image
from scannetpp.common.scene_release import ScannetppScene_Release
from scannetpp.common.utils.colmap import read_model
from sklearn.neighbors import KDTree
from torch import Tensor
from tqdm import tqdm


def load_dino(model_name: str) -> torch.nn.Module:
    """
    Load a DINO model from the Facebook Hub.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        torch.nn.Module: The loaded DINO model.
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name).cuda()
    model.eval()
    return model


def make_transform(smaller_edge_size: int) -> transforms.Compose:
    """
    Create a torchvision transform for resizing images and normalizing them.

    Args:
        smaller_edge_size (int): The size of the smaller edge of the image.

    Returns:
        transforms.Compose: The transform to apply to images.
    """
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BILINEAR

    return transforms.Compose(
        [
            transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def prepare_image_batched(
    images: ArrayLike | Tensor, smaller_edge_size: float, patch_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Prepare a batch of images for processing by DINO.

    Args:
        images (ArrayLike | Tensor): The images to process.
        smaller_edge_size (float): The size of the smaller edge of the image.
        patch_size (int): The size of the patches to use.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: The processed image tensor and the grid size.
    """

    transform = make_transform(int(smaller_edge_size))

    image_tensor = transform(images)
    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[2:]  # B x C x H x W
    cropped_width, cropped_height = (
        width - width % patch_size,
        height - height % patch_size,
    )
    image_tensor = image_tensor[:, :, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size)  # h x w
    return image_tensor, grid_size


@torch.no_grad()
def get_dino_features(model: torch.nn.Module, image: Tensor, patch_size: int = 14):
    """Get DINO features for an image.

    Args:
        model (torch.nn.Module): The DINO model.
        image (Tensor): The image to process.
        patch_size (int): The size of the patches to use.

    Returns:
        Tensor: The DINO features for the image.
    """
    B, C, H, W = image.shape
    smaller_edge_size = min(H, W)

    image_batch, grid_size = prepare_image_batched(image, smaller_edge_size, patch_size)

    # t = model.get_intermediate_layers(image_batch)[0].squeeze().half()
    t = model.forward_features(image_batch)["x_norm_patchtokens"].squeeze().half()

    features = t.reshape(B, grid_size[0], grid_size[1], -1)
    features = rearrange(features, "b h w c -> b c h w")

    features = torch.nn.functional.interpolate(features, size=(H, W), mode="bilinear", antialias=False)
    return features


def project_point_cloud_batch(
    points: ArrayLike, translation: ArrayLike, rotation: ArrayLike, camera_matrix: ArrayLike
) -> ArrayLike:
    """
    Project a batch of 3D points onto the image plane using the camera intrinsics.

    Args:
        points (ArrayLike): The 3D points to project.
        translation (ArrayLike): The translation vector of the camera.
        rotation (ArrayLike): The rotation matrix of the camera.
        camera_matrix (ArrayLike): The camera intrinsics matrix.

    Returns:
        ArrayLike: The projected 2D points.
    """

    # Apply rotation: Bx3x3 dot BxNx3 -> BxNx3
    points_rotated = np.matmul(rotation, points.transpose(0, 2, 1))

    # Apply translation: BxNx3 + Bx3x1 -> BxNx3
    points_transformed = points_rotated + translation

    # Apply camera matrix: Bx3x3 dot BxNx3 -> BxNx3
    points_projected = np.matmul(camera_matrix, points_transformed)

    # Normalize: Divide by the z-coordinate to project onto the image plane
    points_projected = points_projected.transpose(0, 2, 1)
    points_projected[..., :2] /= points_projected[..., 2:3]

    return points_projected


@njit
def filter_points_batch_with_occlusion(
    depth_buffer: ArrayLike,
    valid_indices: ArrayLike,
    B: int,
    N: int,
    points_projected: ArrayLike,
    image_width: int,
    image_height: int,
    min_depth: float = 0.1,
    max_depth: float = 1000,
) -> ArrayLike:
    """
    Filter a batch of 3D points based on occlusion and depth.

    Args:
        depth_buffer (ArrayLike): The depth buffer to use for occlusion checking.
        valid_indices (ArrayLike): The array to store the valid indices.
        B (int): The batch size.
        N (int): The number of points.
        points_projected (ArrayLike): The projected 2D points.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        min_depth (float): The minimum depth to consider.
        max_depth (float): The maximum depth to consider.

    Returns:
        ArrayLike: The valid indices.
    """

    for b in range(B):
        for n in range(N):
            x, y, depth = points_projected[b, n]
            x_int, y_int = int(x), int(y)

            if 0 <= x_int < image_width and 0 <= y_int < image_height and min_depth < depth < max_depth:
                if depth < depth_buffer[b, y_int, x_int]:
                    depth_buffer[b, y_int, x_int] = depth
                    valid_indices[b, n] = True
                else:
                    valid_indices[b, n] = False
            else:
                valid_indices[b, n] = False

    return valid_indices


def map_image_features_to_filtered_ptc_batch(
    images: ArrayLike, projected_points_batch: ArrayLike, valid_indices_batch: ArrayLike
) -> ArrayLike:
    """
    Map image features to a batch of 3D points, filtering out invalid points.

    Args:
        images (ArrayLike): A batch of images.
        projected_points_batch (ArrayLike): A batch of projected 3D points.
        valid_indices_batch (ArrayLike): A batch of boolean valid indices.

    Returns:
        ArrayLike: The mapped features for each batch.
    """
    mapped_features_batch = []

    for image, projected_points, valid_indices in zip(images, projected_points_batch, valid_indices_batch):
        valid_points_2d = projected_points[valid_indices, :2]

        # Extract x and y coordinates, ensuring they are within the image bounds
        x, y = valid_points_2d[:, 0].astype(int), valid_points_2d[:, 1].astype(int)
        x = np.clip(x, 0, image.shape[1] - 1)
        y = np.clip(y, 0, image.shape[0] - 1)

        rgb_values = image[y, x]
        mapped_features_batch.append(rgb_values)

    return mapped_features_batch


@torch.jit.script
def map_image_features_to_filtered_ptc_batch_torch(
    mapped_features: ArrayLike, images: ArrayLike, projected_points_batch: ArrayLike, valid_indices_batch: ArrayLike
):
    """
    Map image features to a batch of 3D points, filtering out invalid points.

    Args:
        mapped_features (ArrayLike): The mapped features for each batch.
        images (ArrayLike): A batch of images.
        projected_points_batch (ArrayLike): A batch of projected 3D points.
        valid_indices_batch (ArrayLike): A batch of boolean valid indices.

    Returns:
        ArrayLike: The mapped features for each batch.
    """
    for image, projected_points, valid_indices in zip(images, projected_points_batch, valid_indices_batch):
        valid_points_2d = projected_points[valid_indices, :2]

        # Extract x and y coordinates, ensuring they are within the image bounds
        x, y = valid_points_2d[:, 0].long(), valid_points_2d[:, 1].long()
        x = torch.clip(x, 0, image.shape[1] - 1)
        y = torch.clip(y, 0, image.shape[0] - 1)

        rgb_values = image[y, x]
        mapped_features = torch.cat([mapped_features, rgb_values])

    return mapped_features


@torch.jit.script
def update_features_batched_torch(
    feature_array: ArrayLike, count_array: ArrayLike, new_features: ArrayLike, valid_indices: ArrayLike
):
    """
    Update a batch of features with new features, incrementing the count for each feature.

    Args:
        feature_array (ArrayLike): The feature array to update.
        count_array (ArrayLike): The count array to update.
        new_features (ArrayLike): The new features to add.
        valid_indices (ArrayLike): The valid indices for the new features.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The updated feature and count arrays.
    """
    for batch_new_features, batch_valid_indices in zip(new_features, valid_indices):
        indices = torch.where(batch_valid_indices)[0]
        count_array[indices] += 1

        current_counts = count_array[indices]
        current_means = feature_array[indices]

        increment = (batch_new_features - current_means) / current_counts
        feature_array[indices] += increment

    return feature_array, count_array


def interpolate_missing_features(
    ptc_feats: ArrayLike, ptc_feats_count: ArrayLike, points: ArrayLike, f_shape: ArrayLike, batch_size: int = 128
):
    """
    Interpolate missing features in a point cloud using KNN interpolation.

    Args:
        ptc_feats (ArrayLike): The point cloud features.
        ptc_feats_count (ArrayLike): The count of features for each point.
        points (ArrayLike): The 3D points.
        f_shape (ArrayLike): The shape of the features.
        batch_size (int): The batch size for KNN interpolation.

    Returns:
        ArrayLike: The interpolated point cloud features.
    """
    missing_idx = np.where(ptc_feats_count == 0)[0]

    if len(missing_idx) == 0:
        return ptc_feats

    logger.info(
        "Interpolating",
        len(missing_idx),
        "missing features. Total points:",
        len(ptc_feats),
    )

    tree = KDTree(points)
    batch_size = min(batch_size, len(missing_idx))
    batches = np.array_split(missing_idx, max(len(missing_idx) // batch_size, 1))

    for batch in tqdm(batches, total=len(batches), desc="KNN interpolation"):
        _, idx = tree.query(points[batch], k=10)
        for batch_idx, neighbor_batch in enumerate(idx):
            nonzero_neighbors = ptc_feats[neighbor_batch, :]
            nonzero_neighbors_mask = np.any(nonzero_neighbors != np.zeros(f_shape), axis=-1)
            nonzero_neighbors = nonzero_neighbors[nonzero_neighbors_mask]
            if nonzero_neighbors.shape[0] == 0:
                neighbors_agg = np.zeros(f_shape)
            else:
                neighbors_agg = np.median(nonzero_neighbors, axis=-2)
            ptc_feats[batch[batch_idx]] = neighbors_agg

    return ptc_feats


def process_scene(
    scene_id: str,
    data_root: str,
    target_path: str,
    feature_type: str,
    feature_suffix: str,
    model: torch.nn.Module,
    f_shape: ArrayLike,
    sampling_rate: int = 5,
    image_width: int = 256,
    image_height: int = 192,
    mask_height: int = 192,
    mask_width: int = 256,
    downscale: bool = True,
    batch_size: int = 2,
    overwrite: bool = False,
    autoskip: bool = False,
    pointcloud_source: Literal["iphone", "faro"] = "iphone",
    intrinsic_scale: float = 256 / 1920,
) -> None:
    """
    Process a scene to extract image features for a point cloud.

    Args:
        scene_id (str): The ID of the scene to process.
        data_root (str): The root directory of the data.
        target_path (str): The path to save the features to.
        feature_type (str): The type of features to extract.
        feature_suffix (str): The suffix to add to the feature file.
        model (torch.nn.Module): The DINO model to use for feature extraction.
        f_shape (ArrayLike): The shape of the features.
        sampling_rate (int): The rate at which to sample frames.
        image_width (int): The width of the images.
        image_height (int): The height of the images.
        mask_height (int): The height of the occlusion mask.
        mask_width (int): The width of the occlusion mask.
        downscale (bool): Whether to downscale the images.
        batch_size (int): The batch size for processing.
        overwrite (bool): Whether to overwrite existing features.
        autoskip (bool): Whether to automatically skip frames.
        pointcloud_source (Literal["iphone", "faro"]): The source of the point cloud.
        intrinsic_scale (float): The scale of the intrinsics.
    """

    if os.path.exists(target_path + ".npy") and not overwrite:
        logger.info("Already processed scene", scene_id)
        return

    # load up scene configuration
    scene = ScannetppScene_Release(scene_id, data_root=data_root)
    mesh_path = scene.scan_mesh_path

    if not os.path.exists(mesh_path):
        logger.warning("No mesh found for scene (this is normal)", scene_id)
        return

    if "iphone" in pointcloud_source:
        mesh_path = os.path.dirname(mesh_path) + f"/iphone{feature_suffix}.ply"
    elif pointcloud_source == "faro":
        pass
    else:
        raise ValueError("Unknown pointcloud source")

    colmap_dir = scene.iphone_colmap_dir
    _, images, _ = read_model(colmap_dir, ".txt", read=["images"])
    iphone_intrinsics_path = scene.iphone_pose_intrinsic_imu_path
    iphone_intrinsics = json.load(open(iphone_intrinsics_path))

    pcd = o3d.io.read_point_cloud(str(mesh_path))

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # remove nans or infs
    removal_mask = np.any(np.isnan(points), axis=1) | np.any(np.isinf(points), axis=1)
    points = points[~removal_mask]
    colors = colors[~removal_mask]

    # initialize features and set count to 0
    ptc_feats = torch.zeros((len(points), f_shape), dtype=torch.float16).cuda()
    ptc_feats_count = torch.zeros((len(points), 1), dtype=torch.long).cuda()

    # calculate features in batches, first skip every nth scan
    total_data = len(images)
    if autoskip:
        if total_data < 25:
            skip_scans = 1
        elif total_data < 50:
            skip_scans = 5
        elif total_data < 100:
            skip_scans = 10
        elif total_data < 500:
            skip_scans = 20
        else:
            skip_scans = sampling_rate
    else:
        skip_scans = sampling_rate

    skip_scans = min(5, skip_scans)

    images = list(images.values())
    images = images[::skip_scans]

    # recalculate total data
    total_data = len(images)

    if total_data == 0:
        print("Not enough frames for scene", scene_id)
        return

    # split into batches of maximum shape of batch_size
    num_batches = int(np.ceil(total_data / batch_size))
    batches = np.array_split(np.arange(total_data), num_batches)

    # read all images and convert to numpy
    images_list = [
        np.array(Image.open(os.path.join(scene.iphone_rgb_dir, image.name.replace("jpg", "png")))) for image in images
    ]
    images_list = np.array(images_list, dtype=np.float32) / 255.0

    for batch in tqdm(batches, total=len(batches), desc="Processing images"):
        # expand dims to batch size
        points_batch = np.expand_dims(points, axis=0).repeat(len(batch), axis=0)

        # create batch of frames
        frame_names = [images[i].name for i in batch]
        videoframes = images_list[batch]

        # get intrinsics and extrinsics
        intrinsic_matrices = np.array(
            [
                iphone_data["intrinsic"]
                for iphone_data in [iphone_intrinsics[frame_name.split(".")[0]] for frame_name in frame_names]
            ]
        )

        # scale intrinsics and videoframes
        intrinsic_matrices[:, :2, :] *= intrinsic_scale

        world_to_cameras = [images[i].world_to_camera for i in batch]

        Rs = np.array([world_to_camera[:3, :3] for world_to_camera in world_to_cameras])
        ts = np.array([world_to_camera[:-1, -1:] for world_to_camera in world_to_cameras])

        points_projected = project_point_cloud_batch(points_batch, ts, Rs, intrinsic_matrices)

        # buffers for occlusino check
        B, N, _ = points_projected.shape
        depth_buffer = np.full((B, image_height, image_width), np.inf)
        valid_indices = np.zeros((B, N), dtype=bool)

        if downscale:
            valid_indices = filter_points_batch_with_occlusion(
                depth_buffer, valid_indices, B, N, points_projected, mask_width, mask_height
            )
        else:
            valid_indices = filter_points_batch_with_occlusion(points_projected, image_width, image_height)

        points_projected = torch.from_numpy(points_projected).cuda()
        valid_indices = torch.from_numpy(valid_indices).cuda()

        # extract features
        if feature_type == "rgb":
            features = map_image_features_to_filtered_ptc_batch(videoframes, points_projected, valid_indices)
        elif feature_type == "dino":
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                videoframes = torch.from_numpy(videoframes).permute(0, 3, 1, 2).type(torch.float16).cuda()
                dino_feats = get_dino_features(model, videoframes).detach()
            dino_feats = rearrange(dino_feats, "b c h w -> b h w c")
            features = torch.tensor([], dtype=torch.float16, device="cuda")
            features = map_image_features_to_filtered_ptc_batch_torch(
                features, dino_feats, points_projected, valid_indices
            )

        ptc_feats, ptc_feats_count = update_features_batched_torch(ptc_feats, ptc_feats_count, features, valid_indices)
        torch.cuda.empty_cache()

    # interpolate missing features using KNN
    ptc_feats = ptc_feats.cpu().numpy()
    ptc_feats_count = ptc_feats_count.cpu().numpy()
    ptc_feats = interpolate_missing_features(ptc_feats, ptc_feats_count, points, f_shape)

    np.nan_to_num(ptc_feats, copy=False)

    # save features in transposed format for faster reading
    np.save(target_path, ptc_feats.astype(np.float16).T)
