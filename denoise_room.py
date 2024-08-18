import argparse
import os
from typing import Optional, Tuple

import fpsample
import numpy as np
import omegaconf
import open3d as o3d
import torch
from einops import rearrange
from loguru import logger
from numba import njit
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from sklearn import neighbors
from torch import Tensor
from tqdm import tqdm

from metrics.chamfer3D import dist_chamfer_3D
from models.model_loader import load_diffusion
from third_party.pvcnn.functional.sampling import furthest_point_sample


def load_rooom(cfg: DictConfig) -> o3d.geometry.PointCloud:
    """
    Load the room point cloud.

    Args:
        cfg (DictConfig): The configuration.

    Returns:
        o3d.geometry.PointCloud: The room point cloud.
    """

    room = o3d.io.read_point_cloud(cfg.room_path)
    return room


@torch.no_grad()
def remove_outliers(gen: Tensor, ref: Tensor, num_outliers: int) -> Tuple[Tensor, Tensor]:
    """
    Remove outliers from the generated point cloud. Used by previous denoising methods such as PD-Flow.

    Args:
        gen (Tensor): The generated point cloud.
        ref (Tensor): The reference point cloud.
        num_outliers (int): The number of outliers to remove.

    Returns:
        Tuple[Tensor, Tensor]: The filtered point cloud and the mask.
    """

    chamfer_loss = dist_chamfer_3D.chamfer_3DDist()

    (B, N, _), device = gen.shape, gen.device
    dist1, dist2, _, _ = chamfer_loss(gen, ref)  # [B, N1], [B, N2]
    idx_dist1 = torch.argsort(dist1, dim=-1, descending=True)  # [B, N1]
    idx_outliers = idx_dist1[:, :num_outliers]

    # Adjusting the approach for inverse indexing
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    idxb = torch.arange(B, device=device).unsqueeze(-1)
    mask[idxb, idx_outliers] = False
    idx_inverse = mask.nonzero(as_tuple=False)[:, 1].view(B, N - num_outliers)

    # Using the mask to filter out outliers directly
    gen_filtered = gen[mask].view(B, N - num_outliers, -1)

    return gen_filtered, mask


def denoise_patch(
    patch: Tensor,
    model: torch.nn.Module,
    args: DictConfig,
    patch_rgb: Optional[Tensor] = None,
    patch_dino: Optional[Tensor] = None,
) -> Tensor:
    """
    Denoise a single patch.

    Args:
        patch (Tensor): The patch to denoise.
        model (torch.nn.Module): The model to use for denoising.
        args (DictConfig): The configuration.
        patch_rgb (Optional[Tensor]): The RGB features.
        patch_dino (Optional[Tensor]): The DINO features.

    Returns:
        Tensor: The denoised patch.
    """
    center = patch.mean(axis=0)
    patch -= center
    scale = np.max(np.linalg.norm(patch, axis=1))
    patch /= scale

    patch = torch.from_numpy(patch.T).float().cuda().unsqueeze(0)
    patch_rgb = torch.from_numpy(patch_rgb.T).float().cuda().unsqueeze(0) if patch_rgb is not None else None
    patch_dino = torch.from_numpy(patch_dino.T).float().cuda().unsqueeze(0) if patch_dino is not None else None

    x_cond = None
    if args.data.use_rgb_features:
        x_cond = patch_rgb
    if args.data.point_features == "dino":
        x_cond = patch_dino if x_cond is None else torch.cat([x_cond, patch_dino], dim=1)

    with torch.no_grad():
        model_pred = model.sample(x_start=patch, x_cond=x_cond, verbose=False, steps=args.steps, use_ema=args.use_ema)
        model_pred = model_pred["x_pred"].detach().transpose(1, 2)

    patch_denoised = model_pred.squeeze().cpu().numpy()
    return patch_denoised * scale + center


@torch.no_grad()
def denoise_patch_batch(
    patch: Tensor,
    model: torch.nn.Module,
    args: DictConfig,
    patch_rgb: Optional[Tensor] = None,
    patch_dino: Optional[Tensor] = None,
    filtering: bool = False,
    return_steps: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Denoise a batch of patches.

    Args:
        patch (Tensor): The patches to denoise.
        model (torch.nn.Module): The model to use for denoising.
        args (DictConfig): The configuration.
        patch_rgb (Optional[Tensor]): The RGB features.
        patch_dino (Optional[Tensor]): The DINO features.
        filtering (bool): Whether to filter out outliers.
        return_steps (bool): Whether to return intermediate steps.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: The denoised patches and the intermediate steps.
    """
    center = patch.mean(axis=1, keepdims=True)
    patch -= center

    scale = np.linalg.norm(patch, axis=2, keepdims=True).max(axis=1, keepdims=True)
    patch /= scale

    patch = torch.from_numpy(patch).float().cuda().transpose(1, 2)
    patch_rgb = torch.from_numpy(patch_rgb).float().cuda().transpose(1, 2) if patch_rgb is not None else None
    patch_dino = torch.from_numpy(patch_dino).float().cuda().transpose(1, 2) if patch_dino is not None else None

    x_cond = None
    if args.data.use_rgb_features:
        x_cond = patch_rgb
    if args.data.point_features == "dino":
        x_cond = patch_dino if x_cond is None else torch.cat([x_cond, patch_dino], dim=1)

    model_pred = model.sample(
        x_start=patch, x_cond=x_cond, verbose=False, steps=args.steps, use_ema=args.use_ema, log_count=args.steps
    )
    x_pred = model_pred["x_pred"].detach().transpose(1, 2)

    if return_steps:
        x_chain = model_pred["x_chain"]
        x_chain = x_chain.detach().transpose(2, 3).cpu().numpy()
        x_chain = x_chain * rearrange(scale, "b n c -> b 1 n c") + rearrange(center, "b n c -> b 1 n c")
        x_chain = rearrange(x_chain, "b t n c -> t b n c")
    if filtering:
        n_outliers = int(x_pred.shape[1] * 0.01)
        x_pred, filter_mask = remove_outliers(x_pred, patch.transpose(1, 2), n_outliers)
        patch_denoised = x_pred.cpu().squeeze().numpy()
        patch_denoised * scale + center
        return patch_denoised, filter_mask
    else:
        patch_denoised = x_pred.cpu().squeeze().numpy()
        patch_denoised = patch_denoised * scale + center
        if return_steps:
            return patch_denoised, x_chain
        else:
            return patch_denoised, None


@njit
def update_prediction_noisy(
    denoised: ArrayLike, denoised_num_updates: ArrayLike, patch: ArrayLike, idxs: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Update the denoised predictions with new patch values using numpy arrays.

    Args:
        denoised (ArrayLike): The denoised predictions.
        denoised_num_updates (ArrayLike): The number of updates for each point.
        patch (ArrayLike): The new patch values.
        idxs (ArrayLike): The indexes of the points to update.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The updated denoised predictions and the updated number of updates.
    """

    denoised_num_updates[idxs] += 1
    is_first_update = denoised_num_updates[idxs] == 1
    broadcast_mask = is_first_update[:, None]

    denoised[idxs] = np.where(
        broadcast_mask,
        patch,
        (denoised[idxs] * (denoised_num_updates[idxs] - 1)[:, None] + patch) / denoised_num_updates[idxs][:, None],
    )

    return denoised, denoised_num_updates


@njit
def update_prediction_zeros(
    denoised: ArrayLike, denoised_num_updates: ArrayLike, patch: ArrayLike, idxs: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Update the denoised predictions with new patch values using numpy arrays.

    Args:
        denoised (ArrayLike): The denoised predictions.
        denoised_num_updates (ArrayLike): The number of updates for each point.
        patch (ArrayLike): The new patch values.
        idxs (ArrayLike): The indexes of the points to update.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The updated denoised predictions and the updated number of updates.
    """
    denoised_num_updates[idxs] += 1

    current_counts = denoised_num_updates[idxs]
    current_mean = denoised[idxs]

    increment = (patch - current_mean) / current_counts[:, None]

    denoised[idxs] += increment

    return denoised, denoised_num_updates


@njit
def update_prediction_zeros_batches(denoised: ArrayLike, denoised_num_updates, patch_batch, idxs_batch):
    assert patch_batch.shape[0] == len(idxs_batch), "Patch and indexes are of different shape"

    for i in range(len(patch_batch)):
        patch = patch_batch[i]
        idxs = idxs_batch[i]
        # cut_idx = cut_idxs[i]
        # patch = patch[:cut_idx]
        # idxs = idxs[:cut_idx]

        denoised_num_updates[idxs] += 1

        current_counts = denoised_num_updates[idxs]
        current_mean = denoised[idxs]

        increment = (patch - current_mean) / current_counts[:, None]

        denoised[idxs] += increment

    return denoised, denoised_num_updates


@njit
def update_prediction_noisy_batches(denoised, denoised_num_updates, patch_batch, idxs_batch, cut_list):
    assert patch_batch.shape[0] == len(
        idxs_batch
    ), f"Patch and indexes are of different shape {patch_batch.shape,}, {len(idxs_batch)}"

    for i in range(len(patch_batch)):
        patch = patch_batch[i]
        idxs = idxs_batch[i]
        cuts = cut_list[i]

        # cut the patch and idxs
        patch = patch[:cuts]
        idxs = idxs[:cuts]

        denoised_num_updates[idxs] += 1

        # check for first updates
        is_first_update = denoised_num_updates[idxs] == 1
        broadcast_mask = is_first_update[:, None]

        denoised[idxs] = np.where(
            broadcast_mask,
            patch,
            (denoised[idxs] * (denoised_num_updates[idxs] - 1)[:, None] + patch) / denoised_num_updates[idxs][:, None],
        )

    return denoised, denoised_num_updates


def parse_args():
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument("--room_path", type=str, required=True, help="Path to the room point cloud.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_ema", type=bool, default=True, help="Use EMA model for prediction.")
    parser.add_argument("--feature_name", type=str, default="dino_iphone")
    parser.add_argument("--out_path", type=str, default=None, help="Path to save the denoised room.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions.")
    # denoising parameters
    parser.add_argument("--average_predictions", type=bool, default=True, help="Average out predictions.")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for the diffusion.")
    parser.add_argument("--k", type=int, default=4, help="Number of patches to sample.")
    parser.add_argument("--intermediate", action="store_true", help="Save intermediate steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for denoising.")
    # computational parameters
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank.")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU to use.")
    parser.add_argument("--distribution_type", default="none")
    args = parser.parse_args()

    # get config path
    cfg_path = os.path.join(os.path.dirname(args.model_path), "opt.yaml")
    cfg = omegaconf.OmegaConf.load(cfg_path)
    # merge with args
    cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(vars(args)))

    # set some default values
    cfg.restart = False
    return cfg


def load_room_files(args):
    room = load_rooom(args)
    room_points = np.array(room.points)
    room_colors = np.array(room.colors)

    # sanity check colors
    if len(room_colors) != len(room_points):
        logger.warning("Color array has different length than point array. Setting colors to None.")
        room_colors = None

    if args.data.point_features == "dino":
        try:
            room_dino = np.load(
                os.path.join(os.path.dirname(args.room_path), "..", "features", f"{args.feature_name}.npy")
            )

            if "arkit" not in args.data.dataset.lower():
                room_dino = room_dino.T
        except:
            room_dino = None
            logger.warning("No dino features found")
    else:
        room_dino = None

    return room_points, room_colors, room_dino


def create_patches(n_batches, room_points, patch_size, idxs_radius_patches, room_colors=None, room_dino=None):
    denoise_batches_xyz = []
    denoise_batches_rgb = []
    denoise_batches_dino = []
    denoise_idxs = []
    denoised_cut_list = []

    for idx in tqdm(range(n_batches), desc="Creating Patches"):
        mapping_idzs = idxs_radius_patches[idx]
        patch_xyz = room_points[mapping_idzs]
        patch_rgb = room_colors[mapping_idzs] if room_colors is not None else None
        patch_dino = room_dino[mapping_idzs] if room_dino is not None else None

        radius_patch_size = len(patch_xyz)

        diff = patch_size - radius_patch_size

        if diff > 0:
            # we have to upsample the patch to the right amount of points
            rand_idx = np.random.randint(0, len(patch_xyz), diff)
            patch_additional_xyz = patch_xyz[rand_idx]
            patch_additional_rgb = patch_rgb[rand_idx] if patch_rgb is not None else None
            patch_additional_dino = patch_dino[rand_idx] if patch_dino is not None else None

            # add noise to the additional points
            noise_level = np.linalg.norm(np.max(patch_xyz, axis=0) - np.min(patch_xyz, axis=0)) * 1e-2
            patch_additional_xyz += np.random.normal(0, noise_level, patch_additional_xyz.shape)

            patch_xyz = np.concatenate([patch_xyz, patch_additional_xyz], axis=0)
            patch_rgb = np.concatenate([patch_rgb, patch_additional_rgb], axis=0) if patch_rgb is not None else None
            patch_dino = np.concatenate([patch_dino, patch_additional_dino], axis=0) if patch_dino is not None else None

            # also update the mapping idxs
            mapping_idzs_concat = np.concatenate([mapping_idzs, mapping_idzs[rand_idx]])

            denoise_batches_xyz.append(patch_xyz)

            if patch_rgb is not None:
                denoise_batches_rgb.append(patch_rgb)
            if patch_dino is not None:
                denoise_batches_dino.append(patch_dino)

            denoise_idxs.append(mapping_idzs_concat)
            denoised_cut_list.append(radius_patch_size)
        else:
            pointcloud_torch = torch.from_numpy(patch_xyz).float().cuda()
            pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 n d")

            fraction = radius_patch_size // patch_size + 1

            for _ in range(fraction):
                # pts, idxs = sample_farthest_points(pointcloud_torch, K=args.data.npoints, random_start_point=True)
                idxs = fpsample.bucket_fps_kdline_sampling(patch_xyz, patch_size, h=5)
                patch_idxs = mapping_idzs[idxs]

                pts = patch_xyz[idxs]
                rgb = patch_rgb[idxs] if patch_rgb is not None else None
                dino = patch_dino[idxs] if patch_dino is not None else None

                denoise_batches_xyz.append(pts)

                if rgb is not None:
                    denoise_batches_rgb.append(rgb)
                if dino is not None:
                    denoise_batches_dino.append(dino)

                denoise_idxs.append(patch_idxs)
                denoised_cut_list.append(patch_size)

    return denoise_batches_xyz, denoise_batches_rgb, denoise_batches_dino, denoise_idxs, denoised_cut_list


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_training_steps = args.model_path.split("_")[-1].split(".")[0]
    model_config = args.model_path.split("/")[-2]
    ema = "_ema" if args.use_ema else ""

    room_source = args.room_path.split("/")[-1].split(".")[0]

    if args.out_path is not None:
        out_path = os.path.abspath(args.out_path)
    else:
        out_path = os.path.join(
            os.path.dirname(args.room_path),
            "..",
            "predictions",
            "P2SB",
            f"{model_config.replace('_','-')}_{room_source.replace('_','-')}_{model_training_steps}_{args.steps}{ema}.ply",
        )

    if os.path.exists(out_path) and not args.overwrite:
        logger.info(f"Prediction already exists at {out_path}")
        return

    model, _ = load_diffusion(args)

    room_points, room_colors, room_dino = load_room_files(args)

    room_tree = neighbors.KDTree(room_points, metric="l2")

    # generate batches
    patch_size = args.data.npoints
    n_batches = int(np.ceil(room_points.shape[0] / patch_size) * args.k)

    pointcloud_torch = torch.from_numpy(room_points).float().cuda()
    pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 d n")
    center_points = furthest_point_sample(pointcloud_torch, n_batches).squeeze().cpu().numpy().T
    query_radius = 0.3 if "scannet" in args.data.dataset.lower() else 0.5
    logger.info(f"Detected dataset: {args.data.dataset}, denoising in radius {query_radius}")
    idxs_radius_patches = room_tree.query_radius(center_points, r=query_radius, return_distance=False)

    prediction = o3d.geometry.PointCloud()

    if args.average_predictions:
        denoised = room_points.copy()
        denoised_num_updates = np.zeros(room_points.shape[0])
    else:
        denoised = []

    return_steps = args.intermediate
    if return_steps:
        denoised_steps = np.array([room_points.copy()] * args.steps)
        denoised_steps_num_updates = np.array([np.zeros(room_points.shape[0])] * args.steps)

    denoise_batches_xyz, denoise_batches_rgb, denoise_batches_dino, denoise_idxs, denoised_cut_list = create_patches(
        n_batches, room_points, patch_size, idxs_radius_patches, room_colors, room_dino
    )

    # create batches
    denoise_batches_xyz = np.array(denoise_batches_xyz)
    denoise_batches_rgb = np.array(denoise_batches_rgb) if len(denoise_batches_rgb) > 0 else None
    denoise_batches_dino = np.array(denoise_batches_dino) if len(denoise_batches_dino) > 0 else None
    denoise_idxs = np.array(denoise_idxs)
    cut_idxs = np.array(denoised_cut_list)

    # crate indexing array for batch size of 32
    n_batches = int(np.ceil(denoise_batches_xyz.shape[0] / args.batch_size))
    batch_idxs = np.array_split(np.arange(denoise_batches_xyz.shape[0]), n_batches)
    filtering = False

    # denoise
    for idx in tqdm(range(n_batches), desc="Denoising"):
        start = batch_idxs[idx][0]
        end = batch_idxs[idx][-1]

        patch_xyz = denoise_batches_xyz[start:end]
        patch_rgb = denoise_batches_rgb[start:end] if denoise_batches_rgb is not None else None
        patch_dino = denoise_batches_dino[start:end] if denoise_batches_dino is not None else None
        patch_idxs = denoise_idxs[start:end]
        patch_cut_idxs = cut_idxs[start:end]

        # denoise
        if filtering:
            patch_xyz, mask = denoise_patch_batch(patch_xyz, model, args, patch_rgb, patch_dino, filtering=filtering)
            B, n_true = patch_xyz.shape[0], patch_xyz.shape[1]
            mask = mask.cpu().numpy()
            patch_idxs = patch_idxs[mask]
            patch_idxs = patch_idxs.reshape(B, n_true)
        else:
            patch_xyz, patch_steps = denoise_patch_batch(
                patch_xyz, model, args, patch_rgb, patch_dino, return_steps=return_steps
            )

        if args.average_predictions:
            denoised, denoised_num_updates = update_prediction_noisy_batches(
                denoised, denoised_num_updates, patch_xyz, patch_idxs, patch_cut_idxs
            )
            if return_steps:
                for i in range(len(patch_steps)):
                    step_denoised = denoised_steps[i]
                    step_num_updates = denoised_steps_num_updates[i]
                    step_patch = patch_steps[i]
                    denoised_steps[i], denoised_steps_num_updates[i] = update_prediction_noisy_batches(
                        step_denoised, step_num_updates, step_patch, patch_idxs, patch_cut_idxs
                    )
        else:
            patch_xyz = (
                furthest_point_sample(torch.from_numpy(patch_xyz).float().cuda().transpose(1, 2), patch_size)
                .transpose(1, 2)
                .cpu()
                .numpy()
            )
            denoised.append(patch_xyz)

    logger.info("Accumulating Patches")

    if args.average_predictions:
        # count number of points that did not get updated
        non_updated_idxs = np.where(denoised_num_updates == 0)[0]

        if len(non_updated_idxs) > 0:
            logger.warning("There are {} points that did not get updated.", len(non_updated_idxs))
            # assign random points to non updated points
            random_idxs = np.random.choice(len(denoised), len(non_updated_idxs))
            denoised[non_updated_idxs] = denoised[random_idxs]

        if return_steps:
            for i in range(len(denoised_steps)):
                step_non_updated_idxs = np.where(denoised_steps_num_updates[i] == 0)[0]
                if len(step_non_updated_idxs) > 0:
                    random_idxs = np.random.choice(len(denoised_steps[i]), len(step_non_updated_idxs))
                    denoised_steps[i][step_non_updated_idxs] = denoised_steps[i][random_idxs]
    else:
        # accumulate and fps
        denoised = np.concatenate(denoised, axis=0).reshape(-1, 3)
        idxs = fpsample.bucket_fps_kdline_sampling(denoised, len(room_points), h=7)
        denoised = denoised[idxs]

    prediction.points = o3d.utility.Vector3dVector(denoised)
    if room_colors is not None:
        prediction.colors = o3d.utility.Vector3dVector(room_colors)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_point_cloud(out_path, prediction)

    if return_steps:
        for i in range(len(denoised_steps)):
            prediction.points = o3d.utility.Vector3dVector(denoised_steps[i])
            o3d.io.write_point_cloud(f"{out_path.split('.')[0]}_step_{i}.ply", prediction)


if __name__ == "__main__":
    main()
