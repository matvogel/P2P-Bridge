import argparse
import os
from typing import Any, Generator

import numpy as np
import omegaconf
import open3d as o3d
import pytorch3d.ops
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig

from models.evaluation import Evaluator, farthest_point_sampling
from models.model_loader import load_diffusion
from models.train_utils import set_seed
from utils.utils import NormalizeUnitSphere, write_array_to_xyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the point cloud.")
    parser.add_argument("--save_path", type=str, required=True, help="Output root directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=3, help="Patch oversampling factor.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for prediction.")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU to use.")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for the diffusion.")
    parser.add_argument("--distribution_type", default="none")
    args = parser.parse_args()

    # load config from checkpoint
    cfg_path = os.path.join(os.path.dirname(args.model_path), "opt.yaml")
    cfg = omegaconf.OmegaConf.load(cfg_path)

    # merge with args
    cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(vars(args)))

    # set some additional parameters
    cfg.restart = False
    cfg.local_rank = 0
    return cfg


def input_iter(input_dir: str) -> Generator[Any, Any, Any]:
    """
    Iterate over the input directory and yield the point cloud data.

    Args:
        input_dir (str): The input directory.

    Yields:
        Any: The point cloud data.
    """

    for fn in os.listdir(input_dir):
        if fn[-3:] != "xyz":
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {"pcl_noisy": pcl_noisy, "name": fn[:-4], "center": center, "scale": scale}


@torch.no_grad()
def patch_based_denoise(
    model: torch.nn.Module,
    pcl_noisy: torch.Tensor,
    patch_size: int,
    seed_k: int = 3,
    cfg: omegaconf.DictConfig = None,
    save_intermediate: bool = False,
):
    """

    Args:
        model (torch.nn.Module): The model to use for denoising.
        pcl_noisy (torch.Tensor): The noisy point cloud.
        patch_size (int): The size of the patches.
        seed_k (int): The oversampling factor for the seed points.
        cfg (omegaconf.DictConfig): The configuration dictionary.
        save_intermediate (bool): Whether to save intermediate steps.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: The denoised point cloud and the intermediate steps.

    """
    assert pcl_noisy.dim() == 2, "The shape of input point cloud must be (N, 3)."
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (N, K, 3)

    model.eval()

    # center and scale the patches
    centers = patches.mean(dim=1, keepdim=True)
    patches = patches - centers
    scale = torch.max(torch.norm(patches, dim=-1))
    patches = patches / scale
    out = model.sample(
        x_start=patches.transpose(1, 2), use_ema=cfg.use_ema, steps=cfg.steps, log_count=cfg.steps, verbose=True
    )

    patches_denoised = out["x_pred"].transpose(1, 2)
    patches_steps = out["x_chain"].transpose(-2, -1)

    patches_denoised = patches_denoised * scale + centers
    patches_steps = patches_steps * scale + centers.unsqueeze(1)
    patches_steps = patches_steps.transpose(1, 0)

    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.reshape(1, -1, d), N)
    pcl_denoised = pcl_denoised[0].squeeze()

    if save_intermediate:
        T, B, n, d = patches_steps.size()
        patches_steps = patches_steps.reshape(T, B * n, d)
        pcl_steps_denoised, _ = farthest_point_sampling(patches_steps, N)
    else:
        pcl_steps_denoised = None

    return pcl_denoised, pcl_steps_denoised


@torch.no_grad()
def sample(
    cfg: DictConfig,
) -> None:
    """
    Sample from the model.

    Args:
        cfg (DictConfig): The configuration dictionary.
    """
    set_seed(cfg)
    torch.cuda.manual_seed_all(cfg.training.seed)
    torch.backends.cudnn.benchmark = True

    model, _ = load_diffusion(cfg)
    model.eval()

    # load the data and normalize it
    if cfg.data_path[-3:] == "ply":
        pcl_noisy = o3d.io.read_point_cloud(cfg.data_path)
        pcl_noisy = torch.FloatTensor(np.asarray(pcl_noisy.points))
    else:
        pcl_noisy = torch.FloatTensor(np.loadtxt(cfg.data_path))

    pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
    data = {"pcl_noisy": pcl_noisy, "center": center, "scale": scale}

    pcl_noisy = data["pcl_noisy"].cuda()
    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        pcl_next, pcl_next_steps = patch_based_denoise(
            model=model,
            pcl_noisy=pcl_next,
            patch_size=2048,
            seed_k=cfg.k,
            cfg=cfg,
        )

    pcl_denoised = pcl_next.cpu()
    pcl_denoised = pcl_denoised * data["scale"] + data["center"]

    if cfg.data_path[-3:] == "xyz":
        write_array_to_xyz(cfg.save_path, pcl_denoised.numpy())
    else:
        raise NotImplementedError("Only .xyz files are supported for now.")


def main():
    opt = parse_args()
    sample(opt)


if __name__ == "__main__":
    main()
