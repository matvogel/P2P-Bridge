import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import pytorch3d
import torch
from loguru import logger
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import DataLoader
from torch_cluster import fps
from tqdm import tqdm

import wandb
from metrics.metrics import calculate_cd, calculate_cd_cuda, calculate_emd_cuda, calculate_emd_exact_cuda
from models.train_utils import get_data_batch
from utils.visualize import visualize_pointcloud_batch


def save_visualizations(items: List[Tuple[Tensor, str]], out_dir: str, step: int) -> None:
    """
    Save visualizations of the given items.

    Args:
        items (List[Tuple[Tensor, str]]): List of items to visualize.
        out_dir (str): Output directory to save the visualizations.
        step (int): Step number.
    """
    for item in items:
        if item is None:
            continue
        ptc, name = item

        # make 3 last channel
        if ptc.shape[-1] > ptc.shape[-2]:
            ptc = ptc.transpose(1, 2)

        visualize_pointcloud_batch(
            "%s/%03d_%s.png" % (out_dir, step, name),
            ptc,
        )


def log_wandb(name: str, out_dir: str, step: int) -> None:
    """
    Log the visualization to wandb.

    Args:
        name (str): Name of the visualization.
        out_dir (str): Output directory to save the visualization.
        step (int): Step number.
    """
    out_path = "%s/%03d_%s.png" % (out_dir, step, name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        return
    wandb_img = wandb.Image(out_path)
    wandb.log({name: wandb_img}, step=step)


def save_ptc(name: str, ptc: Tensor, out_dir: str, step: int) -> None:
    """
    Save the point cloud to a numpy file.

    Args:
        name (str): Name of the point cloud.
        ptc (Tensor): Point cloud tensor.
        out_dir (str): Output directory to save the point cloud.
        step (int): Step number.
    """
    np.save("%s/%03d_%s.npy" % (out_dir, step, name), ptc.cpu().numpy())


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: DictConfig,
    step: int,
    sampling: bool = False,
    save_npy: bool = False,
    fast: bool = False,
) -> dict:
    """
    Evaluate the model on the given validation loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): The validation loader.
        cfg (DictConfig): Configuration dictionary.
        step (int): Step number.
        sampling (bool): Whether to sample from the model.
        save_npy (bool): Whether to save the point clouds as numpy files.
        fast (bool): Whether to use the fast evaluation method.

    Returns:
        dict: Evaluation metrics.
    """
    if sampling:
        out_dir = cfg.out_sampling
    else:
        out_dir = cfg.outf_syn

    sample_data = {
        "x_pred": torch.tensor([]).cuda(),
        "x_chain": None,
        "x_start": torch.tensor([]).cuda(),
        "x_cond": torch.tensor([]).cuda(),
        "x_gt": torch.tensor([]).cuda(),
    }

    accum_iter = cfg.sampling.accum_iter if "accum_iter" in cfg.sampling else 1

    for idx, eval_data in enumerate(val_loader):
        data_batch = get_data_batch(batch=eval_data, cfg=cfg)
        x_gt = data_batch["x_gt"]
        x_cond = data_batch["x_cond"]
        x_start = data_batch["x_start"]

        x_gt = x_gt.cuda() if x_gt is not None else None
        x_cond = x_cond.cuda() if x_cond is not None else None
        x_start = x_start.cuda() if x_start is not None else None

        with torch.no_grad():
            model_out = model.sample(
                x_start=x_start,
                x_cond=x_cond,
                clip=cfg.diffusion.get("clip", False),
                use_ema=cfg.get("use_ema", False),
            )

        sample_data["x_pred"] = torch.cat([sample_data["x_pred"], model_out["x_pred"]], dim=0)
        if sample_data["x_chain"] is not None:
            sample_data["x_chain"] = model_out["x_chain"][0]
        sample_data["x_start"] = torch.cat([sample_data["x_start"], model_out["x_start"]], dim=0)
        sample_data["x_gt"] = torch.cat([sample_data["x_gt"], x_gt], dim=0)
        sample_data["x_cond"] = (
            torch.cat([sample_data["x_cond"], x_cond[:, :3, :]], dim=0) if cfg.model.type == "PVDCond" else None
        )

        if idx >= accum_iter - 1:
            break

    pred = sample_data["x_pred"]
    chain = sample_data["x_chain"]
    x_start = sample_data["x_start"]
    x_cond = sample_data["x_cond"]
    x_gt = sample_data["x_gt"]

    # visualize the pointclouds
    save_visualizations(
        [
            (pred, "pred"),
            (x_start, "noisy") if x_start is not None else None,
            (x_gt, "gt"),
            (chain, "chain") if chain is not None else None,
            (x_cond, "cond") if x_cond is not None else None,
        ],
        out_dir,
        step,
    )

    # calculate stats
    # subsample cloud to closest multiple of 128 (for CUDA evaluation)
    n_points = pred.shape[-1]
    n_points = n_points - n_points % 128

    pred = pred[..., :n_points]
    x_gt = x_gt[..., :n_points]
    x_start = x_start[..., :n_points] if x_start is not None else None
    x_cond = x_cond[..., :n_points] if x_cond is not None else None

    cd, emd, eval_loss = get_metrics(x_gt, pred, model=model, fast=fast)

    batch_metrics = {
        "cd": cd,
        "emd": emd,
        "mse": eval_loss,
    }

    if x_start is not None:
        cd_hint, emd_hint, eval_loss_hint = get_metrics(x_gt, x_start, model=model, fast=fast)
        batch_metrics["cd_noisy"] = cd_hint
        batch_metrics["emd_noisy"] = emd_hint
        batch_metrics["mse_noisy"] = eval_loss_hint

    if x_cond is not None:
        cd_cond, emd_cond, eval_loss_cond = get_metrics(x_gt, x_cond, model=model, fast=fast)
        batch_metrics["cd_cond"] = cd_cond
        batch_metrics["emd_cond"] = emd_cond
        batch_metrics["mse_cond"] = eval_loss_cond

    logger.info(batch_metrics)

    if not sampling:
        wandb.log(batch_metrics, step=step)
        log_wandb("pred", out_dir, step)
        log_wandb("noisy", out_dir, step)
        log_wandb("gt", out_dir, step)
        log_wandb("cond", out_dir, step)

    elif save_npy:
        save_ptc("pred", pred, out_dir, step)
        save_ptc("noisy", x_start, out_dir, step)
        save_ptc("gt", x_gt, out_dir, step)
        if x_cond is not None:
            save_ptc("cond", x_cond, out_dir, step)

    return batch_metrics


def get_metrics(
    gt: Tensor, pred: Tensor, model: Optional[torch.nn.Module] = None, fast: bool = True
) -> Tuple[float, float, float]:
    """
    Calculate evaluation metrics for the given model predictions.

    Args:
        gt (Tensor): Ground truth point cloud.
        pred (Tensor): Predicted point cloud.
        model (Optional[torch.nn.Module]): Model to evaluate.
        fast (bool): Whether to use the fast evaluation method.

    Returns:
        Tuple[float, float, float]: Chamfer distance, Earth Mover's distance, and evaluation loss.
    """
    if pred.shape[-1] < pred.shape[-2]:
        pred = pred.transpose(1, 2)
    if gt.shape[-1] < gt.shape[-2]:
        gt = gt.transpose(1, 2)

    if fast:
        cd = np.mean(calculate_cd_cuda(pred, gt)) * 1000
        eval_loss = np.mean(model.loss(pred, gt).cpu().numpy()) if model is not None else 0
        emd = np.mean(calculate_emd_cuda(pred, gt)) * 1000
    else:
        # make sure that pred and gt are divisable by 128
        n_points = pred.shape[-1]
        n_points = n_points - n_points % 128

        pred = pred[..., :n_points]
        gt = gt[..., :n_points]

        gt = gt.transpose(1, 2).contiguous()
        pred = pred.transpose(1, 2).contiguous()

        cd = np.mean(calculate_cd(pred, gt)) * 1000
        eval_loss = np.mean(model.loss(pred, gt).detach().cpu().numpy()) if model is not None else 0
        emd = np.mean(calculate_emd_exact_cuda(pred, gt)) * 1000
    return cd, emd, eval_loss


"""Evaluation modules from score denoise https://github.com/luost26/score-denoise"""


def load_xyz(xyz_dir):
    all_pcls = {}
    for fn in tqdm(os.listdir(xyz_dir), desc=f"Loading {xyz_dir}"):
        if fn[-3:] != "xyz":
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls


def load_off(off_dir):
    all_meshes = {}
    for fn in tqdm(os.listdir(off_dir), desc=f"Loading {off_dir}"):
        if fn[-3:] != "off":
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces = pcu.load_mesh_vf(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {"verts": verts, "faces": faces}
    return all_meshes


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center


def chamfer_distance_unit_sphere(gen, ref, batch_reduction="mean", point_reduction="mean"):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)


def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i : i + 1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices


def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2  # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc**2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale


def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, "Batch is not supported."

    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls, min_triangle_area=0.0)


class Evaluator(object):
    def __init__(
        self,
        output_pcl_dir,
        dataset_root,
        dataset,
        summary_dir,
        experiment_name,
        device="cuda",
        res_gts="8192_poisson",
    ):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.summary_dir = summary_dir
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, "pointclouds", "test", res_gts)
        self.gts_mesh_dir = os.path.join(dataset_root, dataset, "meshes", "test")
        self.res_gts = res_gts
        self.device = device
        self.load_data()

    def load_data(self):
        self.pcls_up = load_xyz(self.output_pcl_dir)
        self.pcls_high = load_xyz(self.gts_pcl_dir)
        self.meshes = load_off(self.gts_mesh_dir)
        self.pcls_name = list(self.pcls_up.keys())

    def run(self):
        pcls_up, pcls_high, pcls_name = self.pcls_up, self.pcls_high, self.pcls_name
        results = {}

        for name in tqdm(pcls_name, desc="Evaluate"):
            pcl_up = pcls_up[name]
            if pcl_up.dim() != 2:
                continue

            pcl_up = pcl_up[:, :3].unsqueeze(0).to(self.device)

            if name not in pcls_high:
                self.logger.warning("Shape `%s` not found, ignored." % name)
                continue
            pcl_high = pcls_high[name].unsqueeze(0).to(self.device)
            verts = self.meshes[name]["verts"].to(self.device)
            faces = self.meshes[name]["faces"].to(self.device)

            cd_sph = chamfer_distance_unit_sphere(pcl_up, pcl_high)[0].item()

            if "blensor" in self.experiment_name:
                rotmat = torch.FloatTensor(Rotation.from_euler("xyz", [-90, 0, 0], degrees=True).as_matrix()).to(
                    pcl_up[0]
                )
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_up[0].matmul(rotmat.t()), verts=verts, faces=faces
                ).item()
            else:
                p2f = point_mesh_bidir_distance_single_unit_sphere(pcl=pcl_up[0], verts=verts, faces=faces).item()

            results[name] = {
                "cd_sph": cd_sph,
                "p2f": p2f,
            }

        results = pd.DataFrame(results).transpose()
        res_mean = results.mean(axis=0)
        logger.info("\n" + repr(results))
        logger.info("\nMean\n" + "\n".join(["%s\t%.12f" % (k, v) for k, v in res_mean.items()]))

        update_summary(
            os.path.join(self.summary_dir, "Summary_%s.csv" % self.dataset),
            model=self.experiment_name,
            metrics={
                # 'cd(mean)': res_mean['cd'],
                "cd_sph(mean)": res_mean["cd_sph"],
                "p2f(mean)": res_mean["p2f"],
                # 'hd_sph(mean)': res_mean['hd_sph'],
            },
        )


def update_summary(path, model, metrics):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, sep="\s*,\s*", engine="python")
    else:
        df = pd.DataFrame()
    for metric, value in metrics.items():
        setting = metric
        if setting not in df.columns:
            df[setting] = np.nan
        df.loc[model, setting] = value
    df.to_csv(path, float_format="%.12f")
    return df
