from typing import List, Tuple

import numpy as np
import pytorch3d.loss
import torch
from einops import rearrange
from numpy.typing import ArrayLike
from point_cloud_utils import chamfer_distance
from torch import Tensor

from metrics.chamfer3D.dist_chamfer_3D import chamfer_dist_nograd

from .chamfer3D.dist_chamfer_3D import chamfer_dist_nograd
from .emd_assignment.emd_module import emdModule
from .p2m import point_mesh_face_distance_custom
from .PyTorchEMD.emd_nograd import earth_mover_distance_nograd


def calculate_cd(pred: Tensor, gt: Tensor) -> List[float]:
    """
    Calculate the Chamfer distance between two point clouds.

    Args:
        pred (Tensor): The predicted point cloud.
        gt (Tensor): The ground truth point cloud.

    Returns:
        List[float]: The Chamfer distance between the two point clouds.
    """
    cds = []

    # handle shape
    assert (
        pred.shape == gt.shape
    ), f"CD calculation asserts same shape but pred shape: {pred.shape}, gt shape: {gt.shape}"
    if pred.shape[2] > pred.shape[1]:
        pred = rearrange(pred, "b c n -> b n c")
        gt = rearrange(gt, "b c n -> b n c")

    # move to cpu
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    for x_pred, x_gt in zip(pred, gt):
        cd = chamfer_distance(
            x_pred,
            x_gt,
        )
        cds.append(cd)

    return cds


def calculate_cd_cuda(pred: Tensor, gt: Tensor) -> List[float]:
    """
    Calculate the Chamfer distance between two point clouds.

    Args:
        pred (Tensor): The predicted point cloud.
        gt (Tensor): The ground truth point cloud.

    Returns:
        List[float]: The Chamfer distance between the two point clouds.
    """
    # make sure that last diemtnsion is 3
    if pred.shape[-1] != 3:
        pred = pred.transpose(-1, -2)
        gt = gt.transpose(-1, -2)

    pred = torch.split(pred, 4, dim=0)
    gt = torch.split(gt, 4, dim=0)
    cds = []

    for p, g in zip(pred, gt):
        dl, dr = chamfer_dist_nograd(p, g)
        cd = dl.mean(dim=1) + dr.mean(dim=1)
        cd = cd.cpu()
        cds.append(cd)

    cds = torch.cat(cds).cpu().tolist()
    return cds


def calculate_emd_cuda(pred: Tensor, gt: Tensor) -> List[float]:
    """
    Calculate the Earth Mover's Distance between two point clouds.

    Args:
        pred (Tensor): The predicted point cloud.
        gt (Tensor): The ground truth point cloud.

    Returns:
        List[float]: The Earth Mover's Distance between the two point clouds.
    """
    # create batches of size 4 to avoid OOM
    pred = torch.split(pred, 4, dim=0)
    gt = torch.split(gt, 4, dim=0)
    emds = []

    for p, g in zip(pred, gt):
        emd = earth_mover_distance_nograd(p, g, transpose=p.shape[-1] > p.shape[-2])
        emd = emd.cpu().numpy()
        emd = np.mean(emd)
        emds.append(emd)

    return emds


def calculate_emd_exact_cuda(pred: Tensor, gt: Tensor) -> List[float]:
    """
    Calculate the Earth Mover's Distance between two point clouds.

    Args:
        pred (Tensor): The predicted point cloud.
        gt (Tensor): The ground truth point cloud.

    Returns:
        List[float]: The Earth Mover's Distance between the two point clouds.
    """
    # create batches of size 4 to avoid OOM
    pred = torch.split(pred, 4, dim=0)
    gt = torch.split(gt, 4, dim=0)
    emds = []

    emd_function = emdModule()

    for p, g in zip(pred, gt):
        with torch.no_grad():
            dis, _ = emd_function(p, g, eps=0.001, iters=10000)
            dis = dis.detach()
            emd = torch.mean(dis, dim=1).sqrt().cpu().tolist()
        emds.extend(emd)

    return emds


def normalize_sphere(pc: Tensor, radius: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Normalize a point cloud to the unit sphere.

    Args:
        pc (Tensor): The point cloud to normalize.
        radius (float): The radius of the sphere.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The normalized point cloud, center, and scale.
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


def normalize_pcl(pc: Tensor | ArrayLike, center: Tensor | ArrayLike, scale: float | Tensor) -> Tensor | ArrayLike:
    """
    Normalize a point cloud.

    Args:
        pc (Tensor | ArrayLike): The point cloud to normalize.
        center (Tensor | ArrayLike): The center of the point cloud.
        scale (float | Tensor): The scale of the point cloud.

    Returns:
        Tensor | ArrayLike: The normalized point cloud.
    """
    return (pc - center) / scale


@torch.no_grad()
def cd_unit_sphere(gen: Tensor, ref: Tensor, normalize: bool = True) -> Tuple[float, float]:
    """
    Calculate the Chamfer distance between two point clouds.

    Args:
        gen (Tensor): The generated point cloud.
        ref (Tensor): The reference point cloud.
        normalize (bool): Whether to normalize the point clouds.

    Returns:
        Tuple[float, float]: The Chamfer distance between the two point clouds.
    """
    if normalize:
        ref, center, scale = normalize_sphere(ref)
        gen = normalize_pcl(gen, center, scale)
    cd1, cd2 = chamfer_dist_nograd(gen, ref)
    cd1 = cd1.mean().item()
    cd2 = cd2.mean().item()
    return cd1, cd2


@torch.no_grad()
def point_face_dist(pcl: Tensor, verts: Tensor, faces: Tensor, normalize: bool = True) -> Tuple[float, float]:
    """
    Calculate the point-face distance between a point cloud and a mesh.
    Args:
        pcl (Tensor): The point cloud.
        verts (Tensor): The vertices of the mesh.
        faces (Tensor): The faces of the mesh.
        normalize (bool): Whether to normalize the point cloud and mesh.
    Returns:
        Tuple[float, float]: The point-face distance between the point cloud and mesh.
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, "Batch is not supported."

    if normalize:
        # Normalize mesh
        verts, center, scale = normalize_sphere(verts.unsqueeze(0))
        verts = verts[0]
        # Normalize pcl
        pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
        pcl = pcl[0]

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl]).cuda()
    meshes = pytorch3d.structures.Meshes([verts], [faces]).cuda()

    point_dist, face_dist = point_mesh_face_distance_custom(meshes, pcls)

    return point_dist.item(), face_dist.item()
