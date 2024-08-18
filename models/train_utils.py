import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor, nn


class DiffusionModel(ABC, nn.Module):
    """Abstract class for diffusion models."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        x0: Tensor,
        x1: Optional[Tensor] = None,
        x_cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, *args, **kwargs) -> Tensor:
        """Sample from the model."""
        raise NotImplementedError()

    def multi_gpu_wrapper(self, f: callable):
        """Wrap the model for multi-GPU training."""
        self.model = f(self.model)

    def train(self):
        """Set the model to training mode."""
        self.model.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()


def exists(x: Any) -> bool:
    """
    Check if the input variable exists.

    Args:
        x (Any): The input variable.

    Returns:
        bool: True if the variable exists, otherwise False.
    """
    return x is not None


def default(val: Any, d: Any) -> Any:
    """
    Set the default value for a variable.

    Args:
        val: The value to check.
        d: The default value.

    Returns:
        Any: The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def set_seed(opt: DictConfig) -> None:
    """
    Set the random seed for the experiment.

    Args:
        opt (DictConfig): The configuration dictionary.
    """
    if opt.training.seed is None:
        opt.training.seed = 42

    # different seed per gpu
    if "global_rank" not in opt:
        opt.global_rank = 0
    opt.training.seed += opt.global_rank

    logger.info("Random Seed: {}", opt.training.seed)
    random.seed(opt.training.seed)
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)


def to_cuda(data: Union[list, tuple, dict], device: Union[str, torch.device]) -> Union[Tensor, List, Tuple, Dict, None]:
    """
    Move the input data to the specified device.

    Args:
        data (Tensor): The input data.
        device (Union[str, torch.device]): The device to move the data to.

    Returns:
        Union[Tensor, List, Tuple, Dict, None]: The data moved to the specified device.
    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return [to_cuda(d, device) for d in data]
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    if device is None:
        return data.cuda(non_blocking=True)
    else:
        return data.to(device, non_blocking=True)


def ensure_size(x: Tensor) -> Tensor:
    """
    Ensure the size of the input tensor is correct (B D N).

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with the correct size.

    Raises:
        AssertionError: If the input tensor does not have the correct dimensions.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    assert x.dim() == 3
    if x.size(1) > x.size(2):
        x = x.transpose(1, 2)
    return x


def get_data_batch(batch: Dict, cfg: Dict, align_fn=None) -> Dict[str, Tensor]:
    """
    Process a batch of data and return a dictionary containing the processed data.

    Args:
        batch (Dict): The input batch of data.
        cfg (Dict): The configuration dictionary.
        align_fn (Optional): The alignment function to align the data on the fly.

    Returns:
        Dict[str, Tensor]: A dictionary containing the processed data with the following keys:
            - "x_gt": The ground truth clean points.
            - "x_start": The noisy points.
            - "x_cond": The conditional features.
    """
    if cfg.data.dataset == "PUNet":
        clean_points = batch["clean_points"].squeeze()
        noisy_points = batch["noisy_points"].squeeze()
        clean_features = lr_features = None
    else:
        clean_points = batch["clean_points"].transpose(1, 2)

        if not cfg.data.unconditional or cfg.evaluate_uncond_on_iphone:
            lr_features = batch["noisy_features"] if "noisy_features" in batch else None
            noisy_points = batch["noisy_points"] if "noisy_points" in batch else None
            clean_features = batch["clean_features"] if "clean_features" in batch else None
        else:
            lr_features, noisy_points, clean_features = None, None, None

    clean_points = ensure_size(clean_points)
    clean_features = ensure_size(clean_features) if clean_features is not None else None
    lr_features = ensure_size(lr_features) if lr_features is not None else None
    noisy_points = ensure_size(noisy_points) if noisy_points is not None else None
    noisy_colors = ensure_size(batch["noisy_colors"]) if "noisy_colors" in batch else None
    clean_colors = ensure_size(batch["clean_colors"]) if "clean_colors" in batch else None

    # if we are training on PUNet we need to align the data on the fly
    if cfg.data.dataset == "PUNet" and align_fn is not None:
        clean_points = align_fn(noisy_points, clean_points)

    # concatenate colors to features
    if noisy_colors is not None and noisy_colors.shape[-1] > 0 and cfg.data.use_rgb_features:
        lr_features = torch.cat([noisy_colors, lr_features], dim=1) if lr_features is not None else noisy_colors

    if clean_colors is not None and clean_colors.shape[-1] > 0 and cfg.data.use_rgb_features:
        clean_features = (
            torch.cat([clean_colors, clean_features], dim=1) if clean_features is not None else clean_colors
        )

    return {"x_gt": clean_points, "x_start": noisy_points, "x_cond": lr_features}


@torch.no_grad()
def getGradNorm(net: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        net (torch.nn.Module): The network.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The parameter norm and the gradient norm.
    """
    pNorm = torch.sqrt(sum(torch.sum(p**2) for p in net.parameters() if p.requires_grad and p is not None))
    gradNorm = torch.sqrt(
        sum(torch.sum(p.grad**2) for p in net.parameters() if p.requires_grad and p is not None and p.grad is not None)
    )
    return pNorm, gradNorm


def setup_output_subdirs(output_dir: str, *subfolders: List[str] | str) -> List[str]:
    """
    Create the output subdirectories.

    Args:
        output_dir (str): The output directory.
        subfolders (List[str] | str): The subfolders to create.

    Returns:
        List[str]: A list containing the paths to the created subfolders.
    """
    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    subfolder_list = []
    for sf in subfolders:
        curr_subf = os.path.join(output_subdirs, sf)
        try:
            os.makedirs(curr_subf)
        except OSError:
            pass
        subfolder_list.append(curr_subf)

    return subfolder_list
