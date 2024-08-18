from typing import Literal

import torch
from einops import reduce
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss

from metrics.emd_assignment import emd_module as EMD


def mean_squared_error(pred: Tensor, gt: Tensor) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


def mean_squared_error_sum(pred: Tensor, gt: Tensor) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "sum")
    return loss


def l1(pred: Tensor, gt: Tensor) -> Tensor:
    loss = l1_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


class EmdLoss:
    def __init__(self):
        self.emd = EMD.emdModule()

    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        if pred.shape[-1] != 3:
            pred = pred.transpose(1, 2)
        if gt.shape[-1] != 3:
            gt = gt.transpose(1, 2)

        distances, _ = self.emd(pred, gt, eps=0.005, iters=50)

        loss = torch.sqrt(distances)
        loss = reduce(loss, "b ... -> b", "mean")
        return loss


def get_loss(type: Literal["mse", "mse_sum", "l1", "emd"]) -> callable:
    """

    Args:
        type (Literal["mse", "mse_sum", "l1", "emd"]): The type of loss to get.

    Returns:
        callable: The loss function.
    """
    if type == "mse":
        return mean_squared_error
    if type == "mse_sum":
        return mean_squared_error_sum
    if type == "l1":
        return l1
    if type == "emd":
        return EmdLoss()
