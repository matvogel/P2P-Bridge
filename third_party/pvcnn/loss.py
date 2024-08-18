import pvcnn.functional as F
import torch.nn as nn

__all__ = ["KLLoss"]


class KLLoss(nn.Module):
    def forward(self, x, y):
        return F.kl_loss(x, y)
