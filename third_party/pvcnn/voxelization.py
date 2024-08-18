import pvcnn.functional as F
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

__all__ = ["Voxelization"]


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    @custom_bwd
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = (
                norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps)
                + 0.5
            )
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return "resolution={}{}".format(self.r, ", normalized eps = {}".format(self.eps) if self.normalize else "")
