import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from third_party.openpoints.cpp import pointnet2_cuda


class AvgVoxelization(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int()[:, :3].contiguous()
        b, c, _ = features.shape
        out, indices, counts = pointnet2_cuda.avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = pointnet2_cuda.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


avg_voxelize = AvgVoxelization.apply
