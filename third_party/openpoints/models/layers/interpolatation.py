import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from third_party.openpoints.cpp import pointnet2_cuda


class NeighborInterpolation(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, points_coords, centers_coords, centers_features):
        """
        :param ctx:
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param centers_features: features of centers, FloatTensor[B, C, M]
        :return:
            points_features: features of points, FloatTensor[B, C, N]
        """
        centers_coords = centers_coords[:, :3].contiguous()
        points_coords = points_coords[:, :3].contiguous()
        centers_features = centers_features.contiguous()
        (
            points_features,
            indices,
            weights,
        ) = pointnet2_cuda.three_nearest_neighbors_interpolate_forward(points_coords, centers_coords, centers_features)
        ctx.save_for_backward(indices, weights)
        ctx.num_centers = centers_coords.size(-1)
        return points_features

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        grad_centers_features = pointnet2_cuda.three_nearest_neighbors_interpolate_backward(
            grad_output.contiguous(), indices, weights, ctx.num_centers
        )
        return None, None, grad_centers_features


nearest_neighbor_interpolate = NeighborInterpolation.apply
