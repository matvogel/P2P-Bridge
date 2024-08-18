import functools
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from models.modules import AdaGN, LinearAttention, SE3d, Swish
from third_party.openpoints.models.layers import (
    avg_voxelize,
    ball_query,
    furthest_point_sample_pvcnn,
    nearest_neighbor_interpolate,
    pvcnn_grouping,
    trilinear_devoxelize,
)


@dataclass
class PVCData:
    features: torch.Tensor
    coords: torch.Tensor = None
    cond_coords: torch.Tensor = None
    cond_features: torch.Tensor = None
    lower_coords: torch.Tensor = None
    lower_features: torch.Tensor = None
    time_emb: torch.Tensor = None
    cond: any = None


def create_pvc_layer_params(
    npoints: int,
    channels: List,
    n_sa_blocks: List[int],
    n_fp_blocks: List[int],
    radius: List[float],
    voxel_resolutions: List[float],
    downsample_factor: int = 4,
    centers: List[int] = None,
):
    n_centers = []
    sa_blocks = []
    fp_blocks = []
    n_channels = len(channels)

    for i in range(n_channels - 1):
        n_centers.append(npoints // downsample_factor ** (i + 1))
        # create set abstraction blocks
        if i != n_channels - 2:
            sa_blocks.append(
                [
                    [channels[i], n_sa_blocks[i], voxel_resolutions[i]],  # conv config
                    [
                        n_centers[i] if centers is None else centers[i],
                        radius[i],
                        32,
                        [channels[i], channels[i + 1]],
                    ],  # sa config
                ]
            )
        else:
            sa_blocks.append(
                [
                    None,
                    [
                        n_centers[i] if centers is None else centers[i],
                        radius[i],
                        32,
                        [channels[i], channels[i], channels[i + 1]],
                    ],
                ]
            )

    # in_channels, out_channels X | out_channels, num_blocks, voxel_resolution
    fp_blocks = [
        [
            [channels[3], channels[3]],
            [channels[3], n_fp_blocks[3], voxel_resolutions[3]],
        ],
        [
            [channels[3], channels[3]],
            [channels[3], n_fp_blocks[2], voxel_resolutions[2]],
        ],
        [
            [channels[3], channels[2]],
            [channels[2], n_fp_blocks[1], voxel_resolutions[1]],
        ],
        [
            [channels[2], channels[2], channels[1]],
            [channels[1], n_fp_blocks[0], voxel_resolutions[0]],
        ],
    ]
    return sa_blocks, fp_blocks


class BallQuery(nn.Module):
    def __init__(self, radius: float, num_neighbors: int, include_coordinates: bool = True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    @custom_bwd
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, points_coords: Tensor, centers_coords: Tensor, points_features: Optional[Tensor] = None):
        # input: BCN, BCN
        # neighbor_features: B,D(+3),Ncenter
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = pvcnn_grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)

        if points_features is None:
            assert self.include_coordinates, "No Features For Grouping"
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = pvcnn_grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features

    def extra_repr(self):
        return "radius={}, num_neighbors={}{}".format(
            self.radius,
            self.num_neighbors,
            ", include coordinates" if self.include_coordinates else "",
        )


class ScaleShift(nn.Module):
    def __init__(self, time_emb_dim: int, dim_out: int):
        super().__init__()
        self.mlp = nn.Linear(time_emb_dim, dim_out * 2)

        # initialize weights such that scale is 1 and shift is 0
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()

    def forward(self, data: PVCData) -> PVCData:
        time_emb = data.time_emb
        x = data.features

        time_emb = self.mlp(time_emb)

        # Add the necessary number of singleton dimensions to time_emb
        for _ in range(x.ndim - time_emb.ndim):
            time_emb = time_emb.unsqueeze(-1)

        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        data.features = x
        return data


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        dim: int = 1,
        gn_groups: int = 8,
        cond_dim: int = 0,
        affine: bool = True,
    ):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
        else:
            conv = nn.Conv2d

        # additional conditioning
        if cond_dim > 0:
            bn = functools.partial(AdaGN, ctx_dim=cond_dim, num_groups=gn_groups, ndim=dim)
        else:
            bn = functools.partial(torch.nn.GroupNorm, num_groups=gn_groups, affine=affine)

        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for layer_idx, oc in enumerate(out_channels):
            layers.append(conv(in_channels, oc, 1))
            layers.append(bn(num_channels=oc))
            layers.append(Swish())
            in_channels = oc
        self.layers = nn.ModuleList(layers)

    def forward(self, data: PVCData) -> PVCData:
        features = data.features
        cond = data.cond

        for l in self.layers:
            if isinstance(l, AdaGN) and cond is not None:
                features = l(features, cond)
            else:
                features = l(features)

        data.features = features
        return data


class Voxelization(nn.Module):
    def __init__(self, resolution: int, normalize: bool = True, eps: float = 0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features: Tensor, coords: Tensor) -> Tuple[Tensor, Tensor]:
        # features: B,D,N
        # coords:   B,3,N
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
        if features is None:
            return features, norm_coords
        return avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return "resolution={}{}".format(self.r, ", normalized eps = {}".format(self.eps) if self.normalize else "")


class PVConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        resolution: int,
        normalize: bool = True,
        eps: float = 0,
        with_se: bool = True,
        add_point_feat: bool = True,
        attention: bool = False,
        attention_fn: torch.nn.Module = LinearAttention,
        dropout: float = 0.1,
        gn_groups: int = 8,
        cond_dim: int = 0,
        affine: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # For each PVConv we use (Conv3d, GroupNorm(8), Swish, dropout, Conv3d, GroupNorm(8), Attention)
        if cond_dim > 0:
            NormLayer = functools.partial(AdaGN, ctx_dim=cond_dim, num_groups=gn_groups, ndim=3)
        else:
            NormLayer = functools.partial(torch.nn.GroupNorm, gn_groups, affine=affine)

        voxel_layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            NormLayer(out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            NormLayer(out_channels),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))

        self.voxel_layers = nn.ModuleList(voxel_layers)

        if attention:
            self.attn = attention_fn(out_channels)
        else:
            self.attn = None

        if add_point_feat:
            self.point_features = SharedMLP(
                in_channels,
                out_channels,
                gn_groups=gn_groups,
                cond_dim=cond_dim,
                affine=affine,
            )

        self.add_point_feat = add_point_feat

    def forward(self, data: PVCData) -> PVCData:
        coords = data.coords
        features = data.features
        cond = data.cond

        assert features.shape[0] == coords.shape[0], f"get feat: {features.shape} and {coords.shape}"
        assert features.shape[2] == coords.shape[2], f"get feat: {features.shape} and {coords.shape}"
        assert coords.shape[1] == 3, f"expect coords: B,3,Npoint, get: {coords.shape}"

        voxel_features_4d, voxel_coords = self.voxelization(features, coords)
        r = self.resolution

        for voxel_layers in self.voxel_layers:
            if isinstance(voxel_layers, AdaGN):
                voxel_features_4d = voxel_layers(voxel_features_4d, cond=cond)
            else:
                voxel_features_4d = voxel_layers(voxel_features_4d)

        voxel_features = trilinear_devoxelize(voxel_features_4d, voxel_coords, r, self.training)

        fused_features = voxel_features
        if self.add_point_feat:
            fused_features = fused_features + self.point_features(data).features
        if self.attn is not None:
            fused_features = self.attn(fused_features)

        data.features = fused_features

        return data


class PointNetSAModule(nn.Module):
    def __init__(
        self,
        num_centers: int,
        radius: float,
        num_neighbors: int,
        in_channels: int,
        out_channels: Tuple[Union[int, List[int]]],
        include_coordinates: bool = True,
        gn_groups: int = 8,
        cond_dim: int = 0,
        affine_gn: bool = True,
    ):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)

        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(
                    radius=_radius,
                    num_neighbors=_num_neighbors,
                    include_coordinates=include_coordinates,
                )
            )
            mlp = SharedMLP(
                in_channels=in_channels + (3 if include_coordinates else 0),
                out_channels=_out_channels,
                dim=2,
                gn_groups=gn_groups,
                cond_dim=cond_dim,
                affine=affine_gn,
            )
            mlps.append(mlp)
            total_out_channels += _out_channels[-1]

        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, data: PVCData) -> PVCData:
        coords = data.coords
        features = data.features
        time_emb = data.time_emb
        cond = data.cond

        if coords.shape[1] > 3:
            coords = coords[:, :3]

        # subsampling
        centers_coords = furthest_point_sample_pvcnn(coords, self.num_centers)

        # cutting the time embedding to the same size as the centers
        S = centers_coords.shape[-1]
        time_emb = time_emb[:, :, :S] if time_emb is not None and type(time_emb) is not dict else time_emb
        data.time_emb = time_emb

        features_list = []

        # process the features by grouping and reducing it's dimensionality
        for grouper, mlp in zip(self.groupers, self.mlps):
            # grouping features
            grouper_output = grouper(coords, centers_coords, features)
            # apply mlp
            group_features = mlp(PVCData(features=grouper_output, cond=cond)).features
            # reduction
            group_features = group_features.max(dim=-1).values
            features_list.append(group_features)

        if len(features_list) > 1:
            features = (torch.cat(features_list, dim=1),)
        else:
            features = features_list[0]

        data.features = features
        data.coords = centers_coords
        return data


class PointNetFPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Tuple[Union[int, List[int]]],
        gn_groups: int = 8,
        cond_dim: int = 0,
        affine_gn: bool = True,
    ):
        super().__init__()
        self.mlp = SharedMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=1,
            gn_groups=gn_groups,
            cond_dim=cond_dim,
            affine=affine_gn,
        )

    def forward(self, data: PVCData) -> PVCData:
        coords = data.coords
        features = data.features
        lower_dim_coords = data.lower_coords
        lower_dim_features = data.lower_features
        time_emb = data.time_emb

        interpolated_features = nearest_neighbor_interpolate(coords, lower_dim_coords, lower_dim_features)
        if features is not None:
            interpolated_features = torch.cat([interpolated_features, features], dim=1)
        if time_emb is not None:
            B, D, S = time_emb.shape
            N = coords.shape[-1]
            time_emb = time_emb[:, :, 0:1].expand(-1, -1, N)
            data.time_emb = time_emb

        # set new features as the interpolated features
        data.features = interpolated_features

        data = self.mlp(data)

        return data


def _linear_gn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GroupNorm(8, out_channels, affine=False),
        Swish(),
    )


def create_mlp_components(
    in_channels: int,
    out_channels: List[int],
    classifier: bool = False,
    dim: int = 2,
    width_multiplier: int = 1,
    gn_groups: int = 8,
    cond_dim: int = 0,
    affine_gn: bool = True,
) -> Tuple[nn.ModuleList, int]:
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = functools.partial(SharedMLP, gn_groups=gn_groups, cond_dim=cond_dim, affine=affine_gn)
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(
                SharedMLP(
                    in_channels,
                    int(r * out_channels[-1]),
                    gn_groups=gn_groups,
                    cond_dim=cond_dim,
                    affine=affine_gn,
                )
            )
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_sa_components(
    sa_blocks,
    extra_feature_channels: int,
    input_dim: int = 3,
    embed_dim: int = 64,
    attention_fn: Optional[nn.Module] = None,
    attention_layers: List[bool] = None,
    dropout: float = 0.1,
    with_se: bool = False,
    normalize: bool = True,
    eps: float = 0,
    has_temb: bool = True,
    width_multiplier: int = 1,
    voxel_resolution_multiplier: int = 1,
    gn_groups: int = 8,
    cond_dim: int = 0,
    affine_gn: bool = True,
):
    """
    Creates the components for the Set Abstraction (SA) module in the PVCNN model.

    Args:
        sa_blocks (List[Tuple]): List of tuples containing the configurations for each SA block.
        extra_feature_channels (int): Number of extra feature channels.
        input_dim (int, optional): Dimension of the input. Defaults to 3.
        embed_dim (int, optional): Dimension of the embedding. Defaults to 64.
        attention_fn (Optional[nn.Module], optional): Attention module for the SA blocks. Defaults to None.
        attention_layers (List[bool], optional): List of attention layers for each SA block. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        with_se (bool, optional): Whether to use Squeeze-and-Excitation (SE) blocks. Defaults to False.
        normalize (bool, optional): Whether to normalize the features. Defaults to True.
        eps (float, optional): Small value to avoid division by zero. Defaults to 0.
        has_temb (bool, optional): Whether to use temporal embeddings. Defaults to True.
        width_multiplier (int, optional): Width multiplier for the SA blocks. Defaults to 1.
        voxel_resolution_multiplier (int, optional): Voxel resolution multiplier for the SA blocks. Defaults to 1.
        gn_groups (int, optional): Number of groups for Group Normalization. Defaults to 8.
        cond_dim (int, optional): Dimension of the conditional input. Defaults to 0.
        affine_gn (bool, optional): Whether to use affine transformation in Group Normalization. Defaults to True.

    Returns:
        Tuple: A tuple containing the SA layers, SA input channels, output channels, and number of centers.
    """

    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + input_dim

    sa_layers, sa_in_channels = [], []
    c = 0
    num_centers = None
    for idx, (conv_configs, sa_configs) in enumerate(sa_blocks):
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []
        use_att = attention_layers[idx] if attention_layers is not None else False

        # create point voxel layers
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = use_att and p == 0
                if voxel_resolution is None:
                    # without voxelization we just have a globally shared mlp
                    block = functools.partial(
                        SharedMLP,
                        gn_groups=gn_groups,
                        cond_dim=cond_dim,
                        affine=affine_gn,
                    )
                else:
                    # create pointvoxel layer to extract global level features
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        attention=attention,
                        attention_fn=attention_fn,
                        dropout=dropout,
                        with_se=with_se,
                        normalize=normalize,
                        eps=eps,
                        gn_groups=gn_groups,
                        cond_dim=cond_dim,
                        affine=affine_gn,
                    )

                #
                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_blocks.append(block(in_channels + embed_dim * has_temb, out_channels))

                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels

        if sa_configs is not None:
            num_centers, radius, num_neighbors, out_channels = sa_configs
            _out_channels = []
            for oc in out_channels:
                if isinstance(oc, (list, tuple)):
                    _out_channels.append([int(r * _oc) for _oc in oc])
                else:
                    _out_channels.append(int(r * oc))
            out_channels = _out_channels
            if num_centers is None:
                raise ValueError("Number of centers must never be None.")
            else:
                block = functools.partial(
                    PointNetSAModule,
                    num_centers=num_centers,
                    radius=radius,
                    num_neighbors=num_neighbors,
                    gn_groups=gn_groups,
                    cond_dim=cond_dim,
                    affine_gn=affine_gn,
                )
            sa_blocks.append(
                block(
                    in_channels=extra_feature_channels + (embed_dim * has_temb if k == 0 else 0),
                    out_channels=out_channels,
                    include_coordinates=True,
                )
            )
            in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        c += 1

        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return (
        sa_layers,
        sa_in_channels,
        in_channels,
        1 if num_centers is None else num_centers,
    )


def create_fp_components(
    fp_blocks,
    in_channels: int,
    sa_in_channels: List[int],
    attention_layers: List[bool],
    attention_fn: Optional[nn.Module] = None,
    embed_dim: int = 64,
    dropout: float = 0.1,
    has_temb: bool = True,
    with_se: bool = False,
    normalize: bool = True,
    eps: float = 0,
    width_multiplier: int = 1,
    voxel_resolution_multiplier: int = 1,
    gn_groups: int = 8,
    cond_dim: int = 0,
    affine_gn: bool = True,
):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0

    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(
                in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim * has_temb,
                out_channels=out_channels,
                gn_groups=gn_groups,
                cond_dim=cond_dim,
            )
        )
        in_channels = out_channels[-1]
        use_att = attention_layers[fp_idx]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = functools.partial(
                        SharedMLP,
                        gn_groups=gn_groups,
                        cond_dim=cond_dim,
                        affine=affine_gn,
                    )
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        attention=attention,
                        attention_fn=attention_fn,
                        dropout=dropout,
                        with_se=with_se,
                        normalize=normalize,
                        eps=eps,
                        gn_groups=gn_groups,
                        cond_dim=cond_dim,
                    )

                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels


# adapted from https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement/blob/main/pointnet2/models/pnet.py
class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x):
        # x is of shape BCHW
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # some times we may attach position info to the end of feature in the channel dimension
            # we do not need to normalize them
            x0 = x[:, 0 : self.num_channels, :, :]
            res = x[:, self.num_channels :, :, :]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out


def shared_mlp(
    channels: List[int],
    dim: int = 1,
    bn: bool = True,
    bn_first: bool = False,
    bias: bool = False,
    activation: str = "relu",
    min_groups: int = 32,
):
    assert activation in ["relu", "swish"]
    layers = []

    if dim == 1:
        conv = nn.Conv1d
    else:
        conv = nn.Conv2d

    for i in range(1, len(channels)):
        if bn_first:
            if bn:
                layers.append(MyGroupNorm(min(min_groups, channels[i - 1]), channels[i - 1]))
            if activation == "relu":
                layers.append(nn.ReLU(True))
            elif activation == "swish":
                layers.append(Swish())
        layers.append(conv(channels[i - 1], channels[i], kernel_size=1, bias=bias))
        if not bn_first:
            if bn:
                layers.append(MyGroupNorm(min_groups, channels[i]))
            if activation == "relu":
                layers.append(nn.ReLU(True))
            elif activation == "swish":
                layers.append(Swish())

    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        channels: List[int],
        dim: int = 1,
        bn: bool = True,
        bn_first: bool = False,
        bias: bool = False,
        activation: str = "relu",
        min_groups: int = 32,
        add_abs_coordinates: bool = False,
    ):
        super().__init__()
        self.mlp = shared_mlp(channels, dim, bn, bn_first, bias, activation, min_groups)
        self.add_coordinates = add_abs_coordinates

    def forward(self, data: PVCData) -> PVCData:
        features = data.features
        features = self.mlp(features)
        data.features = features
        return data


class ConditionedSharedMLPLayer(nn.Module):
    def __init__(
        self,
        channels: List[int],
        dim: int = 1,
        time_emb_dim=None,
        cond_emb_dim=None,
        use_residual=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        if dim == 1:
            conv = nn.Conv1d
        else:
            conv = nn.Conv2d

        assert len(channels) > 2, "The number of channels should be at least 3: {}".format(channels)

        self.has_cond_embed = False
        self.has_time_embed = False
        self.has_residual = False

        # create time embedding layer
        if time_emb_dim is not None:
            self.has_time_embed = True
            self.time_emb_layer = ScaleShift(time_emb_dim, channels[1])

        # create conditional embedding layer
        if cond_emb_dim is not None:
            self.has_cond_embed = True
            self.cond_emb_layer = ScaleShift(cond_emb_dim, channels[2])

        # create residual layer
        if use_residual:
            self.has_residual = True
            self.residual_layer = conv(channels[0], channels[-1], kernel_size=1)

        self.shared_mlp_0 = MLP(
            [channels[0], channels[1]], dim=dim, bn=True, bn_first=False, bias=True, activation="swish"
        )
        self.shared_mlp_1 = MLP(
            [channels[1], channels[2]], dim=dim, bn=True, bn_first=False, bias=True, activation="swish"
        )

        last_mlp_layers = []
        if len(channels) >= 3:
            in_channels = channels[2]
            for oc in channels[3:]:
                last_mlp_layers.append(
                    MLP([in_channels, oc], dim=dim, bn=True, bn_first=False, bias=True, activation="swish")
                )
                in_channels = oc

        self.last_mlp_layers = nn.ModuleList(last_mlp_layers)

    def forward(self, data: PVCData) -> PVCData:
        in_features = data.features

        data = self.shared_mlp_0(data)

        if self.has_time_embed:
            data.features = self.time_emb_layer(data.features, data.time_emb)

        data = self.shared_mlp_1(data)

        if self.has_cond_embed:
            data.features = self.cond_emb_layer(data.features, data.cond)

        for l in self.last_mlp_layers:
            data = l(data)

        if self.has_residual:
            data.features = data.features + self.residual_layer(in_features)

        return data


class Pnet2Stage(nn.Module):
    def __init__(self, mlp1, mlp2):
        super().__init__()
        self.mlp1 = ConditionedSharedMLPLayer(
            mlp1,
            dim=2,
        )
        mlp2 = [2 * mlp1[-1]] + mlp2
        self.mlp2 = ConditionedSharedMLPLayer(
            mlp2,
            dim=2,
        )

    def forward(self, x: PVCData):
        # x should be of size (B, mlp1[0], num_points)
        x.features = x.features.unsqueeze(-1)  # shape (B, mlp1[0], num_points, 1)
        feature = self.mlp1(x).features
        # feature is of shape (B, mlp1[-1], num_points, 1)
        global_feature = F.max_pool2d(feature, kernel_size=[feature.size(2), 1])
        # global_feature is of shape (B, mlp1[-1], 1, 1)
        global_feature = global_feature.expand(-1, -1, feature.size(2), -1)
        feature = torch.cat([feature, global_feature], dim=1)

        x.features = feature
        feature = self.mlp2(x).features
        global_feature = F.max_pool2d(feature, kernel_size=[feature.size(2), 1])
        global_feature = global_feature.squeeze(-1).squeeze(-1)
        return global_feature
