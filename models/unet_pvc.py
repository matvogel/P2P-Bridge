from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention

from .pvcnn import (
    LinearAttention,
    Pnet2Stage,
    PVCData,
    SharedMLP,
    Swish,
    create_fp_components,
    create_mlp_components,
    create_pvc_layer_params,
    create_sa_components,
)


# adapted from https://github.com/alexzhou907/PVD
class PVCNN2Unet(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        return_layers: bool = False,
    ):
        super().__init__()

        model_cfg = cfg.model
        pvd_cfg = model_cfg.PVD

        # initialize class variables
        self.return_layers = return_layers
        self.input_dim = train_utils.default(model_cfg.in_dim, 3)

        if "extra_feature_channels" in pvd_cfg:
            self.extra_feature_channels = pvd_cfg.extra_feature_channels
        elif "extra_feature_channels" in model_cfg:
            self.extra_feature_channels = model_cfg.extra_feature_channels

        self.embed_dim = train_utils.default(model_cfg.time_embed_dim, 64)

        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)
        attn_type = train_utils.default(pvd_cfg.attention_type, "linear")

        self.embedf = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # global embedding
        if pvd_cfg.use_global_embedding:
            self.cond_emb_dim = pvd_cfg.global_embedding_dim
            c = self.cond_emb_dim
            global_pnet = Pnet2Stage(
                [self.input_dim, c // 8, c // 4],
                [c // 2, c],
            )
            self.global_pnet = global_pnet
        else:
            self.global_pnet = None
            self.cond_emb_dim = 0

        self.f_embed_dim = pvd_cfg.get("feat_embed_dim", self.extra_feature_channels)

        self.embed_feats = None
        if self.f_embed_dim != self.extra_feature_channels:
            in_dim = self.extra_feature_channels
            if in_dim == 0:
                in_dim = self.input_dim
            self.embed_feats = nn.Sequential(
                nn.Conv1d(in_dim, self.f_embed_dim, kernel_size=1, bias=True),
                nn.GroupNorm(8, self.f_embed_dim),
                Swish(),
                nn.Conv1d(self.f_embed_dim, self.f_embed_dim, kernel_size=1, bias=True),
            )

        sa_blocks, fp_blocks = create_pvc_layer_params(
            npoints=cfg.data.npoints,
            channels=cfg.model.PVD.channels,
            n_sa_blocks=cfg.model.PVD.n_sa_blocks,
            n_fp_blocks=cfg.model.PVD.n_fp_blocks,
            radius=cfg.model.PVD.radius,
            voxel_resolutions=cfg.model.PVD.voxel_resolutions,
            centers=pvd_cfg.centers if "centers" in pvd_cfg else None,
        )

        # prepare attention
        if attn_type.lower() == "linear":
            attention_fn = partial(LinearAttention, heads=cfg.model.PVD.attention_heads)
        elif attn_type.lower() == "flash":
            attention_fn = partial(Attention, norm=False, flash=True, heads=cfg.model.PVD.attention_heads)
        else:
            attention_fn = None

        # create set abstraction layers
        (
            sa_layers,
            sa_in_channels,
            channels_sa_features,
            *_,
        ) = create_sa_components(
            input_dim=self.input_dim,
            sa_blocks=sa_blocks,
            extra_feature_channels=self.f_embed_dim,
            with_se=pvd_cfg.get("use_se", True),
            embed_dim=self.embed_dim,  # time embedding dim
            attention_fn=attention_fn,
            attention_layers=cfg.model.PVD.attentions,
            dropout=dropout,
            gn_groups=8,
            cond_dim=self.cond_emb_dim,
        )

        self.sa_layers = nn.ModuleList(sa_layers)

        if attention_fn is not None:
            self.global_att = attention_fn(dim=channels_sa_features)

        # create feature propagation layers
        # only use extra features in the last fp module WHY ACTUALLY??
        sa_in_channels[0] = self.f_embed_dim + self.input_dim

        fp_layers, channels_fp_features = create_fp_components(
            fp_blocks=fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=pvd_cfg.get("use_se", True),
            embed_dim=self.embed_dim,
            attention_layers=cfg.model.PVD.attentions,
            attention_fn=attention_fn,
            dropout=dropout,
            gn_groups=8,
            cond_dim=self.cond_emb_dim,
        )

        self.fp_layers = nn.ModuleList(fp_layers)

        # output projection
        out_mlp = cfg.model.PVD.get("out_mlp", 128)
        layers, *_ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[out_mlp, dropout, out_dim],
            classifier=True,
            dim=2,
        )
        self.classifier = nn.ModuleList(layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        assert len(timesteps.shape) == 1, f"get shape: {timesteps.shape}"

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, x, t, x_cond=None):
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)

        (B, C, N), device = x.shape, x.device
        assert (
            C == self.input_dim + self.extra_feature_channels
        ), f"input dim: {C}, expected: {self.input_dim + self.extra_feature_channels}"

        coords = x[:, : self.input_dim, :].contiguous()
        features = x[:, self.input_dim :, :].contiguous()

        # embed features if we set a feature embedding dimension
        if self.embed_feats is not None:
            if self.extra_feature_channels == 0:
                features = self.embed_feats(coords)
            else:
                features = self.embed_feats(features)

        # initialize data class
        data = PVCData(coords=coords, features=coords)

        # global embedding
        if self.global_pnet is not None:
            global_feature = self.global_pnet(data)
            data.cond = global_feature
        else:
            global_feature = None

        # take coords + extra features as the feature input to the model
        features = torch.cat([coords, features], dim=1)

        # initialize lists
        coords_list, in_features_list = [], []
        out_features_list = []

        # append concatenated coords and features to lists
        in_features_list.append(features)

        time_emb = None
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            time_emb = self.embedf(self.get_timestep_embedding(t, device))[:, :, None].expand(-1, -1, N)

        # initialize dataclass
        data.features = features
        data.time_emb = time_emb

        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(data.features)
            coords_list.append(data.coords)

            if i > 0 and data.time_emb is not None:
                data.features = torch.cat([data.features, data.time_emb], dim=1)
                data = sa_blocks(data)
            else:
                data = sa_blocks(data)

        # remove first added feature in feature list
        in_features_list.pop(1)

        # global attention at middle layer
        if self.global_att is not None:
            features = data.features
            if isinstance(self.global_att, LinearAttention):
                features = self.global_att(features)
            elif isinstance(self.global_att, Attention):
                features = rearrange(features, "b n c -> b c n")
                features = self.global_att(features)
                features = rearrange(features, "b c n -> b n c")
            else:
                raise ValueError(f"Invalid attention type: {type(self.global_att)}")
            data.features = features

        # add first element to out_features_list, after attention mechanism
        out_features_list.append(data.features)

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            data_fp = PVCData(
                features=in_features_list[-1 - fp_idx],
                coords=coords_list[-1 - fp_idx],
                lower_coords=data.coords,
                lower_features=(
                    torch.cat([data.features, data.time_emb], dim=1) if data.time_emb is not None else data.features
                ),
                time_emb=data.time_emb,
                cond=data.cond,
            )
            data = fp_blocks(data_fp)
            out_features_list.append(data.features)

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                data.features = l(data).features
            else:
                data.features = l(data.features)

        return data.features
