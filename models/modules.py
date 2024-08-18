import math
from collections import namedtuple
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from packaging import version
from torch import Tensor, einsum
from torch.nn.init import _calculate_fan_in_and_fan_out


def exists(x: Any) -> bool:
    return x is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


@torch.jit.script
def swish(input: Tensor) -> Tensor:
    return input * torch.sigmoid(input)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return swish(input)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py
class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.get_buffer("beta"))


class MultiHeadedRMSNorm(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


FlashAttentionConfig = namedtuple("FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])


class Attend(nn.Module):
    def __init__(self, dropout: float = 0.0, flash: bool = False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask: Optional[Tensor] = None):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return out

    def forward(self, q, k, v, mask: Optional[Tensor] = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, "b j -> b 1 1 j")

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out


class LinearAttention(nn.Module):
    """
    copied and modified from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L159
    """

    def __init__(self, dim, heads=4, dim_head=32, verbose=True):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x: torch.tensor (B,C,N), C=num-channels, N=num-points
        Returns:
            out: torch.tensor (B,C,N)
        """
        x = x.unsqueeze(-1)  # add w dimension
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        out = self.to_out(out)
        out = out.squeeze(-1)  # B,C,N,1 -> B,C,N
        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        heads: int = 4,
        dim_head: int = 32,
        norm: bool = False,
        norm_context: bool = False,
        time_cond_dim: int = None,
        flash: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim * 2), Rearrange("b d -> b 1 d"))

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.attend = Attend(flash=flash)

    def forward(self, x, context=None, time=None):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# adapted from https://github.com/nv-tlabs/LION/blob/main/models/adagn.py
def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out", "fan_avg"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(tensor, gain=1.0, mode="fan_in"):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode="fan_avg")


def dense(in_channels, out_channels, init_scale=1.0):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


class AdaGN(nn.Module):
    """
    adaptive group normalization
    """

    def __init__(self, num_channels, ctx_dim, ndim, num_groups=8):
        """
        ndim: dim of the input features
        n_channel: number of channels of the inputs
        ndim_style: channel of the style features
        """
        super().__init__()
        self.ndim = ndim
        self.n_channel = num_channels
        self.style_dim = ctx_dim
        self.out_dim = num_channels * 2
        self.norm = nn.GroupNorm(num_groups, num_channels)
        in_channel = num_channels
        self.emd = dense(ctx_dim, num_channels * 2)
        self.emd.bias.data[:in_channel] = 1
        self.emd.bias.data[in_channel:] = 0

    def forward(self, x, cond):
        # style: B,D
        # image: B,D,N,1
        # CHECK2D(style)
        cond = self.emd(cond)
        if self.ndim == 3:  # B,D,V,V,V
            cond = cond.view(cond.shape[0], -1, 1, 1, 1)  # 5D
        elif self.ndim == 2:  # B,D,N,1
            cond = cond.view(cond.shape[0], -1, 1, 1)  # 4D
        elif self.ndim == 1:  # B,D,N
            cond = cond.view(cond.shape[0], -1, 1)  # 4D
        else:
            raise NotImplementedError

        factor, bias = cond.chunk(2, 1)
        result = self.norm(x)
        result = result * factor + bias
        return result


# adapted from https://github.com/nv-tlabs/LION/tree/main/models
class SE3d(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

        self.channel = channel

    def __repr__(self):
        return f"SE({self.channel}, {self.channel})"

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)
