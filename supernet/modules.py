"""Elastic building blocks.

Every weight tensor is allocated for the *widest* subnet and a narrower subnet uses
the leading input/output channels of that tensor. This leading-slice trick is the
standard one-shot NAS approximation (OFA): one shared tensor serves every subnet, so
the whole family trains with a single optimizer state. Slicing changes which weights
are used but never mixes two branches, so an unused branch cannot leak into the
output of a subnet.
"""

from __future__ import annotations

import math
from typing import Callable

from torch import Tensor, nn
import torch
import torch.nn.functional as F

ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "gelu": F.gelu,
    "silu": F.silu,
}


def branch_key(ffn_ratio: int, ffn_kernel: int) -> str:
    return f"r{ffn_ratio}k{ffn_kernel}"


class ElasticConv1d(nn.Module):
    """Conv1d whose leading out_channels / in_channels form the active slice."""

    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.empty(max_out_channels, max_in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(max_out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.max_in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # [snippet:elastic-conv]
    def forward(self, x: Tensor, out_channels: int, in_channels: int) -> Tensor:
        weight = self.weight[:out_channels, :in_channels]
        bias = None if self.bias is None else self.bias[:out_channels]
        return F.conv1d(x, weight, bias, padding=self.padding)
    # [/snippet:elastic-conv]


class ElasticLayerNorm(nn.Module):
    """LayerNorm over channels, sliced to the subnet width."""

    def __init__(self, max_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(max_channels))
        self.bias = nn.Parameter(torch.zeros(max_channels))

    def forward(self, x: Tensor, channels: int) -> Tensor:
        y = x.transpose(1, 2)
        y = F.layer_norm(
            y, (channels,), self.weight[:channels], self.bias[:channels], self.eps
        )
        return y.transpose(1, 2)


class ElasticAttention(nn.Module):
    """Multi-head self-attention over the token sequence, elastic width and head size."""

    def __init__(self, max_dim: int) -> None:
        super().__init__()
        self.max_dim = max_dim
        self.qkv = ElasticConv1d(max_dim, 3 * max_dim, 1)
        self.proj = ElasticConv1d(max_dim, max_dim, 1)

    def forward(self, x: Tensor, dim: int, head_dim: int) -> Tensor:
        heads = dim // head_dim
        batch, _, seq = x.shape
        qkv = self.qkv(x, 3 * dim, dim)
        q, k, v = qkv.split(dim, dim=1)
        shape = (batch, heads, head_dim, seq)
        q = q.reshape(shape).transpose(2, 3)
        k = k.reshape(shape).transpose(2, 3)
        v = v.reshape(shape).transpose(2, 3)
        context = F.scaled_dot_product_attention(q, k, v)
        context = context.transpose(2, 3).reshape(batch, dim, seq)
        return self.proj(context, dim, dim)


class _FFNBranch(nn.Module):
    """One (ratio, kernel) Conv1d feed-forward branch, sized at max capacity."""

    def __init__(self, max_dim: int, ratio: int, kernel: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel = kernel
        self.max_hidden = max_dim * ratio
        self.conv1 = ElasticConv1d(max_dim, self.max_hidden, kernel)
        self.conv2 = ElasticConv1d(self.max_hidden, max_dim, 1)

    def forward(self, x: Tensor, dim: int, activation: str) -> Tensor:
        hidden = dim * self.ratio
        h = self.conv1(x, hidden, dim)
        h = ACTIVATIONS[activation](h)
        return self.conv2(h, dim, hidden)


class ElasticFFN(nn.Module):
    """A set of Conv1d feed-forward branches; a subnet activates exactly one."""

    def __init__(
        self, max_dim: int, ratios: tuple[int, ...], kernels: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                branch_key(ratio, kernel): _FFNBranch(max_dim, ratio, kernel)
                for ratio in ratios
                for kernel in kernels
            }
        )

    # [snippet:elastic-ffn]
    def forward(
        self, x: Tensor, dim: int, ffn_ratio: int, ffn_kernel: int, activation: str
    ) -> Tensor:
        return self.branches[branch_key(ffn_ratio, ffn_kernel)](x, dim, activation)
    # [/snippet:elastic-ffn]


class ElasticBlock(nn.Module):
    """Pre-norm Transformer block with elastic width, head size and FFN choice."""

    def __init__(
        self, max_dim: int, ratios: tuple[int, ...], kernels: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.norm1 = ElasticLayerNorm(max_dim)
        self.attn = ElasticAttention(max_dim)
        self.norm2 = ElasticLayerNorm(max_dim)
        self.ffn = ElasticFFN(max_dim, ratios, kernels)

    def forward(
        self,
        x: Tensor,
        dim: int,
        head_dim: int,
        ffn_ratio: int,
        ffn_kernel: int,
        activation: str,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x, dim), dim, head_dim)
        x = x + self.ffn(self.norm2(x, dim), dim, ffn_ratio, ffn_kernel, activation)
        return x
