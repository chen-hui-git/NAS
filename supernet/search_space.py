"""弹性（权重共享）Conv1d Transformer 的声明式搜索空间。

一个子网由冻结的 SubnetSpec 描述，里面不含任何张量状态。因此 spec 可以被记录、哈希、精确重放，
这也是超网能作为搜索策略的"适应度函数"的前提：策略只输出数据，永远不输出模型代码。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

MIN_DEPTH = 4
MAX_DEPTH = 10
EMBED_DIMS = (64, 96, 128)
HEAD_DIMS = (16, 32)
FFN_RATIOS = (2, 3, 4)
FFN_KERNELS = (1, 3, 5)
ACTIVATIONS = ("gelu", "silu")


@dataclass(frozen=True)
class LayerChoice:
    """每个 block 的弹性旋钮：Conv1d 前馈层的扩张倍数、卷积核大小与激活函数。"""

    ffn_ratio: int
    ffn_kernel: int
    activation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ffn_ratio": self.ffn_ratio,
            "ffn_kernel": self.ffn_kernel,
            "activation": self.activation,
        }


@dataclass(frozen=True)
class SubnetSpec:
    """搜索空间中的一个点：弹性深度 + 宽度 + 逐 block 的选择。"""

    depth: int
    embed_dim: int
    head_dim: int
    layers: tuple[LayerChoice, ...]

    @property
    def num_heads(self) -> int:
        return self.embed_dim // self.head_dim

    @property
    def arch_id(self) -> str:
        payload = json.dumps(self._signature(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _signature(self) -> dict[str, Any]:
        """定义架构本身的字段，不含由它们推导出的哈希值。"""
        return {
            "depth": self.depth,
            "embed_dim": self.embed_dim,
            "head_dim": self.head_dim,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "num_heads": self.num_heads,
            **self._signature(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubnetSpec":
        return cls(
            depth=int(data["depth"]),
            embed_dim=int(data["embed_dim"]),
            head_dim=int(data["head_dim"]),
            layers=tuple(LayerChoice(**layer) for layer in data["layers"]),
        )


@dataclass(frozen=True)
class LayerCapacity:
    """某个超网层为弹性 FFN 实例化了哪些分支。"""

    ratios: tuple[int, ...]
    kernels: tuple[int, ...]


@dataclass(frozen=True)
class SearchSpace:
    """冻结的空间定义；超网的尺寸由各项最大值决定。"""

    min_depth: int = MIN_DEPTH
    max_depth: int = MAX_DEPTH
    embed_dims: tuple[int, ...] = EMBED_DIMS
    head_dims: tuple[int, ...] = HEAD_DIMS
    ffn_ratios: tuple[int, ...] = FFN_RATIOS
    ffn_kernels: tuple[int, ...] = FFN_KERNELS
    activations: tuple[str, ...] = ACTIVATIONS
    num_classes: int = 10
    input_features: int = 28
    max_seq_len: int = 32

    def valid_dim_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (dim, head_dim)
            for dim in self.embed_dims
            for head_dim in self.head_dims
            if dim % head_dim == 0
        )

    def architecture_count(self) -> int:
        """这个空间能容纳的候选架构总数（宽度-head 组合 × 每层分支数的 depth 次幂之和）。"""
        per_layer = len(self.ffn_ratios) * len(self.ffn_kernels) * len(self.activations)
        pairs = len(self.valid_dim_pairs())
        return sum(
            pairs * per_layer ** depth
            for depth in range(self.min_depth, self.max_depth + 1)
        )

    @property
    def max_embed_dim(self) -> int:
        return max(self.embed_dims)

    @property
    def min_embed_dim(self) -> int:
        return min(self.embed_dims)

    @property
    def max_ratio(self) -> int:
        return max(self.ffn_ratios)

    @property
    def max_kernel(self) -> int:
        return max(self.ffn_kernels)

    def max_subnet(self) -> SubnetSpec:
        """最贵的子网：三明治规则里的 max 腿。"""
        embed_dim = max(self.embed_dims)
        head_dim = self._head_dim_for(embed_dim, prefer="min")
        layer = LayerChoice(max(self.ffn_ratios), max(self.ffn_kernels), self.activations[0])
        return SubnetSpec(
            depth=self.max_depth,
            embed_dim=embed_dim,
            head_dim=head_dim,
            layers=tuple(layer for _ in range(self.max_depth)),
        )

    def min_subnet(self) -> SubnetSpec:
        """最便宜的子网：三明治规则里的 min 腿。"""
        embed_dim = min(self.embed_dims)
        head_dim = self._head_dim_for(embed_dim, prefer="max")
        layer = LayerChoice(min(self.ffn_ratios), min(self.ffn_kernels), self.activations[0])
        return SubnetSpec(
            depth=self.min_depth,
            embed_dim=embed_dim,
            head_dim=head_dim,
            layers=tuple(layer for _ in range(self.min_depth)),
        )

    def _head_dim_for(self, embed_dim: int, *, prefer: str) -> int:
        candidates = [h for h in self.head_dims if embed_dim % h == 0]
        return min(candidates) if prefer == "min" else max(candidates)

    def sample(self, rng: "random.Random | None" = None) -> SubnetSpec:
        """在深度、宽度与逐层选择上均匀采样一个子网。"""
        source = random if rng is None else rng
        depth = source.randint(self.min_depth, self.max_depth)
        embed_dim, head_dim = source.choice(self.valid_dim_pairs())
        layers = tuple(
            LayerChoice(
                ffn_ratio=source.choice(self.ffn_ratios),
                ffn_kernel=source.choice(self.ffn_kernels),
                activation=source.choice(self.activations),
            )
            for _ in range(depth)
        )
        return SubnetSpec(
            depth=depth, embed_dim=embed_dim, head_dim=head_dim, layers=layers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "embed_dims": list(self.embed_dims),
            "head_dims": list(self.head_dims),
            "ffn_ratios": list(self.ffn_ratios),
            "ffn_kernels": list(self.ffn_kernels),
            "activations": list(self.activations),
            "num_classes": self.num_classes,
            "input_features": self.input_features,
            "max_seq_len": self.max_seq_len,
        }


def point_space(
    spec: SubnetSpec, space: SearchSpace
) -> tuple[SearchSpace, tuple[LayerCapacity, ...]]:
    """把空间收缩到单个子网。

    返回的空间恰好只接受 spec，并且只实例化 spec 用到的那些分支——子网导出成独立模型就是靠它。
    """
    point = SearchSpace(
        min_depth=spec.depth,
        max_depth=spec.depth,
        embed_dims=(spec.embed_dim,),
        head_dims=(spec.head_dim,),
        ffn_ratios=space.ffn_ratios,
        ffn_kernels=space.ffn_kernels,
        activations=space.activations,
        num_classes=space.num_classes,
        input_features=space.input_features,
        max_seq_len=space.max_seq_len,
    )
    capacities = tuple(
        LayerCapacity((layer.ffn_ratio,), (layer.ffn_kernel,)) for layer in spec.layers
    )
    return point, capacities