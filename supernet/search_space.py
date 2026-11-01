"""Declarative search space for an elastic (weight-sharing) Conv1d Transformer.

A subnet is described by a frozen SubnetSpec that carries no tensor state. Specs can
therefore be logged, hashed and replayed exactly, which is what makes a supernet
usable as a *fitness function* for a search policy: the policy only ever emits data,
never model code.
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
    """Per-block elastic knobs: Conv1d feed-forward expansion, kernel and activation."""

    ffn_ratio: int
    ffn_kernel: int
    activation: str

    def validate(self, space: "SearchSpace") -> None:
        if self.ffn_ratio not in space.ffn_ratios:
            raise ValueError(f"ffn_ratio {self.ffn_ratio} not in {space.ffn_ratios}")
        if self.ffn_kernel not in space.ffn_kernels:
            raise ValueError(f"ffn_kernel {self.ffn_kernel} not in {space.ffn_kernels}")
        if self.activation not in space.activations:
            raise ValueError(f"activation {self.activation} not in {space.activations}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ffn_ratio": self.ffn_ratio,
            "ffn_kernel": self.ffn_kernel,
            "activation": self.activation,
        }


@dataclass(frozen=True)
class SubnetSpec:
    """One point in the search space: elastic depth + width + per-block choice."""

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
        """The fields that define the architecture, without the derived hash."""
        return {
            "depth": self.depth,
            "embed_dim": self.embed_dim,
            "head_dim": self.head_dim,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def validate(self, space: "SearchSpace") -> None:
        if not space.min_depth <= self.depth <= space.max_depth:
            raise ValueError(
                f"depth {self.depth} outside [{space.min_depth}, {space.max_depth}]"
            )
        if self.embed_dim not in space.embed_dims:
            raise ValueError(f"embed_dim {self.embed_dim} not in {space.embed_dims}")
        if self.head_dim not in space.head_dims:
            raise ValueError(f"head_dim {self.head_dim} not in {space.head_dims}")
        if self.embed_dim % self.head_dim != 0:
            raise ValueError("embed_dim must be divisible by head_dim")
        if len(self.layers) != self.depth:
            raise ValueError(
                f"spec has {len(self.layers)} layer choices but depth {self.depth}"
            )
        for layer in self.layers:
            layer.validate(space)

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
    """Which branches a supernet layer instantiates for its elastic FFN."""

    ratios: tuple[int, ...]
    kernels: tuple[int, ...]


@dataclass(frozen=True)
class SearchSpace:
    """Frozen space definition; the supernet is sized by its maxima."""

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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.min_depth < 1:
            raise ValueError("min_depth must be positive")
        if self.max_depth < self.min_depth:
            raise ValueError("max_depth must be >= min_depth")
        for name, values in (
            ("embed_dims", self.embed_dims),
            ("head_dims", self.head_dims),
            ("ffn_ratios", self.ffn_ratios),
            ("ffn_kernels", self.ffn_kernels),
            ("activations", self.activations),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
        if self.num_classes < 2 or self.max_seq_len < 1:
            raise ValueError("num_classes and max_seq_len must be positive")
        if self.input_features < 1:
            raise ValueError("input_features must be positive")
        for ratio in self.ffn_ratios:
            if ratio < 1:
                raise ValueError("ffn_ratios must be positive")
        for kernel in self.ffn_kernels:
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError("ffn_kernels must be positive and odd")
        if not self.valid_dim_pairs():
            raise ValueError("no (embed_dim, head_dim) pair is divisible")

    def valid_dim_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (dim, head_dim)
            for dim in self.embed_dims
            for head_dim in self.head_dims
            if dim % head_dim == 0
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
        """The most expensive subnet: the "max" leg of the sandwich rule."""
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
        """The cheapest subnet: the "min" leg of the sandwich rule."""
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
        if not candidates:
            raise ValueError(f"no head_dim divides embed_dim {embed_dim}")
        return min(candidates) if prefer == "min" else max(candidates)

    def sample(self, rng: "random.Random | None" = None) -> SubnetSpec:
        """Sample one subnet uniformly over depth, width and per-layer choices."""
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
    """Collapse the space onto a single subnet.

    The result admits exactly spec and instantiates only the branches that spec
    needs, which is how a subnet is exported into a standalone model.
    """
    spec.validate(space)
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
