"""The weight-sharing supernet and the subnet export path."""

from __future__ import annotations

from torch import Tensor, nn
import torch

from .modules import ElasticBlock, ElasticConv1d, ElasticLayerNorm
from .search_space import LayerCapacity, SearchSpace, SubnetSpec, point_space


class Supernet(nn.Module):
    """Maximum-width, maximum-depth digit classifier in which every subnet is a slice.

    The input is a (batch, input_features, seq_len) tensor: for MNIST each of the 28
    image rows is one token whose features are the 28 pixel values of that row. Depth is
    elastic by truncation: a subnet of depth L runs layers 0..L-1. Width, head size and
    the per-block FFN choice are elastic by leading-channel slicing. All of it is one
    parameter set, so 4..10 layers x 3 widths x 6 FFN branches share one optimizer state.
    """

    def __init__(
        self,
        space: SearchSpace,
        capacities: tuple[LayerCapacity, ...] | None = None,
    ) -> None:
        super().__init__()
        self.space = space
        if capacities is None:
            capacities = tuple(
                LayerCapacity(space.ffn_ratios, space.ffn_kernels)
                for _ in range(space.max_depth)
            )
        max_dim = space.max_embed_dim
        self.input_proj = ElasticConv1d(space.input_features, max_dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, max_dim, space.max_seq_len))
        self.layers = nn.ModuleList(
            [
                ElasticBlock(max_dim, capacity.ratios, capacity.kernels)
                for capacity in capacities
            ]
        )
        self.norm = ElasticLayerNorm(max_dim)
        self.head = ElasticConv1d(max_dim, space.num_classes, 1)

    # [snippet:supernet-forward]
    def forward(self, inputs: Tensor, spec: SubnetSpec) -> Tensor:
        """Run one subnet over a batch; returns class logits (batch, num_classes)."""
        features, seq = inputs.shape[1], inputs.shape[2]
        dim = spec.embed_dim
        x = self.input_proj(inputs, dim, features) + self.pos[:, :dim, :seq]
        for index in range(spec.depth):
            layer = spec.layers[index]
            x = self.layers[index](
                x,
                dim,
                spec.head_dim,
                layer.ffn_ratio,
                layer.ffn_kernel,
                layer.activation,
            )
        x = self.norm(x, dim)
        pooled = x.mean(dim=-1, keepdim=True)
        return self.head(pooled, self.space.num_classes, dim).squeeze(-1)
    # [/snippet:supernet-forward]

    @torch.no_grad()
    def predict(self, inputs: Tensor, spec: SubnetSpec) -> Tensor:
        """Predicted class indices for a batch."""
        was_training = self.training
        self.eval()
        labels = self(inputs, spec).argmax(dim=1)
        if was_training:
            self.train()
        return labels

    def subnet_parameter_count(self, spec: SubnetSpec) -> int:
        """Analytic parameter count of one subnet, excluding unused branches."""
        dim = spec.embed_dim
        classes = self.space.num_classes
        total = self.space.input_features * dim + dim
        total += dim * self.space.max_seq_len
        for layer in spec.layers:
            hidden = dim * layer.ffn_ratio
            total += 2 * dim
            total += 3 * dim * dim + 3 * dim
            total += dim * dim + dim
            total += 2 * dim
            total += hidden * dim * layer.ffn_kernel + hidden
            total += dim * hidden + dim
        total += 2 * dim
        total += classes * dim + classes
        return total

    @torch.no_grad()
    def export_subnet(self, spec: SubnetSpec) -> "Supernet":
        """Copy one subnet into a standalone model that holds only its own weights."""
        point, capacities = point_space(spec, self.space)
        exported = Supernet(point, capacities=capacities)
        copy_subnet_weights(exported, self, spec)
        exported.eval()
        return exported


@torch.no_grad()
def copy_subnet_weights(target: Supernet, source: Supernet, spec: SubnetSpec) -> None:
    """Slice the shared weights of source into the narrower target."""
    dim = spec.embed_dim
    _copy_conv(target.input_proj, source.input_proj, dim, source.space.input_features)
    target.pos.copy_(source.pos[:, :dim, :])
    for index in range(spec.depth):
        layer = spec.layers[index]
        src_layer = source.layers[index]
        dst_layer = target.layers[index]
        _copy_conv(dst_layer.attn.qkv, src_layer.attn.qkv, 3 * dim, dim)
        _copy_conv(dst_layer.attn.proj, src_layer.attn.proj, dim, dim)
        _copy_norm(dst_layer.norm1, src_layer.norm1, dim)
        _copy_norm(dst_layer.norm2, src_layer.norm2, dim)
        key = f"r{layer.ffn_ratio}k{layer.ffn_kernel}"
        src_branch = src_layer.ffn.branches[key]
        dst_branch = dst_layer.ffn.branches[key]
        hidden = dim * layer.ffn_ratio
        _copy_conv(dst_branch.conv1, src_branch.conv1, hidden, dim)
        _copy_conv(dst_branch.conv2, src_branch.conv2, dim, hidden)
    _copy_norm(target.norm, source.norm, dim)
    _copy_conv(target.head, source.head, source.space.num_classes, dim)


def _copy_conv(
    target: ElasticConv1d, source: ElasticConv1d, out_channels: int, in_channels: int
) -> None:
    target.weight.copy_(source.weight[:out_channels, :in_channels])
    if target.bias is not None and source.bias is not None:
        target.bias.copy_(source.bias[:out_channels])


def _copy_norm(target: ElasticLayerNorm, source: ElasticLayerNorm, channels: int) -> None:
    target.weight.copy_(source.weight[:channels])
    target.bias.copy_(source.bias[:channels])
