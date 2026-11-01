"""The weight-sharing supernet and the subnet export path."""

from __future__ import annotations

from torch import Tensor, nn
import torch

from .modules import ElasticBlock, ElasticConv1d, ElasticLayerNorm
from .search_space import LayerCapacity, SearchSpace, SubnetSpec, point_space


class Supernet(nn.Module):
    """Maximum-width, maximum-depth Transformer in which every subnet is a slice.

    Depth is elastic by truncation: a subnet of depth L runs layers 0..L-1. Width,
    head size and the per-block FFN choice are elastic by leading-channel slicing.
    All of it is one parameter set, so 4..10 layers x 3 widths x 6 FFN branches share
    a single optimizer state.
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
        if len(capacities) != space.max_depth:
            raise ValueError(
                f"got {len(capacities)} layer capacities for depth {space.max_depth}"
            )
        max_dim = space.max_embed_dim
        self.tok_emb = nn.Embedding(space.vocab_size, max_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_dim, space.max_seq_len))
        self.layers = nn.ModuleList(
            [
                ElasticBlock(max_dim, capacity.ratios, capacity.kernels)
                for capacity in capacities
            ]
        )
        self.norm = ElasticLayerNorm(max_dim)
        self.head = ElasticConv1d(max_dim, space.vocab_size, 1)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    # [snippet:supernet-forward]
    def forward(self, tokens: Tensor, spec: SubnetSpec) -> Tensor:
        """Run one subnet over a token batch; returns logits (batch, vocab, seq)."""
        spec.validate(self.space)
        if tokens.dim() != 2:
            raise ValueError("tokens must be (batch, seq)")
        seq = tokens.shape[1]
        if seq > self.space.max_seq_len:
            raise ValueError(f"seq {seq} exceeds max_seq_len {self.space.max_seq_len}")
        dim = spec.embed_dim
        x = self.tok_emb(tokens).transpose(1, 2)[:, :dim] + self.pos[:, :dim, :seq]
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
        return self.head(x, self.space.vocab_size, dim)
    # [/snippet:supernet-forward]

    def subnet_parameter_count(self, spec: SubnetSpec) -> int:
        """Analytic parameter count of one subnet, excluding unused branches."""
        spec.validate(self.space)
        dim = spec.embed_dim
        vocab = self.space.vocab_size
        total = vocab * dim + dim * self.space.max_seq_len
        for layer in spec.layers:
            hidden = dim * layer.ffn_ratio
            total += 2 * dim
            total += 3 * dim * dim + 3 * dim
            total += dim * dim + dim
            total += 2 * dim
            total += hidden * dim * layer.ffn_kernel + hidden
            total += dim * hidden + dim
        total += 2 * dim
        total += vocab * dim + vocab
        return total

    @torch.no_grad()
    def generate(
        self,
        spec: SubnetSpec,
        prompt: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Greedy/temperature sampling from one subnet; used for teaching samples."""
        spec.validate(self.space)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if prompt.dim() == 1:
            prompt = prompt.reshape(1, -1)
        if prompt.dim() != 2 or prompt.shape[0] != 1:
            raise ValueError("generate expects a single prompt sequence")
        was_training = self.training
        self.eval()
        ids = prompt
        for _ in range(max_new_tokens):
            window = ids[:, -self.space.max_seq_len :]
            logits = self(window, spec)[:, :, -1] / max(temperature, 1e-6)
            next_id = torch.multinomial(
                logits.softmax(dim=-1), 1, generator=generator
            )
            ids = torch.cat([ids, next_id], dim=1)
        if was_training:
            self.train()
        return ids

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
    target.tok_emb.weight.copy_(source.tok_emb.weight[:, :dim])
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
    _copy_conv(target.head, source.head, source.space.vocab_size, dim)


def _copy_conv(target: ElasticConv1d, source: ElasticConv1d, out_channels: int, in_channels: int) -> None:
    target.weight.copy_(source.weight[:out_channels, :in_channels])
    if target.bias is not None and source.bias is not None:
        target.bias.copy_(source.bias[:out_channels])


def _copy_norm(target: ElasticLayerNorm, source: ElasticLayerNorm, channels: int) -> None:
    target.weight.copy_(source.weight[:channels])
    target.bias.copy_(source.bias[:channels])