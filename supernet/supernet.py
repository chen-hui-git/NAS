"""权重共享的超网本体，以及子网导出路径。"""

from __future__ import annotations

from torch import Tensor, nn
import torch

from .modules import ElasticBlock, ElasticConv1d, ElasticLayerNorm
from .search_space import LayerCapacity, SearchSpace, SubnetSpec, point_space


class Supernet(nn.Module):
    """最宽、最深的数字分类器，其中每个子网都是一次切片。

    输入是 (batch, input_features, seq_len)：对 MNIST 来说，28 行图像各是一个 token，
    它的特征就是该行的 28 个像素值。深度靠截断伸缩：深度 L 的子网只跑 0..L-1 层。
    宽度、head 大小与逐 block 的 FFN 选择靠"取前 N 个通道"伸缩。全部是一套参数，
    所以 4..10 层 × 3 种宽度 × 6 条 FFN 分支共用一个优化器状态。
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
        """用某一个子网跑一个批次；返回类别 logits，形状 (batch, num_classes)。"""
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
        """返回一个批次的预测类别索引。"""
        was_training = self.training
        self.eval()
        labels = self(inputs, spec).argmax(dim=1)
        if was_training:
            self.train()
        return labels

    def subnet_parameter_count(self, spec: SubnetSpec) -> int:
        """解析式计算单个子网的参数量，不计未被用到的分支。"""
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
        """把某个子网拷贝成一个独立模型，里面只保留它自己的权重。"""
        point, capacities = point_space(spec, self.space)
        exported = Supernet(point, capacities=capacities)
        copy_subnet_weights(exported, self, spec)
        exported.eval()
        return exported


@torch.no_grad()
def copy_subnet_weights(target: Supernet, source: Supernet, spec: SubnetSpec) -> None:
    """把 source 的共享权重切片后写入更窄的 target。"""
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

def parameter_breakdown(space: SearchSpace) -> dict[str, object]:
    """按组成拆解超网参数量（解析式，不需要实例化模型）。

    超网把每条 FFN 分支都按最大形状各存一份，因为卷积核无法切片；宽度与深度则可以切片，
    所以不必为每种宽度/层数再存一份。于是"超网 - 最大子网"几乎正好等于每层另外 8 条分支。
    """
    d = space.max_embed_dim
    depth = space.max_depth
    attention = (3 * d * d + 3 * d) + (d * d + d)
    norms = 2 * (2 * d)
    io = (
        (space.input_features * d + d)
        + d * space.max_seq_len
        + 2 * d
        + (space.num_classes * d + space.num_classes)
    )
    branches = {}
    for ratio in space.ffn_ratios:
        for kernel in space.ffn_kernels:
            hidden = d * ratio
            branches[f"r{ratio}k{kernel}"] = hidden * d * kernel + hidden + d * hidden + d
    ffn_per_layer = sum(branches.values())
    max_branch = f"r{max(space.ffn_ratios)}k{max(space.ffn_kernels)}"
    total = depth * (attention + norms + ffn_per_layer) + io
    return {
        "supernet": total,
        "attention": depth * attention,
        "norms": depth * norms,
        "io": io,
        "ffn_all": depth * ffn_per_layer,
        "ffn_max_branch": depth * branches[max_branch],
        "ffn_unused": depth * (ffn_per_layer - branches[max_branch]),
        "max_branch_key": max_branch,
        "branches": branches,
        "max_subnet": depth * (attention + norms + branches[max_branch]) + io,
    }
