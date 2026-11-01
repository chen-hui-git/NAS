"""渐进收缩：先把最大子网训好，再逐步解锁更小的子网。

这是 OFA（arXiv:1908.09791）用来让 one-shot 超网可用的训练日程：不是一上来就把整个弹性族
同时从零训练，而是从最大配置开始，反复下移"前沿"，让新解锁的小子网加入三明治。前沿只会变宽，
所以一旦训练过的子网不会变得不可达。

日程用"占计划步数的比例"表达，因此能适配规划器最终选出的任何步数。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .search_space import SearchSpace


@dataclass(frozen=True)
class ShrinkStage:
    """渐进收缩的一个阶段：哪些尺寸被解锁。"""

    index: int
    until: float  # cumulative fraction of the budget at which this stage ends
    min_depth: int
    embed_dims: tuple[int, ...]
    ffn_ratios: tuple[int, ...]
    ffn_kernels: tuple[int, ...]

    def space(self, space: SearchSpace) -> SearchSpace:
        """本阶段视角下的搜索空间（上界不变，只是下界收窄）。"""
        return replace(
            space,
            min_depth=self.min_depth,
            embed_dims=self.embed_dims,
            ffn_ratios=self.ffn_ratios,
            ffn_kernels=self.ffn_kernels,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "until": round(self.until, 3),
            "min_depth": self.min_depth,
            "embed_dims": list(self.embed_dims),
            "ffn_ratios": list(self.ffn_ratios),
            "ffn_kernels": list(self.ffn_kernels),
        }


def _keep_largest(values: tuple[int, ...], count: int) -> tuple[int, ...]:
    return tuple(sorted(values, reverse=True)[:count])


def build_shrink_schedule(space: SearchSpace, stages: int) -> tuple[ShrinkStage, ...]:
    """等长的若干阶段，逐级解锁更小的尺寸。

    第 0 阶段只有最大配置，最后一个阶段就是完整空间：深度从 max_depth 逐步降到 min_depth，
    宽度/ratio/kernel 从"只留最大项"开始逐步变大，且始终保留最大值（否则 max 腿就不再是全局最大子网）。
    """
    if stages <= 1:
        return ()
    built = []
    for index in range(stages):
        progress = index / (stages - 1)
        min_depth = round(
            space.max_depth - progress * (space.max_depth - space.min_depth)
        )
        widths = _keep_largest(space.embed_dims, 1 + round(progress * (len(space.embed_dims) - 1)))
        ratios = _keep_largest(space.ffn_ratios, 1 + round(progress * (len(space.ffn_ratios) - 1)))
        kernels = _keep_largest(
            space.ffn_kernels, 1 + round(progress * (len(space.ffn_kernels) - 1))
        )
        built.append(
            ShrinkStage(
                index=index,
                until=min(1.0, (index + 1) / stages),
                min_depth=min_depth,
                embed_dims=tuple(sorted(widths)),
                ffn_ratios=tuple(sorted(ratios)),
                ffn_kernels=tuple(sorted(kernels)),
            )
        )
    return tuple(built)


def stage_at(schedule: tuple[ShrinkStage, ...], progress: float) -> ShrinkStage:
    """返回覆盖该预算比例的阶段（超出末尾时返回最后一个）。"""
    for stage in schedule:
        if progress <= stage.until:
            return stage
    return schedule[-1]