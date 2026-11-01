"""Progressive shrinking: train the biggest subnet first, then unlock smaller ones.

This is the training schedule OFA (arXiv:1908.09791) uses to make one-shot supernets
usable: instead of training the whole elastic family from scratch at once, start with the
largest configuration, then repeatedly lower the frontier and let the newly unlocked
smaller subnets join the sandwich. The frontier only ever widens, so nothing that was
once trained becomes unreachable.

The schedule is expressed as stage *fractions* of the planned step budget, so it adapts to
whatever step count the demo's planner ends up choosing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .search_space import SearchSpace


@dataclass(frozen=True)
class ShrinkStage:
    """One stage of progressive shrinking: which sizes are unlocked."""

    index: int
    until: float  # cumulative fraction of the budget at which this stage ends
    min_depth: int
    embed_dims: tuple[int, ...]
    ffn_ratios: tuple[int, ...]
    ffn_kernels: tuple[int, ...]

    def space(self, space: SearchSpace) -> SearchSpace:
        """The search space as it looks during this stage (maxima unchanged)."""
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
    """Equal-length stages that unlock smaller sizes step by step.

    Stage 0 is the single largest configuration; the last stage is the full space.
    depths descend from max_depth to min_depth, widths/ratios/kernels keep the largest
    entries first and grow their unlocked set while always keeping the maximum.
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
    """The stage covering a fraction of the budget (the last one when past the end)."""
    for stage in schedule:
        if progress <= stage.until:
            return stage
    return schedule[-1]
