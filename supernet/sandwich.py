"""The sandwich-rule training loop.

Each step trains four subnets and sums their losses into one optimizer step:

    max     the most expensive subnet (deepest, widest, largest kernel/expansion)
    min     the cheapest subnet
    random  n_random uniformly sampled subnets

Rationale (slimmable networks, refined by Once-for-All): the max subnet is usually the
best performing one, the min subnet keeps the shared weights honest at the low end, and
the random legs bring coverage; every subnet also acts as an in-place distillation target
for the others, because they all update the same weights.

shrink_stages > 1 switches on progressive shrinking (OFA): the sandwich starts with the
largest configuration only, and smaller depths/widths/kernels join stage by stage until
the full space is unlocked in the last stage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random
import time
from typing import Callable

from torch import Tensor
import torch

from .losses import accuracy, classification_loss, distillation_loss
from .search_space import SearchSpace, SubnetSpec
from .shrink import ShrinkStage, build_shrink_schedule, stage_at
from .supernet import Supernet


@dataclass(frozen=True)
class SandwichConfig:
    lr: float = 3e-3
    weight_decay: float = 1e-2
    n_random: int = 2
    max_weight: float = 1.0
    min_weight: float = 1.0
    random_weight: float = 1.0
    grad_clip: float = 1.0
    seed: int = 0
    shrink_stages: int = 0
    distill_weight: float = 0.0

@dataclass
class SandwichStats:
    """Coverage evidence: which parts of the space actually received gradients."""

    steps: int = 0
    subnets_trained: int = 0
    depth_counts: Counter = field(default_factory=Counter)
    kernel_counts: Counter = field(default_factory=Counter)
    ratio_counts: Counter = field(default_factory=Counter)
    dim_counts: Counter = field(default_factory=Counter)
    activation_counts: Counter = field(default_factory=Counter)

    def record(self, spec: SubnetSpec) -> None:
        self.subnets_trained += 1
        self.depth_counts[spec.depth] += 1
        self.dim_counts[spec.embed_dim] += 1
        for layer in spec.layers:
            self.kernel_counts[layer.ffn_kernel] += 1
            self.ratio_counts[layer.ffn_ratio] += 1
            self.activation_counts[layer.activation] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "subnets_trained": self.subnets_trained,
            "depth_counts": {str(k): v for k, v in sorted(self.depth_counts.items())},
            "dim_counts": {str(k): v for k, v in sorted(self.dim_counts.items())},
            "kernel_counts": {str(k): v for k, v in sorted(self.kernel_counts.items())},
            "ratio_counts": {str(k): v for k, v in sorted(self.ratio_counts.items())},
            "activation_counts": {
                str(k): v for k, v in sorted(self.activation_counts.items())
            },
        }


@dataclass(frozen=True)
class StepRecord:
    step: int
    total_loss: float
    grad_norm: float
    role_losses: dict[str, float]
    role_accuracy: dict[str, float]
    role_arch_ids: dict[str, str]
    stage: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "total_loss": self.total_loss,
            "grad_norm": self.grad_norm,
            "role_losses": dict(self.role_losses),
            "role_accuracy": dict(self.role_accuracy),
            "role_arch_ids": dict(self.role_arch_ids),
            "stage": self.stage,
        }


class SandwichTrainer:
    """Trains one Supernet by sampling a sandwich of subnets on every step."""

    def __init__(
        self,
        space: SearchSpace,
        config: SandwichConfig | None = None,
        model: Supernet | None = None,
    ) -> None:
        self.space = space
        self.config = config or SandwichConfig()
        self.rng = random.Random(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.model = model if model is not None else Supernet(space)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.stats = SandwichStats()
        self.step_index = 0
        self.total_steps: int | None = None
        self.schedule = build_shrink_schedule(self.space, self.config.shrink_stages)

    def current_stage(self) -> ShrinkStage | None:
        """The unlocked frontier for the step about to run (None when shrink is off)."""
        if not self.schedule:
            return None
        progress = 1.0 if not self.total_steps else self.step_index / self.total_steps
        return stage_at(self.schedule, progress)

    # [snippet:sandwich-build]
    def build_sandwich(self) -> list[tuple[str, SubnetSpec, float]]:
        """Return the (role, spec, loss_weight) legs of this step's sandwich."""
        config = self.config
        stage = self.current_stage()
        space = self.space if stage is None else stage.space(self.space)
        legs: list[tuple[str, SubnetSpec, float]] = [
            ("max", space.max_subnet(), config.max_weight),
            ("min", space.min_subnet(), config.min_weight),
        ]
        for index in range(config.n_random):
            legs.append(
                (f"random_{index}", space.sample(self.rng), config.random_weight)
            )
        return legs
    # [/snippet:sandwich-build]

    def leg_loss(
        self, logits: Tensor, labels: Tensor, teacher: Tensor | None
    ) -> Tensor:
        """Cross-entropy, plus an optional KL term towards the max leg.

        distill_weight = 0 (the default) returns plain cross-entropy, so a run without
        distillation is exactly the sandwich rule with no extra machinery.
        """
        loss = classification_loss(logits, labels)
        weight = self.config.distill_weight
        if weight <= 0 or teacher is None:
            return loss
        return (1.0 - weight) * loss + weight * distillation_loss(logits, teacher)

    # [snippet:sandwich-step]
    def train_step(self, inputs: Tensor, labels: Tensor) -> StepRecord:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        role_losses: dict[str, float] = {}
        role_accuracy: dict[str, float] = {}
        role_arch_ids: dict[str, str] = {}
        total = 0.0
        stage = self.current_stage()
        teacher: Tensor | None = None
        for role, spec, weight in self.build_sandwich():
            logits = self.model(inputs, spec)
            loss = self.leg_loss(logits, labels, teacher)
            if role == "max":
                teacher = logits.detach()   # the max leg is the teacher for every other leg
            if weight > 0:
                (loss * weight).backward()
            role_losses[role] = float(loss.detach())
            role_accuracy[role] = accuracy(logits.detach(), labels)
            role_arch_ids[role] = spec.arch_id
            total += weight * float(loss.detach())
            self.stats.record(spec)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()
        self.step_index += 1
        self.stats.steps = self.step_index
        return StepRecord(
            step=self.step_index,
            total_loss=total,
            grad_norm=float(grad_norm),
            role_losses=role_losses,
            role_accuracy=role_accuracy,
            role_arch_ids=role_arch_ids,
            stage=0 if stage is None else stage.index,
        )
    # [/snippet:sandwich-step]

    def train(
        self,
        batch_fn: Callable[[int], tuple[Tensor, Tensor]],
        steps: int,
        log_every: int = 0,
        on_log: Callable[[StepRecord], None] | None = None,
        deadline: float | None = None,
    ) -> list[StepRecord]:
        """Train at most steps times; stop early past an absolute time.monotonic() deadline."""
        self.total_steps = self.step_index + steps
        history: list[StepRecord] = []
        for step in range(steps):
            if deadline is not None and time.monotonic() > deadline:
                break
            inputs, labels = batch_fn(step)
            record = self.train_step(inputs, labels)
            history.append(record)
            if on_log is not None and log_every > 0 and (
                (step + 1) % log_every == 0 or step == 0
            ):
                on_log(record)
        return history
