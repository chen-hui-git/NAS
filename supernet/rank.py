"""Does the supernet rank subnets the way standalone training does?

This is the failure mode that makes one-shot NAS dangerous: shared weights are trained
under co-adaptation, so their ordering of subnets can disagree with the ordering you get
by training each subnet from scratch. The experiment below samples candidates, scores
them inside the supernet, trains each of them independently with an identical budget, and
reports Kendall tau / Spearman rho between the two rankings plus the regret of trusting
the supernet's top-1.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable

from torch import Tensor
import torch

from .losses import accuracy, classification_loss
from .search_space import SearchSpace, SubnetSpec, point_space
from .supernet import Supernet
from .tasks import Batch, Task


@dataclass(frozen=True)
class RankingExperimentConfig:
    num_candidates: int = 10
    standalone_steps: int = 250
    eval_batches: int = 8
    batch_size: int = 32
    lr: float = 3e-3
    grad_clip: float = 1.0
    seed: int = 0

def build_dense_subnet(space: SearchSpace, spec: SubnetSpec) -> Supernet:
    """A standalone model with exactly one subnet's parameters and no sharing."""
    point, capacities = point_space(spec, space)
    return Supernet(point, capacities=capacities)


@torch.no_grad()
def average_subnet_loss(model: Supernet, spec: SubnetSpec, batches: list[Batch]) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for inputs, labels in batches:
        total += float(classification_loss(model(inputs, spec), labels))
    if was_training:
        model.train()
    return total / len(batches)


@torch.no_grad()
def subnet_accuracy(model: Supernet, spec: SubnetSpec, batches: list[Batch]) -> float:
    """Fraction of correctly classified samples over the given batches."""
    was_training = model.training
    model.eval()
    correct = 0
    seen = 0
    for inputs, labels in batches:
        correct += int((model(inputs, spec).argmax(dim=1) == labels).sum())
        seen += int(labels.numel())
    if was_training:
        model.train()
    return correct / max(1, seen)


def evaluate_specs(
    model: Supernet, specs: list[SubnetSpec], batches: list[Batch]
) -> list[float]:
    """Validation loss of several subnets inside the same weights."""
    return [average_subnet_loss(model, spec, batches) for spec in specs]


def train_standalone(
    space: SearchSpace,
    spec: SubnetSpec,
    task: Task,
    config: RankingExperimentConfig,
    *,
    seed: int,
) -> Supernet:
    """Train one subnet from scratch on an identical data stream and step budget."""
    torch.manual_seed(seed)
    device = getattr(task, "device", "cpu")
    model = build_dense_subnet(space, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    batch_fn = task.batch_fn(config.batch_size, seed=seed + 1)
    model.train()
    for step in range(config.standalone_steps):
        inputs, labels = batch_fn(step)
        optimizer.zero_grad(set_to_none=True)
        loss = classification_loss(model(inputs, spec), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    model.eval()
    return model


def fine_tune_subnet(
    supernet: Supernet,
    spec: SubnetSpec,
    task: Task,
    *,
    steps: int = 300,
    lr: float = 3e-4,
    batch_size: int = 128,
    seed: int = 0,
) -> Supernet:
    """Export one subnet and keep training it alone: post-hoc specialization.

    The exported model starts from the shared supernet's sliced weights, so its first
    score is exactly its in-supernet score. The learning rate is a tenth of the pretraining
    one because those weights are already trained; a few hundred steps is usually enough.
    Nothing here touches the supernet, so other subnets stay unaffected.
    """
    model = supernet.export_subnet(spec).to(getattr(task, "device", "cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch_fn = task.batch_fn(batch_size, seed=seed)
    model.train()
    for step in range(steps):
        inputs, labels = batch_fn(step)
        optimizer.zero_grad(set_to_none=True)
        classification_loss(model(inputs, spec), labels).backward()
        optimizer.step()
    model.eval()
    return model


def kendall_tau(supernet_scores: list[float], standalone_scores: list[float]) -> float:
    """Kendall tau-a over all pairs; ties inside a pair count as neither."""
    concordant = 0
    discordant = 0
    count = len(supernet_scores)
    for i in range(count):
        for j in range(i + 1, count):
            product = (supernet_scores[i] - supernet_scores[j]) * (
                standalone_scores[i] - standalone_scores[j]
            )
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    if denominator == 0:
        return 0.0
    return (concordant - discordant) / denominator


def spearman_rho(supernet_scores: list[float], standalone_scores: list[float]) -> float:
    """Pearson correlation of average ranks; ties share the average rank."""
    left = _average_ranks(supernet_scores)
    right = _average_ranks(standalone_scores)
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left == 0 or var_right == 0:
        return 0.0
    return covariance / (var_left * var_right) ** 0.5


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def _rank_positions(values: list[float]) -> list[int]:
    """1-based ranks with 1 = lowest loss = best."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def run_ranking_experiment(
    supernet: Supernet,
    space: SearchSpace,
    task: Task,
    config: RankingExperimentConfig,
    progress: Callable[[int, int, SubnetSpec], None] | None = None,
    deadline: float | None = None,
    min_candidates: int = 3,
) -> dict[str, object]:
    """Compare the supernet's ordering of subnets with from-scratch training.

    deadline is an absolute time.monotonic() value: candidates are trained while time
    remains, but never fewer than min_candidates so the correlation stays defined.
    """
    rng = random.Random(config.seed)
    candidates = [space.sample(rng) for _ in range(config.num_candidates)]
    batches = task.fixed_batches(
        config.batch_size, config.eval_batches, seed=config.seed + 7
    )

    supernet_losses = evaluate_specs(supernet, candidates, batches)
    supernet_accuracy = [
        subnet_accuracy(supernet, spec, batches) for spec in candidates
    ]
    standalone_losses: list[float] = []
    standalone_accuracy: list[float] = []
    completed = 0
    for index, spec in enumerate(candidates):
        if progress is not None:
            progress(index, len(candidates), spec)
        model = train_standalone(
            space, spec, task, config, seed=config.seed * 1000 + index + 1
        )
        standalone_losses.append(average_subnet_loss(model, spec, batches))
        standalone_accuracy.append(subnet_accuracy(model, spec, batches))
        completed += 1
        if (
            deadline is not None
            and time.monotonic() > deadline
            and completed >= min(min_candidates, len(candidates))
        ):
            break

    candidates = candidates[:completed]
    supernet_losses = supernet_losses[:completed]
    supernet_accuracy = supernet_accuracy[:completed]
    supernet_ranks = _rank_positions(supernet_losses)
    standalone_ranks = _rank_positions(standalone_losses)
    supernet_top = min(range(completed), key=lambda i: supernet_losses[i])
    standalone_top = min(range(completed), key=lambda i: standalone_losses[i])

    rows = []
    for index, spec in enumerate(candidates):
        rows.append(
            {
                "arch_id": spec.arch_id,
                "spec": spec.to_dict(),
                "parameters": supernet.subnet_parameter_count(spec),
                "supernet_loss": supernet_losses[index],
                "standalone_loss": standalone_losses[index],
                "supernet_accuracy": supernet_accuracy[index],
                "standalone_accuracy": standalone_accuracy[index],
                "supernet_rank": supernet_ranks[index],
                "standalone_rank": standalone_ranks[index],
            }
        )

    return {
        "num_candidates": completed,
        "kendall_tau": kendall_tau(supernet_losses, standalone_losses),
        "spearman_rho": spearman_rho(supernet_losses, standalone_losses),
        "top1_agreement": supernet_top == standalone_top,
        "supernet_top1_arch_id": candidates[supernet_top].arch_id,
        "standalone_top1_arch_id": candidates[standalone_top].arch_id,
        "regret": standalone_losses[supernet_top] - standalone_losses[standalone_top],
        "best_standalone_loss": standalone_losses[standalone_top],
        "best_supernet_loss": supernet_losses[supernet_top],
        "best_standalone_accuracy": standalone_accuracy[standalone_top],
        "candidates": rows,
    }
