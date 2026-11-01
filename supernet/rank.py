"""超网给出的子网排序，和独立训练得到的排序一致吗？

这正是 one-shot NAS 最危险的失效模式：共享权重是在"互相适配"的前提下训出来的，
所以它的子网排序可能和"把每个子网从零训练一遍"得到的排序不一致。下面的实验抽取若干候选，
先在超网内打分，再让每个候选以相同预算独立训练后打分，最后报告两个排序之间的
Kendall τ / Spearman ρ，以及"相信超网 top-1"要付的 regret。
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable

import torch

from .losses import classification_loss
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
    """独立模型：只包含一个子网的参数，没有任何共享。"""
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
    """在给定批次上分类正确的样本比例。"""
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
    """同一份权重下若干子网的验证损失。"""
    return [average_subnet_loss(model, spec, batches) for spec in specs]


def train_standalone(
    space: SearchSpace,
    spec: SubnetSpec,
    task: Task,
    config: RankingExperimentConfig,
    *,
    seed: int,
) -> Supernet:
    """在完全相同的数据流与步数预算下，把某个子网从零训练一遍。"""
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
    """把某个子网导出后单独继续训练：事后的专化（fine-tune）。

    导出的模型从共享超网的切片权重起步，所以它的初始分数就等于它在超网内的分数。
    学习率取预训练的十分之一，因为这些权重已经被训练过；几百步通常足够。
    整个过程不碰超网，因此其它子网完全不受影响。
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
    """所有两两组合上的 Kendall τ-a；某一对内打平则既不算同序也不算反序。"""
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
    """平均秩上的 Pearson 相关；打平的名次共享平均秩。"""
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
    """从 1 开始的排名，1 表示损失最低（最好）。"""
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
    """比较"超网内排序"与"从零独立训练排序"。

    deadline 是一个绝对时间（time.monotonic()）：只要还有时间就继续训练下一个候选，
    但至少完成 min_candidates 个，保证相关系数仍有定义。
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