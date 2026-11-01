"""评估、导出微调与排序审计。

把 run_demo 里"训练之后"的三件事集中在这里：给子网族打分、量出切片近似要付的代价、
以及比较超网内排序与从零训练排序。这些函数只做计算并返回结构化结果，打印留给调用方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable

from .rank import (
    RankingExperimentConfig,
    average_subnet_loss,
    fine_tune_subnet,
    run_ranking_experiment,
    subnet_accuracy,
)
from .search_space import LayerChoice, SearchSpace, SubnetSpec
from .supernet import Supernet
from .tasks import Task, normalize_images


@dataclass
class FamilyReport:
    """一次子网族评估的全部结果（含定性对照用的原图与标签）。"""

    batches: list
    chance_loss: float
    chance_accuracy: float
    evals: list[dict] = field(default_factory=list)
    depth_sweep: list[dict] = field(default_factory=list)
    width_sweep: list[dict] = field(default_factory=list)
    example: SubnetSpec | None = None
    predictions: dict = field(default_factory=dict)
    preview_images: object | None = None

    @property
    def example_dict(self) -> dict:
        return self.example.to_dict() if self.example else {}


def quality_spec(space: SearchSpace, depth: int, embed_dim: int) -> SubnetSpec:
    """固定"质量最高"的 block 选择，只让深度或宽度变化。"""
    head_dim = min(h for h in space.head_dims if embed_dim % h == 0)
    choice = LayerChoice(max(space.ffn_ratios), max(space.ffn_kernels), space.activations[0])
    return SubnetSpec(
        depth=depth, embed_dim=embed_dim, head_dim=head_dim,
        layers=tuple(choice for _ in range(depth)),
    )


def _row(model: Supernet, spec: SubnetSpec, batches: list, **extra) -> dict:
    return {
        "arch_id": spec.arch_id,
        "parameters": model.subnet_parameter_count(spec),
        "loss": average_subnet_loss(model, spec, batches),
        "accuracy": subnet_accuracy(model, spec, batches),
        **extra,
    }


def evaluate_family(
    model: Supernet,
    space: SearchSpace,
    task: Task,
    *,
    batch_size: int,
    eval_batches: int,
    seed: int,
    example: SubnetSpec | None = None,
) -> FamilyReport:
    """在固定的测试批次上给整个子网族打分（不做任何训练）。"""
    batches = task.fixed_batches(batch_size, eval_batches, seed=seed)
    report = FamilyReport(
        batches=batches,
        chance_loss=math.log(task.num_classes),
        chance_accuracy=1.0 / task.num_classes,
    )
    max_spec, min_spec = space.max_subnet(), space.min_subnet()
    report.example = example or space.sample(random.Random(seed + 7))
    report.evals = [
        _row(model, max_spec, batches, label="max"),
        _row(model, min_spec, batches, label="min"),
        _row(model, report.example, batches, label="random sample"),
    ]
    report.depth_sweep = [
        _row(model, quality_spec(space, depth, space.max_embed_dim), batches, depth=depth)
        for depth in range(space.min_depth, space.max_depth + 1)
    ]
    report.width_sweep = [
        _row(model, quality_spec(space, space.max_depth, embed_dim), batches, embed_dim=embed_dim)
        for embed_dim in space.embed_dims
    ]
    images, labels = task.preview()
    report.preview_images = images
    inputs = normalize_images(images).to(getattr(task, "device", "cpu"))
    report.predictions = {
        "true": labels.tolist(),
        "max": model.predict(inputs, max_spec).tolist(),
        "min": model.predict(inputs, min_spec).tolist(),
    }
    return report


def fine_tune_report(
    model: Supernet,
    space: SearchSpace,
    task: Task,
    report: FamilyReport,
    *,
    steps: int,
    lr: float = 3e-4,
    batch_size: int,
    seed: int,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """把每个已评估的子网导出后单独微调，量出"切片是近似"的代价。"""
    if steps <= 0:
        return []
    specs = [space.max_subnet(), space.min_subnet(), report.example]
    rows = []
    for row, spec in zip(report.evals, specs):
        tuned = fine_tune_subnet(
            model, spec, task, steps=steps, lr=lr, batch_size=batch_size, seed=seed
        )
        tuned_accuracy = subnet_accuracy(tuned, spec, report.batches)
        rows.append(
            {
                "label": row["label"],
                "arch_id": row["arch_id"],
                "parameters": row["parameters"],
                "zero_shot": row["accuracy"],
                "fine_tuned": tuned_accuracy,
                "gain": tuned_accuracy - row["accuracy"],
            }
        )
        if log:
            log(
                f"      {row['label']:14s} zero-shot {row['accuracy'] * 100:5.2f}% -> "
                f"fine-tuned {tuned_accuracy * 100:5.2f}%  ({rows[-1]['gain'] * 100:+.2f} pt)"
            )
    return rows


def audit_ranking(
    model: Supernet,
    space: SearchSpace,
    task: Task,
    *,
    candidates: int,
    standalone_steps: int,
    eval_batches: int,
    batch_size: int,
    lr: float,
    seed: int,
    deadline: float | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """超网内排序 vs 从零训练排序。"""
    config = RankingExperimentConfig(
        num_candidates=candidates,
        standalone_steps=standalone_steps,
        eval_batches=eval_batches,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )

    def progress(index: int, total: int, spec: SubnetSpec) -> None:
        if log:
            log(
                f"      独立训练候选 {index + 1}/{total}  "
                f"depth={spec.depth} width={spec.embed_dim} id={spec.arch_id}"
            )

    return run_ranking_experiment(
        model, space, task, config,
        progress=progress, deadline=deadline, min_candidates=3,
    )
