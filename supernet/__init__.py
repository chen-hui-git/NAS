"""One-shot supernet demo: an elastic Conv1d Transformer trained with the sandwich rule."""

from __future__ import annotations

from .data import InductionTask, InductionTaskConfig, induction_batch
from .dataset import CharLMTask, ensure_tiny_shakespeare
from .losses import masked_next_token_loss
from .rank import (
    RankingExperimentConfig,
    average_subnet_loss,
    build_dense_subnet,
    evaluate_specs,
    kendall_tau,
    run_ranking_experiment,
    spearman_rho,
    train_standalone,
)
from .sandwich import SandwichConfig, SandwichStats, SandwichTrainer, StepRecord
from .search_space import (
    ACTIVATIONS,
    EMBED_DIMS,
    FFN_KERNELS,
    FFN_RATIOS,
    HEAD_DIMS,
    LayerCapacity,
    LayerChoice,
    SearchSpace,
    SubnetSpec,
    point_space,
)
from .supernet import Supernet, copy_subnet_weights
from .tasks import Task, make_task

__all__ = [
    "ACTIVATIONS",
    "EMBED_DIMS",
    "FFN_KERNELS",
    "FFN_RATIOS",
    "HEAD_DIMS",
    "CharLMTask",
    "InductionTask",
    "InductionTaskConfig",
    "LayerCapacity",
    "LayerChoice",
    "RankingExperimentConfig",
    "SandwichConfig",
    "SandwichStats",
    "SandwichTrainer",
    "SearchSpace",
    "StepRecord",
    "SubnetSpec",
    "Supernet",
    "Task",
    "average_subnet_loss",
    "build_dense_subnet",
    "copy_subnet_weights",
    "ensure_tiny_shakespeare",
    "evaluate_specs",
    "induction_batch",
    "kendall_tau",
    "make_task",
    "masked_next_token_loss",
    "point_space",
    "run_ranking_experiment",
    "spearman_rho",
    "train_standalone",
]
