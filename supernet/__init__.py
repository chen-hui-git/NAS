"""一次性（one-shot）超网实践：用三明治规则训练一个弹性 Conv1d Transformer。"""

from __future__ import annotations

from .dataset import MNISTTask, ensure_mnist
from .losses import accuracy, classification_loss, distillation_loss
from .rank import (
    RankingExperimentConfig,
    average_subnet_loss,
    build_dense_subnet,
    fine_tune_subnet,
    evaluate_specs,
    kendall_tau,
    run_ranking_experiment,
    spearman_rho,
    subnet_accuracy,
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
from .shrink import ShrinkStage, build_shrink_schedule, stage_at
from .supernet import Supernet, copy_subnet_weights, parameter_breakdown
from .synthetic import BarsTask
from .tasks import MNIST_MEAN, MNIST_STD, Task, make_task, normalize_images

__all__ = [
    "ACTIVATIONS",
    "EMBED_DIMS",
    "FFN_KERNELS",
    "FFN_RATIOS",
    "HEAD_DIMS",
    "BarsTask",
    "LayerCapacity",
    "LayerChoice",
    "MNIST_MEAN",
    "MNIST_STD",
    "MNISTTask",
    "RankingExperimentConfig",
    "SandwichConfig",
    "SandwichStats",
    "SandwichTrainer",
    "SearchSpace",
    "ShrinkStage",
    "StepRecord",
    "SubnetSpec",
    "Supernet",
    "Task",
    "accuracy",
    "average_subnet_loss",
    "build_dense_subnet",
    "build_shrink_schedule",
    "classification_loss",
    "copy_subnet_weights",
    "distillation_loss",
    "ensure_mnist",
    "evaluate_specs",
    "fine_tune_subnet",
    "kendall_tau",
    "make_task",
    "normalize_images",
    "parameter_breakdown",
    "point_space",
    "run_ranking_experiment",
    "spearman_rho",
    "stage_at",
    "subnet_accuracy",
    "train_standalone",
]