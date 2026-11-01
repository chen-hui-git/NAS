"""数字分类任务的损失与指标。"""

from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """一批类别标签上的平均交叉熵。"""
    return F.cross_entropy(logits, labels)


def distillation_loss(
    logits: Tensor, teacher_logits: Tensor, reduction: str = "batchmean"
) -> Tensor:
    """对已 detach 的教师网络算 KL 散度，即 in-place distillation（子网蒸馏）。"""
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(teacher_logits, dim=-1).detach(),
        reduction=reduction,
    )


def accuracy(logits: Tensor, labels: Tensor) -> float:
    """这一批里分类正确的样本比例。"""
    return float((logits.argmax(dim=1) == labels).to(logits.dtype).mean())