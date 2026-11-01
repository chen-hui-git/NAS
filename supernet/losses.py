"""Losses and metrics for the digit-classification task."""

from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross-entropy over a batch of class labels."""
    return F.cross_entropy(logits, labels)


def accuracy(logits: Tensor, labels: Tensor) -> float:
    """Fraction of correctly classified samples in this batch."""
    return float((logits.argmax(dim=1) == labels).to(logits.dtype).mean())
