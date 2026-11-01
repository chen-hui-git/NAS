"""Losses and metrics for the digit-classification task."""

from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross-entropy over a batch of class labels."""
    if logits.dim() != 2:
        raise ValueError("expected logits of shape (batch, num_classes)")
    if labels.dim() != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("expected labels of shape (batch,)")
    return F.cross_entropy(logits, labels)


def accuracy(logits: Tensor, labels: Tensor) -> float:
    """Fraction of correctly classified samples in this batch."""
    if logits.dim() != 2 or labels.dim() != 1:
        raise ValueError("expected logits (batch, num_classes) and labels (batch,)")
    return float((logits.argmax(dim=1) == labels).to(logits.dtype).mean())
