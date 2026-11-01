"""Masked causal language-modelling loss shared by training and evaluation."""

from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def masked_next_token_loss(logits: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy of tokens[t + 1] predicted from position t.

    logits is (batch, vocab, seq), tokens is (batch, seq) and mask is a boolean
    (batch, seq) selecting the scored positions. Only masked positions contribute;
    the returned value is the masked mean.
    """
    if logits.dim() != 3 or tokens.dim() != 2 or mask.dim() != 2:
        raise ValueError("expected logits (B,V,T), tokens (B,T) and mask (B,T)")
    if logits.shape[0] != tokens.shape[0] or tokens.shape != mask.shape:
        raise ValueError("logits, tokens and mask must agree on batch and length")
    if logits.shape[-1] != tokens.shape[-1]:
        raise ValueError("logits and tokens must agree on sequence length")

    vocab = logits.shape[1]
    flat_logits = logits[:, :, :-1].transpose(1, 2).reshape(-1, vocab)
    flat_targets = tokens[:, 1:].reshape(-1)
    flat_mask = mask[:, :-1].reshape(-1).to(flat_logits.dtype)
    per_position = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (per_position * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
