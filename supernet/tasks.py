"""A narrow task interface so one trainer can run any token stream.

A task owns the data distribution, the sequence length and the vocabulary: the
supernet, the trainer and the ranking experiment only ever ask it for batches. That
keeps the demo free to swap the synthetic induction task for a public text corpus
without touching any model code.
"""

from __future__ import annotations

from typing import Callable, Sequence

from torch import Tensor
import torch

Batch = tuple[Tensor, Tensor, Tensor]


class Task:
    """Base task: a reproducible stream of (tokens, targets, mask) batches."""

    name: str = "task"
    vocab_size: int = 2
    seq_len: int = 1

    def batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> Batch:
        raise NotImplementedError

    def fixed_batches(
        self, batch_size: int, count: int, seed: int = 0
    ) -> list[Batch]:
        """A held-out batch list; every subnet is scored on the very same batches."""
        if count < 1:
            raise ValueError("count must be positive")
        generator = torch.Generator().manual_seed(seed)
        return [self.batch(batch_size, generator) for _ in range(count)]

    def batch_fn(
        self, batch_size: int, seed: int = 0
    ) -> Callable[[int], Batch]:
        """Infinite reproducible training stream; same seed means same batches."""
        generator = torch.Generator().manual_seed(seed)

        def next_batch(_step: int) -> Batch:
            return self.batch(batch_size, generator)

        return next_batch

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "vocab_size": self.vocab_size, "seq_len": self.seq_len}


def loss_mask(length: int, batch_size: int) -> Tensor:
    """Mask for plain causal LM scoring: every position but the last counts."""
    mask = torch.ones(batch_size, length, dtype=torch.bool)
    mask[:, -1] = False
    return mask


def stack_windows(
    ids: Tensor,
    batch_size: int,
    window: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample batch_size contiguous windows of window tokens from a 1-D id tensor."""
    if ids.dim() != 1:
        raise ValueError("ids must be 1-D")
    limit = ids.numel() - window
    if limit < 1:
        raise ValueError("corpus is shorter than one window")
    starts = torch.randint(0, limit, (batch_size,), generator=generator)
    offsets = torch.arange(window)
    return ids[starts.unsqueeze(1) + offsets.unsqueeze(0)]


def make_task(
    name: str,
    *,
    seq_len: int = 32,
    data_dir: Sequence[str] | None = None,
    verbose: bool = True,
) -> Task:
    """Build a task by name: 'shakespeare' (public text) or 'induction' (synthetic)."""
    key = name.strip().lower()
    if key in {"shakespeare", "tiny-shakespeare", "tinyshakespeare", "char"}:
        from .dataset import CharLMTask

        return CharLMTask(seq_len=seq_len, data_dir=data_dir, verbose=verbose)
    if key in {"induction", "synthetic"}:
        from .data import InductionTask

        return InductionTask()
    raise ValueError(f"unknown task {name!r}; use 'shakespeare' or 'induction'")
