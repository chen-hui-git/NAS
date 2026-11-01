"""A narrow task interface so one trainer can run any dataset.

A task owns the data distribution, the sequence length, the per-token feature width and
the number of classes: the supernet, the trainer and the ranking experiment only ever
ask a task for batches of (inputs, labels). That keeps the demo free to swap MNIST for
a synthetic fallback without touching any model code.
"""

from __future__ import annotations

from typing import Callable, Sequence

from torch import Tensor
import torch

Batch = tuple[Tensor, Tensor]

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_images(images: Tensor) -> Tensor:
    """(N, rows, cols) pixels -> (N, cols, rows).

    One image row becomes one token, and the pixel values of that row are the token
    features. The same mean/std normalisation is used by every task so a subnet's
    quality is comparable across datasets.
    """
    if images.dim() != 3:
        raise ValueError("expected images of shape (batch, rows, cols)")
    pixels = images.to(torch.float32)
    if images.dtype == torch.uint8:
        pixels = pixels / 255.0
    return ((pixels - MNIST_MEAN) / MNIST_STD).transpose(1, 2).contiguous()


class Task:
    """Base task: a reproducible stream of (inputs, labels) batches.

    inputs has shape (batch, input_features, seq_len); labels has shape (batch,).
    """

    name: str = "task"
    num_classes: int = 2
    input_features: int = 1
    seq_len: int = 1
    device: str = "cpu"

    def _finish(self, inputs: Tensor, labels: Tensor) -> Batch:
        """Move a batch to the task device so model and data always agree."""
        return inputs.to(self.device), labels.to(self.device)

    def batch(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        raise NotImplementedError

    def fixed_batches(self, batch_size: int, count: int, seed: int = 0) -> list[Batch]:
        """A held-out batch list; every subnet is scored on the very same batches."""
        if count < 1:
            raise ValueError("count must be positive")
        generator = torch.Generator().manual_seed(seed)
        return [self.batch(batch_size, generator) for _ in range(count)]

    def batch_fn(self, batch_size: int, seed: int = 0) -> Callable[[int], Batch]:
        """Infinite reproducible training stream; same seed means same batches."""
        generator = torch.Generator().manual_seed(seed)

        def next_batch(_step: int) -> Batch:
            return self.batch(batch_size, generator)

        return next_batch

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "input_features": self.input_features,
            "seq_len": self.seq_len,
        }


def make_task(
    name: str,
    *,
    data_dir: Sequence[str] | None = None,
    train_limit: int | None = None,
    verbose: bool = True,
    device: str = "cpu",
) -> Task:
    """Build a task by name: 'mnist' (public digits) or 'bars' (synthetic, offline)."""
    key = name.strip().lower()
    if key in {"mnist", "digits", "mnist-digits"}:
        from .dataset import MNISTTask

        return MNISTTask(
            data_dir=data_dir, train_limit=train_limit, verbose=verbose, device=device
        )
    if key in {"bars", "synthetic"}:
        from .synthetic import BarsTask

        return BarsTask(device=device)
    raise ValueError(f"unknown task {name!r}; use 'mnist' or 'bars'")
