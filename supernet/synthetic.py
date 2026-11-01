"""Offline fallback task: synthetic images of a bright bar.

Class d is a bright bar of a few rows placed at a class-specific position, plus gaussian
noise. Nothing here is a real dataset -- it exists so the demo still runs with no
network.
"""

from __future__ import annotations

from torch import Tensor
import torch

from .tasks import Batch, Task, normalize_images


class BarsTask(Task):
    """Ten classes of bar position; trivially learnable, useful only as a smoke test."""

    name = "synthetic-bars"
    num_classes = 10

    def __init__(
        self,
        noise: float = 0.25,
        bar_rows: int = 2,
        input_features: int = 28,
        seq_len: int = 28,
        device: str = "cpu",
    ) -> None:
        self.device = device
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if bar_rows < 1:
            raise ValueError("bar_rows must be positive")
        if input_features < 1 or seq_len < bar_rows + 1:
            raise ValueError("image too small for the requested bar")
        self.noise = noise
        self.bar_rows = bar_rows
        self.input_features = input_features
        self.seq_len = seq_len

    def _render(self, labels: Tensor, generator: torch.Generator | None) -> Tensor:
        """Draw one bar per sample; returns pixels in [0, 1] of shape (n, rows, cols)."""
        count = int(labels.numel())
        pixels = self.noise * torch.randn(
            count, self.seq_len, self.input_features, generator=generator
        ) + 0.1
        span = self.seq_len - self.bar_rows
        starts = (labels * span) // self.num_classes
        rows = torch.arange(count)
        for offset in range(self.bar_rows):
            pixels[rows, (starts + offset).clamp(max=self.seq_len - 1), :] = 1.0
        return pixels.clamp(0.0, 1.0)

    def batch(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        labels = torch.randint(0, self.num_classes, (batch_size,), generator=generator)
        return self._finish(normalize_images(self._render(labels, generator)), labels)

    def preview(self) -> tuple[Tensor, Tensor]:
        """One deterministic sample per class, as uint8 images for the report figure."""
        labels = torch.arange(self.num_classes)
        pixels = self._render(labels, torch.Generator().manual_seed(0))
        images = (pixels * 255).round().to(torch.uint8)
        return images, labels
