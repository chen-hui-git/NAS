"""离线回退任务：合成"亮条"图像。

类别 d 对应一条亮度为 1 的横条，位置由类别决定，其余像素是高斯噪声。这不是真实数据集，
存在的意义只是让 demo 在没有网络时也能跑通链路。
"""

from __future__ import annotations

from torch import Tensor
import torch

from .tasks import Batch, Task, normalize_images


class BarsTask(Task):
    """十个"条带位置"类别；极易学习，只适合当冒烟测试用。"""

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
        self.noise = noise
        self.bar_rows = bar_rows
        self.input_features = input_features
        self.seq_len = seq_len

    def _render(self, labels: Tensor, generator: torch.Generator | None) -> Tensor:
        """每个样本画一条横条；返回形状 (n, rows, cols)、取值在 [0, 1] 的像素。"""
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
        """每个类别取一张确定性样本，返回 uint8 图像供报告插图使用。"""
        labels = torch.arange(self.num_classes)
        pixels = self._render(labels, torch.Generator().manual_seed(0))
        images = (pixels * 255).round().to(torch.uint8)
        return images, labels