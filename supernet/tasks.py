"""一个窄任务接口，让同一套训练器能跑任何数据集。

任务对象持有数据分布、序列长度、每个 token 的特征宽度与类别数；超网、训练器与排序实验
只向它索取 (inputs, labels) 批次。这样换数据集（MNIST 换成合成回退任务）不需要改动任何模型代码。
"""

from __future__ import annotations

from typing import Callable, Sequence

from torch import Tensor
import torch

Batch = tuple[Tensor, Tensor]

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_images(images: Tensor) -> Tensor:
    """(N, rows, cols) 像素 -> (N, cols, rows)。

    图像的一行变成一个 token，该行的像素值就是这个 token 的特征。所有任务共用同一套
    均值/方差归一化，这样不同数据集上的子网质量才可比。
    """
    pixels = images.to(torch.float32)
    if images.dtype == torch.uint8:
        pixels = pixels / 255.0
    return ((pixels - MNIST_MEAN) / MNIST_STD).transpose(1, 2).contiguous()


class Task:
    """任务基类：可复现的 (inputs, labels) 批次流。

    inputs 形状为 (batch, input_features, seq_len)，labels 形状为 (batch,)。
    """

    name: str = "task"
    num_classes: int = 2
    input_features: int = 1
    seq_len: int = 1
    device: str = "cpu"

    def _finish(self, inputs: Tensor, labels: Tensor) -> Batch:
        """把批次搬到任务所在设备，保证模型与数据始终一致。"""
        return inputs.to(self.device), labels.to(self.device)

    def batch(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        raise NotImplementedError

    def fixed_batches(self, batch_size: int, count: int, seed: int = 0) -> list[Batch]:
        """留出的评测批次列表；所有子网都在完全相同的批次上打分。"""
        generator = torch.Generator().manual_seed(seed)
        return [self.batch(batch_size, generator) for _ in range(count)]

    def batch_fn(self, batch_size: int, seed: int = 0) -> Callable[[int], Batch]:
        """无限可复现的训练流；同一个 seed 就是同一批数据。"""
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
    """按名字构建任务：'mnist'（公开手写数字）或 'bars'（合成、可离线）。"""
    key = name.strip().lower()
    if key in {"bars", "synthetic"}:
        from .synthetic import BarsTask

        return BarsTask(device=device)
    from .dataset import MNISTTask

    return MNISTTask(
        data_dir=data_dir, train_limit=train_limit, verbose=verbose, device=device
    )