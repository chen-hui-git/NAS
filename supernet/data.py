"""A synthetic in-context induction task.

Each sequence has two halves. The first half lists key/value pairs; the second half
repeats the same keys in a fresh order and the model must emit each key's value.
Retrieval of that kind is exactly what attention depth and width buy, so the task
gives architectures measurably different quality without any downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor
import torch

from .tasks import Task


@dataclass(frozen=True)
class InductionTaskConfig:
    num_pairs: int = 6
    num_keys: int = 16
    num_values: int = 32

    def __post_init__(self) -> None:
        if self.num_pairs < 1:
            raise ValueError("num_pairs must be positive")
        if self.num_keys < self.num_pairs:
            raise ValueError("num_keys must be >= num_pairs for distinct keys")

    @property
    def num_tokens(self) -> int:
        return 4 * self.num_pairs

    @property
    def vocab_size(self) -> int:
        return self.num_keys + self.num_values


def induction_batch(
    config: InductionTaskConfig,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (tokens, targets, mask) for one batch of induction sequences.

    targets is a copy of tokens because scoring predicts tokens[t + 1] at position t;
    mask marks the second-half key positions, the only ones that count.
    """
    pairs = config.num_pairs
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores_first = torch.rand(batch_size, config.num_keys, generator=generator)
    keys_first = scores_first.argsort(dim=1)[:, :pairs]
    values = (
        torch.randint(0, config.num_values, (batch_size, pairs), generator=generator)
        + config.num_keys
    )
    value_of_key = torch.zeros(batch_size, config.num_keys, dtype=torch.long)
    value_of_key.scatter_(1, keys_first, values)
    shuffle = torch.rand(batch_size, pairs, generator=generator).argsort(dim=1)
    keys_second = keys_first.gather(1, shuffle)
    values_second = value_of_key.gather(1, keys_second)

    seq = config.num_tokens
    tokens = torch.zeros(batch_size, seq, dtype=torch.long)
    mask = torch.zeros(batch_size, seq, dtype=torch.bool)
    tokens[:, 0 : 2 * pairs : 2] = keys_first
    tokens[:, 1 : 2 * pairs : 2] = values
    tokens[:, 2 * pairs :: 2] = keys_second
    tokens[:, 2 * pairs + 1 :: 2] = values_second
    mask[:, 2 * pairs :: 2] = True
    return tokens.to(device), tokens.to(device), mask.to(device)


def make_batch_fn(
    config: InductionTaskConfig,
    batch_size: int,
    seed: int = 0,
    device: str = "cpu",
) -> Callable[[int], tuple[Tensor, Tensor, Tensor]]:
    """Infinite, reproducible batch stream; identical streams for fair comparisons."""
    generator = torch.Generator().manual_seed(seed)

    def batch_fn(_step: int) -> tuple[Tensor, Tensor, Tensor]:
        return induction_batch(config, batch_size, generator, device)

    return batch_fn


def fixed_eval_batches(
    config: InductionTaskConfig,
    batch_size: int,
    count: int,
    seed: int = 0,
    device: str = "cpu",
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """A held-out batch list shared by every subnet so their losses are comparable."""
    generator = torch.Generator().manual_seed(seed)
    return [
        induction_batch(config, batch_size, generator, device) for _ in range(count)
    ]
class InductionTask(Task):
    """The synthetic induction task behind the Task interface."""

    name = "synthetic-induction"

    def __init__(self, config: InductionTaskConfig | None = None) -> None:
        self.config = config or InductionTaskConfig()
        self.vocab_size = self.config.vocab_size
        self.seq_len = self.config.num_tokens

    def batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        return induction_batch(self.config, batch_size, generator)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "num_pairs": self.config.num_pairs,
            "num_keys": self.config.num_keys,
            "num_values": self.config.num_values,
        }
