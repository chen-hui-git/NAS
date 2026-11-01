"""Tiny Shakespeare: a small, public, character-level language-modelling corpus.

Project Gutenberg's Shakespeare plays, distributed by Andrej Karpathy's char-rnn
repository (public domain text). The download is verified by SHA-256 and cached, so a
teaching session needs the network exactly once; pass --dataset induction to stay fully
offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

from torch import Tensor
import torch

from .tasks import Batch, Task, loss_mask, stack_windows

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_tiny_shakespeare(
    data_dir: str | Path | None = None,
    *,
    url: str = TINY_SHAKESPEARE_URL,
    expected_sha256: str = TINY_SHAKESPEARE_SHA256,
    verbose: bool = True,
) -> Path:
    """Return the local corpus path, downloading and verifying it when missing."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tiny_shakespeare.txt"
    if path.exists() and _sha256(path) == expected_sha256:
        return path
    if verbose:
        print(f"  downloading tiny shakespeare -> {path}")
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return path


class CharLMTask(Task):
    """Character-level next-token prediction over a public text corpus."""

    def __init__(
        self,
        seq_len: int = 32,
        val_fraction: float = 0.1,
        data_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        path = ensure_tiny_shakespeare(data_dir, verbose=verbose)
        text = path.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}
        ids = torch.tensor([self.stoi[char] for char in text], dtype=torch.long)
        split = int(ids.numel() * (1.0 - val_fraction))
        self.train_ids = ids[:split]
        self.val_ids = ids[split:]
        self.seq_len = seq_len
        self.vocab_size = len(self.chars)
        self.name = "tiny-shakespeare"

    def _windows(self, pool: Tensor, batch_size: int, generator: torch.Generator | None) -> Batch:
        tokens = stack_windows(pool, batch_size, self.seq_len, generator)
        return tokens, tokens, loss_mask(self.seq_len, batch_size)

    def batch(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        return self._windows(self.train_ids, batch_size, generator)

    def fixed_batches(
        self, batch_size: int, count: int, seed: int = 0
    ) -> list[Batch]:
        generator = torch.Generator().manual_seed(seed)
        return [self._windows(self.val_ids, batch_size, generator) for _ in range(count)]

    def decode(self, ids: Tensor) -> str:
        return "".join(self.itos[int(index)] for index in ids.flatten().tolist())

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "corpus_chars": int(self.train_ids.numel() + self.val_ids.numel()),
            "train_chars": int(self.train_ids.numel()),
            "val_chars": int(self.val_ids.numel()),
        }
