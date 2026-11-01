"""MNIST handwritten digits: the classic teaching dataset.

The four IDX files are the original LeCun MNIST distribution. Each download is checked
twice -- gzip + IDX header/body length (catches truncation, the usual failure) and a
recorded SHA-256 -- then cached, so a teaching session downloads once and works offline
afterwards. Use --dataset bars for a purely synthetic fallback.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import struct
import urllib.request

from torch import Tensor
import torch

from .tasks import Batch, Task, normalize_images

MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# name -> (expected decompressed file name, expected gzipped size, expected sha256)
MNIST_FILES: dict[str, tuple[int, str]] = {
    "train-images-idx3-ubyte.gz": (9912422, "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    "train-labels-idx1-ubyte.gz": (28881, "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    "t10k-images-idx3-ubyte.gz": (1648877, "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    "t10k-labels-idx1-ubyte.gz": (4542, "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(name: str, payload: bytes) -> bytes:
    """Validate an IDX file structurally; returns the decompressed bytes."""
    raw = gzip.decompress(payload)
    if "images" in name:
        magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
        if magic != 2051:
            raise ValueError(f"{name}: bad magic {magic}, expected 2051")
        if len(raw) - 16 != count * rows * cols:
            raise ValueError(f"{name}: body length does not match header")
    else:
        magic, count = struct.unpack(">II", raw[:8])
        if magic != 2049:
            raise ValueError(f"{name}: bad magic {magic}, expected 2049")
        if len(raw) - 8 != count:
            raise ValueError(f"{name}: body length does not match header")
    return raw


def _file_is_valid(path: Path, name: str) -> bool:
    try:
        _decode(name, path.read_bytes())
    except Exception:
        return False
    return True


def ensure_mnist(
    data_dir: str | Path | None = None,
    *,
    mirrors: tuple[str, ...] = MNIST_MIRRORS,
    verbose: bool = True,
) -> Path:
    """Return the local MNIST directory, downloading and validating what is missing."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    for name, (size, expected_sha) in MNIST_FILES.items():
        path = root / name
        if path.exists() and path.stat().st_size == size and _file_is_valid(path, name):
            continue
        last_error: Exception | None = None
        for mirror in mirrors:
            url = f"{mirror}/{name}"
            if verbose:
                print(f"  downloading {name} from {mirror}")
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = response.read()
                _decode(name, payload)
            except Exception as error:  # network or format failure: try the next mirror
                last_error = error
                continue
            path.write_bytes(payload)
            if _sha256(path) != expected_sha:
                if verbose:
                    print(f"  warning: {name} differs from the recorded sha256 "
                          f"(mirror recompressed the file); structure verified")
            break
        else:
            raise RuntimeError(f"could not download {name}: {last_error}")
    return root


def _read_images(path: Path) -> Tensor:
    raw = _decode(path.name, path.read_bytes())
    _, count, rows, cols = struct.unpack(">IIII", raw[:16])
    data = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8)
    return data.reshape(count, rows, cols)


def _read_labels(path: Path) -> Tensor:
    raw = _decode(path.name, path.read_bytes())
    _, count = struct.unpack(">II", raw[:8])
    data = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8)
    return data.reshape(count).to(torch.long)


class MNISTTask(Task):
    """MNIST digit classification, one image row per token."""

    name = "mnist"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        train_limit: int | None = None,
        verbose: bool = True,
        device: str = "cpu",
    ) -> None:
        root = ensure_mnist(data_dir, verbose=verbose)
        self.device = device
        self.train_images = _read_images(root / "train-images-idx3-ubyte.gz")
        self.train_labels = _read_labels(root / "train-labels-idx1-ubyte.gz")
        self.test_images = _read_images(root / "t10k-images-idx3-ubyte.gz")
        self.test_labels = _read_labels(root / "t10k-labels-idx1-ubyte.gz")
        if train_limit is not None:
            limit = min(int(train_limit), self.train_images.shape[0])
            self.train_images = self.train_images[:limit]
            self.train_labels = self.train_labels[:limit]
        self.num_classes = 10
        self.input_features = int(self.train_images.shape[2])
        self.seq_len = int(self.train_images.shape[1])

    def batch(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        indices = torch.randint(0, self.train_images.shape[0], (batch_size,), generator=generator)
        return self._finish(
            normalize_images(self.train_images[indices]), self.train_labels[indices]
        )

    def fixed_batches(self, batch_size: int, count: int, seed: int = 0) -> list[Batch]:
        generator = torch.Generator().manual_seed(seed)
        batches = []
        for _ in range(count):
            indices = torch.randint(
                0, self.test_images.shape[0], (batch_size,), generator=generator
            )
            batches.append(
                self._finish(
                    normalize_images(self.test_images[indices]),
                    self.test_labels[indices],
                )
            )
        return batches

    def preview(self) -> tuple[Tensor, Tensor]:
        """One test image per class, for the qualitative figure."""
        chosen = []
        for label in range(self.num_classes):
            index = int((self.test_labels == label).nonzero()[0])
            chosen.append(index)
        indices = torch.tensor(chosen, dtype=torch.long)
        return self.test_images[indices], self.test_labels[indices]

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "train_images": int(self.train_images.shape[0]),
            "test_images": int(self.test_images.shape[0]),
            "image_shape": f"{self.seq_len}x{self.input_features}",
            "source": "LeCun MNIST (idx3/idx1 ubyte), cached and sha256-checked",
        }
