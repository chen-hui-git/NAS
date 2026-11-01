"""Teaching figures for the supernet demo.

Headless matplotlib only. Every figure uses English labels because the default
matplotlib font has no CJK glyphs; the surrounding HTML report is in Chinese.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "supernet-mpl-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK = "#1f2933"
MUTED = "#7b8794"
MAX_COLOR = "#d9480f"
MIN_COLOR = "#1c7ed6"
RANDOM_COLOR = "#2f9e44"
GRID = "#dde3ea"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _cell_size(max_dim: int, dim: int, max_cells: int = 16) -> int:
    """Largest divisor shared by both widths that still gives a readable grid."""
    common = [c for c in range(1, min(max_dim, dim) + 1) if max_dim % c == 0 and dim % c == 0]
    usable = [c for c in common if max_dim // c <= max_cells]
    return min(usable) if usable else max(common)


def plot_slicing_schematic(
    spec: dict,
    path: Path,
    *,
    max_embed_dim: int,
    ffn_ratios: tuple[int, ...],
    ffn_kernels: tuple[int, ...],
) -> Path:
    """Draw the exact slice one concrete subnet takes, straight from its spec.

    Nothing here is a placeholder: the grid scale comes from max_embed_dim and the
    subnet width, and the per-layer matrix shows the branch each layer actually chose.
    """
    _style()
    embed_dim = int(spec["embed_dim"])
    head_dim = int(spec["head_dim"])
    layers = spec["layers"]
    depth = len(layers)

    cell = _cell_size(max_embed_dim, embed_dim)
    cells_max = max_embed_dim // cell
    cells_sub = embed_dim // cell

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, max(4.0, 0.42 * depth + 1.6)),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
    )

    # ---- left: the shared weight tensor and the block this subnet uses
    for row in range(cells_max):
        for column in range(cells_max):
            active = row < cells_sub and column < cells_sub
            left.add_patch(
                Rectangle(
                    (column, cells_max - row - 1),
                    1,
                    1,
                    facecolor="#4c7bd9" if active else "#eef1f6",
                    edgecolor="#2b5bb0" if active else "#c9d2e0",
                    linewidth=0.5,
                )
            )
    left.add_patch(
        Rectangle((0, cells_max - cells_sub), cells_sub, cells_sub, fill=False,
                  edgecolor=MAX_COLOR, linewidth=2.2)
    )
    left.add_patch(
        Rectangle((0, 0), cells_max, cells_max, fill=False, edgecolor=INK,
                  linewidth=1.0, linestyle="--")
    )
    left.set_xlim(-0.2, cells_max + 0.2)
    left.set_ylim(-0.2, cells_max + 1.6)
    left.set_xticks([])
    left.set_yticks([])
    left.grid(False)
    left.set_aspect("equal")
    left.set_xlabel(f"input channels  (1 cell = {cell})")
    left.set_ylabel("output channels")
    left.set_title(
        f"Shared weight tensor, widest = {max_embed_dim}\n"
        f"this subnet uses the leading {cells_sub}/{cells_max} = {embed_dim} channels"
    )
    left.annotate(
        f"{embed_dim} ch used",
        xy=(cells_sub / 2, cells_max - cells_sub / 2),
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    if cells_max > cells_sub:
        left.annotate(
            f"{max_embed_dim - embed_dim} ch unused\n(belong to wider subnets)",
            xy=(cells_sub + (cells_max - cells_sub) / 2, cells_max - cells_sub / 2),
            ha="center",
            va="center",
            fontsize=8,
            color=MUTED,
        )

    # ---- right: which FFN branch each layer picked
    keys = [f"r{ratio}k{kernel}" for ratio in ffn_ratios for kernel in ffn_kernels]
    for layer_index, layer in enumerate(layers):
        chosen = f"r{int(layer['ffn_ratio'])}k{int(layer['ffn_kernel'])}"
        for column, key in enumerate(keys):
            active = key == chosen
            right.add_patch(
                Rectangle(
                    (column + 0.06, depth - layer_index - 0.92),
                    0.88,
                    0.84,
                    facecolor=RANDOM_COLOR if active else "#eef1f6",
                    edgecolor=RANDOM_COLOR if active else "#c9d2e0",
                    linewidth=1.0 if active else 0.5,
                )
            )
            if active:
                right.text(
                    column + 0.5,
                    depth - layer_index - 0.5,
                    chosen,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
    right.set_xlim(0, len(keys))
    right.set_ylim(-0.2, depth + 0.2)
    right.set_xticks([column + 0.5 for column in range(len(keys))])
    right.set_xticklabels(keys, rotation=45, ha="right", fontsize=7.5)
    right.set_yticks([depth - index - 0.5 for index in range(depth)])
    right.set_yticklabels([f"L{index}" for index in range(depth)], fontsize=7.5)
    right.grid(axis="x", visible=False)
    right.grid(axis="y", visible=False)
    right.set_title(
        f"Per-layer FFN choice of this subnet (depth {depth})\n"
        f"{embed_dim} ch, head_dim {head_dim} -> {embed_dim // head_dim} heads"
    )
    fig.tight_layout()
    return _save(fig, path)


def _leg_series(history: list, key: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """Extract (steps, max, min, random-mean) series from step records."""
    steps = [record.step for record in history]
    max_values = []
    min_values = []
    random_values = []
    for record in history:
        table = getattr(record, key)
        max_values.append(table.get("max", float("nan")))
        min_values.append(table.get("min", float("nan")))
        values = [value for role, value in table.items() if role.startswith("random")]
        random_values.append(sum(values) / len(values) if values else float("nan"))
    return steps, max_values, min_values, random_values


def plot_sandwich_curves(
    history: list, path: Path, chance_loss: float | None = None
) -> Path:
    """Training loss and accuracy of each sandwich leg, step by step."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    steps, max_losses, min_losses, random_losses = _leg_series(history, "role_losses")
    _, max_acc, min_acc, random_acc = _leg_series(history, "role_accuracy")

    for axis, series, target, ylabel in (
        (left, (max_losses, min_losses, random_losses), chance_loss, "training cross-entropy"),
        (right, (max_acc, min_acc, random_acc), None, "training accuracy"),
    ):
        for values, color, label in zip(
            series,
            (MAX_COLOR, MIN_COLOR, RANDOM_COLOR),
            ("max subnet", "min subnet", "random subnets (mean)"),
        ):
            axis.plot(steps, values, color=color, alpha=0.25, linewidth=0.8)
            axis.plot(steps, _smooth(values), color=color, linewidth=1.9, label=label)
        axis.set_xlabel("sandwich step (4 subnets per step)")
        axis.set_ylabel(ylabel)
        axis.legend(loc="best")
    if chance_loss is not None:
        left.axhline(chance_loss, color=MUTED, linestyle=":", linewidth=1.2)
        left.annotate(
            "random guessing",
            xy=(steps[0], chance_loss),
            xytext=(steps[0], chance_loss + 0.08),
            fontsize=8,
            color=MUTED,
        )
    left.set_title("Loss of each leg (thin = raw, thick = smoothed)")
    right.set_title("Accuracy of each leg")
    fig.tight_layout()
    return _save(fig, path)


def plot_predictions(
    images,
    true_labels,
    predictions: dict,
    path: Path,
) -> Path:
    """One test image per class, with true and predicted labels for two subnets."""
    _style()
    import numpy as np

    count = len(true_labels)
    fig, axes = plt.subplots(1, count, figsize=(1.05 * count + 0.8, 2.9))
    if count == 1:
        axes = [axes]
    rows = list(predictions.items())
    for index, axis in enumerate(axes):
        axis.imshow(np.asarray(images[index]), cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(f"true {int(true_labels[index])}", fontsize=8.5, color=INK)
        for offset, (label, guesses) in enumerate(rows):
            guess = int(guesses[index])
            color = RANDOM_COLOR if guess == int(true_labels[index]) else MAX_COLOR
            axis.text(
                0.5,
                -0.10 - 0.16 * offset,
                f"{label}: {guess}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color=color,
            )
    fig.suptitle("Same shared weights, two very different subnets", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _smooth(values: list[float], window: int = 15) -> list[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = [value for value in values[start : index + 1] if value == value]
        smoothed.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return smoothed


def plot_coverage(stats, path: Path) -> Path:
    """Coverage of the space: proof that the sandwich touches every depth and branch."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    panels = (
        (axes[0], stats.depth_counts, "depth", "times trained"),
        (axes[1], stats.dim_counts, "embed_dim", "times trained"),
        (axes[2], stats.kernel_counts, "ffn_kernel", "layer-visits"),
    )
    for axis, counter, title, ylabel in panels:
        keys = sorted(counter)
        values = [counter[key] for key in keys]
        axis.bar([str(key) for key in keys], values, color="#4c7bd9", width=0.65)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.grid(axis="x", visible=False)
    fig.suptitle("Sandwich coverage of the search space", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_family_curve(depth_rows: list, width_rows: list, path: Path) -> Path:
    """One training run, a whole family of subnets: depth curve and cost/quality scatter."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.1))

    depths = [row["depth"] for row in depth_rows]
    losses = [row["loss"] for row in depth_rows]
    left.plot(depths, losses, marker="o", color=MAX_COLOR, linewidth=1.8)
    for depth, loss in zip(depths, losses):
        left.annotate(f"{loss:.3f}", xy=(depth, loss), textcoords="offset points",
                      xytext=(0, 7), ha="center", fontsize=7.5, color=MUTED)
    left.set_xlabel("depth (layers)")
    left.set_ylabel("validation loss")
    left.set_title("Depth sweep at maximum width")

    for rows, color, label in (
        (depth_rows, MAX_COLOR, "depth sweep"),
        (width_rows, MIN_COLOR, "width sweep"),
    ):
        params = [row["parameters"] / 1e6 for row in rows]
        values = [row["loss"] for row in rows]
        right.plot(params, values, marker="o", color=color, linewidth=1.6, label=label)
    right.set_xlabel("subnet parameters (millions)")
    right.set_ylabel("validation loss")
    right.set_title("Cost vs quality from the same shared weights")
    right.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def plot_ranking_scatter(result: dict, path: Path) -> Path:
    """The caveat: supernet ranking vs from-scratch ranking."""
    _style()
    rows = result["candidates"]
    fig, axis = plt.subplots(figsize=(6.2, 4.6))
    low = min(min(row["supernet_loss"] for row in rows), min(row["standalone_loss"] for row in rows))
    high = max(max(row["supernet_loss"] for row in rows), max(row["standalone_loss"] for row in rows))
    pad = (high - low) * 0.12 + 1e-3
    axis.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED,
              linestyle="--", linewidth=1.0, label="perfect agreement")
    axis.scatter(
        [row["supernet_loss"] for row in rows],
        [row["standalone_loss"] for row in rows],
        s=52,
        color="#4c7bd9",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for row in rows:
        axis.annotate(
            row["arch_id"][:6],
            xy=(row["supernet_loss"], row["standalone_loss"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color=MUTED,
        )
    best_supernet = min(rows, key=lambda row: row["supernet_loss"])
    best_standalone = min(rows, key=lambda row: row["standalone_loss"])
    axis.scatter([best_supernet["supernet_loss"]], [best_supernet["standalone_loss"]],
                 s=190, facecolor="none", edgecolor=MAX_COLOR, linewidth=1.8, zorder=2,
                 label="supernet top-1")
    axis.scatter([best_standalone["supernet_loss"]], [best_standalone["standalone_loss"]],
                 s=190, marker="s", facecolor="none", edgecolor=RANDOM_COLOR, linewidth=1.8,
                 zorder=2, label="standalone top-1")
    axis.set_xlim(low - pad, high + pad)
    axis.set_ylim(low - pad, high + pad)
    axis.set_xlabel("loss inside the supernet (shared weights)")
    axis.set_ylabel("loss after training this subnet alone")
    axis.set_title(
        f"Kendall tau = {result['kendall_tau']:.2f}, "
        f"Spearman = {result['spearman_rho']:.2f}, "
        f"regret = {result['regret']:+.3f}"
    )
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)
