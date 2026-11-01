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


def plot_slicing_schematic(path: Path) -> Path:
    """Show that a subnet is a leading-channel slice of one shared tensor."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    total = 8
    used = 5
    for row in range(total):
        for column in range(total):
            active = row < used and column < used
            left.add_patch(
                Rectangle(
                    (column, total - row - 1),
                    1,
                    1,
                    facecolor="#4c7bd9" if active else "#eef1f6",
                    edgecolor="#c9d2e0" if not active else "#2b5bb0",
                    linewidth=0.7,
                )
            )
    left.add_patch(
        Rectangle((0, total - used), used, used, fill=False, edgecolor=MAX_COLOR, linewidth=2.4)
    )
    left.set_xlim(0, total)
    left.set_ylim(0, total)
    left.set_xticks([])
    left.set_yticks([])
    left.grid(False)
    left.set_xlabel("input channels (widest = 128)")
    left.set_ylabel("output channels")
    left.set_title("One shared weight tensor:\nthe subnet takes the leading block")
    left.annotate(
        "max subnet\nuses everything",
        xy=(total - 0.4, total - 0.5),
        ha="right",
        va="top",
        color=MUTED,
        fontsize=8,
    )
    left.annotate(
        "narrow subnet\nuses only this block",
        xy=(used + 0.15, total - used + 0.2),
        xytext=(used + 1.4, 1.2),
        color=MAX_COLOR,
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": MAX_COLOR, "lw": 1.2},
    )

    branches = ["r2k1", "r2k3", "r2k5", "r3k1", "r3k3", "r3k5", "r4k1", "r4k3", "r4k5"]
    chosen = "r3k3"
    for index, name in enumerate(branches):
        active = name == chosen
        right.add_patch(
            Rectangle(
                (0, index),
                3.0,
                0.8,
                facecolor=RANDOM_COLOR if active else "#eef1f6",
                edgecolor=RANDOM_COLOR if active else "#c9d2e0",
                linewidth=1.2 if active else 0.7,
            )
        )
        right.text(0.15, index + 0.4, f"conv1d branch {name}", va="center", fontsize=8,
                   color="white" if active else INK)
    right.set_xlim(0, 5.4)
    right.set_ylim(-0.2, len(branches) + 0.2)
    right.set_xticks([])
    right.set_yticks([])
    right.grid(False)
    right.set_title("Per-block FFN: every branch exists,\nthe subnet activates exactly one")
    right.annotate(
        "chosen by the sampled\narchitecture spec",
        xy=(3.0, branches.index(chosen) + 0.4),
        xytext=(3.4, branches.index(chosen) + 3.0),
        fontsize=8.5,
        color=RANDOM_COLOR,
        arrowprops={"arrowstyle": "->", "color": RANDOM_COLOR, "lw": 1.2},
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_sandwich_losses(history: list, path: Path, uniform_loss: float | None = None) -> Path:
    """Loss of each sandwich leg over training: max is low, min is high, random sits between."""
    _style()
    steps = [record.step for record in history]
    max_losses = [record.role_losses.get("max", float("nan")) for record in history]
    min_losses = [record.role_losses.get("min", float("nan")) for record in history]
    random_losses = []
    for record in history:
        values = [
            loss for role, loss in record.role_losses.items() if role.startswith("random")
        ]
        random_losses.append(sum(values) / len(values) if values else float("nan"))

    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    for values, color, label in (
        (max_losses, MAX_COLOR, "max subnet"),
        (min_losses, MIN_COLOR, "min subnet"),
        (random_losses, RANDOM_COLOR, "random subnets (mean)"),
    ):
        axis.plot(steps, values, color=color, alpha=0.28, linewidth=0.9)
        axis.plot(steps, _smooth(values), color=color, linewidth=1.9, label=label)
    if uniform_loss is not None:
        axis.axhline(uniform_loss, color=MUTED, linestyle=":", linewidth=1.2)
        axis.annotate(
            "uniform guess (log vocab)",
            xy=(steps[0], uniform_loss),
            xytext=(steps[0], uniform_loss + 0.12),
            fontsize=8,
            color=MUTED,
        )
    axis.set_xlabel("sandwich step (4 subnets per step)")
    axis.set_ylabel("masked cross-entropy")
    axis.set_title("Every step trains max + min + 2 random subnets in one optimizer step")
    axis.legend(loc="upper right")
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
