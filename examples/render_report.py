#!/usr/bin/env python3
"""用已有的 metrics.json 与图片重新渲染 HTML 报告。

这里不训练：它只用于修改报告文字并重新输出同一批数字。
要产生新数字请跑 examples/run_demo.py。

    PYTHONPATH=. python3 examples/render_report.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supernet import SearchSpace, parameter_breakdown
from supernet.report import build_report

LOGGER = logging.getLogger("supernet")

FIGURE_FILES = {
    "slicing": "slicing.png",
    "sandwich": "sandwich_curves.png",
    "coverage": "coverage.png",
    "family": "family.png",
    "ranking": "ranking.png",
    "predictions": "predictions.png",
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="report/metrics.json")
    parser.add_argument("--figures", default="report/figures")
    parser.add_argument("--out", default="report/supernet_demo.html")
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} not found; run examples/run_demo.py first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    figure_dir = Path(args.figures)
    figures = {}
    for key, filename in FIGURE_FILES.items():
        path = figure_dir / filename
        if not path.exists():
            raise SystemExit(f"missing figure {path}; run examples/run_demo.py first")
        figures[key] = path

    example_spec = metrics.get("example_spec")
    if example_spec is None:
        example_spec = metrics["ranking"]["candidates"][0]["spec"]
        source = "ranking candidate (metrics.json has no example_spec)"
    else:
        source = "metrics.json example_spec"

    # 按正文所显示的同一个 spec 重画示意图，这样即使 metrics.json 来自旧运行，
    # 图与文字也不会互相矛盾。
    space_meta = metrics["run"]["space"]
    space = SearchSpace(
        min_depth=space_meta["min_depth"],
        max_depth=space_meta["max_depth"],
        embed_dims=tuple(space_meta["embed_dims"]),
        head_dims=tuple(space_meta["head_dims"]),
        ffn_ratios=tuple(space_meta["ffn_ratios"]),
        ffn_kernels=tuple(space_meta["ffn_kernels"]),
        activations=tuple(space_meta["activations"]),
        num_classes=space_meta["num_classes"],
        input_features=space_meta["input_features"],
        max_seq_len=space_meta["max_seq_len"],
    )
    from supernet import viz

    viz.plot_slicing_schematic(
        example_spec,
        figure_dir / FIGURE_FILES["slicing"],
        max_embed_dim=max(space_meta["embed_dims"]),
        ffn_ratios=tuple(space_meta["ffn_ratios"]),
        ffn_kernels=tuple(space_meta["ffn_kernels"]),
    )
    LOGGER.info("schematic redrawn from %s: %s", source, example_spec.get("arch_id", "?"))

    context = {
        "run": metrics["run"],
        "stats": metrics["sandwich"],
        "evals": metrics["evals"],
        "depth_sweep": metrics["depth_sweep"],
        "width_sweep": metrics["width_sweep"],
        "ranking": metrics["ranking"],
        "predictions": metrics["predictions"],
        "figures": figures,
        "example_spec": json.dumps(example_spec, indent=2, ensure_ascii=False),
        "fine_tune": metrics.get("fine_tune", {}),
        "breakdown": metrics["run"]["params"].get("breakdown")
        or parameter_breakdown(space),
    }
    out = build_report(context, Path(args.out))
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())