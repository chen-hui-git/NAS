#!/usr/bin/env python3
"""Re-render the HTML report from an existing metrics.json and figures.

No training happens here: this is for editing the report wording and re-emitting the
same numbers. Run examples/run_demo.py to produce new numbers.

    PYTHONPATH=. python3 examples/render_report.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    # Redraw the schematic from the very spec the text shows, so figure and text can
    # never disagree even when metrics.json came from an older run.
    space_meta = metrics["run"]["space"]
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
    }
    out = build_report(context, Path(args.out))
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())