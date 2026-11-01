#!/usr/bin/env python3
"""End-to-end supernet demo: train with the sandwich rule, then teach from the result.

Task: MNIST digit classification, where each of the 28 image rows is one token. The run
is organised around a wall-clock budget (default 300 s):

    load data -> build supernet -> warm up and measure step time -> plan the number of
    sandwich steps -> train -> evaluate the subnet family -> compare the supernet's
    ranking with from-scratch training -> draw figures -> render the HTML report

Run it with the interpreter that has torch installed, e.g.

    PYTHONPATH=. ../.venv/bin/python examples/run_demo.py

Fast mode (about a minute):

    PYTHONPATH=. ../.venv/bin/python examples/run_demo.py --budget 60 --steps 300 --candidates 4
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supernet import (
    RankingExperimentConfig,
    SandwichConfig,
    SandwichTrainer,
    SearchSpace,
    SubnetSpec,
    average_subnet_loss,
    make_task,
    normalize_images,
    run_ranking_experiment,
    subnet_accuracy,
)
from supernet.search_space import LayerChoice


def say(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist",
                        help="mnist (public handwritten digits) or bars (synthetic, offline)")
    parser.add_argument("--budget", type=float, default=300.0,
                        help="wall-clock budget in seconds for the whole demo")
    parser.add_argument("--steps", type=int, default=2400,
                        help="upper bound on sandwich steps (the planner may use fewer)")
    parser.add_argument("--min-steps", type=int, default=60, help="lower bound on sandwich steps")
    parser.add_argument("--device", default="auto",
                        help="auto (mps when available, else cpu), cpu or mps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default 128 on mps, 8 on cpu: measured, not guessed")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--deterministic", action="store_true",
                        help="ask torch for deterministic kernels (cpu: usually fine, mps: may refuse)")
    parser.add_argument("--shrink-stages", type=int, default=0,
                        help="progressive shrinking stages; 0 = off (joint training), 4 = OFA style")
    parser.add_argument("--distill-weight", type=float, default=0.5,
                        help="KL weight towards the max leg (in-place distillation); "
                             "0 = off. 0.5 is the measured default: same wall clock, "
                             "small/random subnets +2 to +4 points")
    parser.add_argument("--n-random", type=int, default=2,
                        help="random legs of the sandwich (max and min are always trained)")
    parser.add_argument("--candidates", type=int, default=10,
                        help="candidates in the supernet-vs-standalone ranking experiment")
    parser.add_argument("--standalone-steps", type=int, default=None,
                        help="training steps per standalone candidate; default keeps the "
                             "audit at ~16000 images per candidate (= 250 steps at batch 64)")
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--train-share", type=float, default=0.6,
                        help="fraction of the remaining budget spent on sandwich training")
    parser.add_argument("--report-reserve", type=float, default=45.0,
                        help="seconds reserved for the ranking experiment, figures and report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="report", help="output directory for figures and HTML")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--save-weights", action="store_true",
                        help="write the trained supernet to <out>/supernet.pt (~74 MB) "
                             "so subnets can be exported and fine-tuned later")
    parser.add_argument("--open", action="store_true", help="open the HTML report when finished")
    return parser.parse_args(argv)


def git_state() -> str:
    """Short HEAD hash (+dirty when the worktree differs), so a report ties to real code."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        # Only source files count: a run rewrites report/ artifacts, and those should
        # not make the code look modified.
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--",
             "supernet", "tests", "examples", "docs", "README.md", ".gitignore"],
            cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def resolve_device(requested: str) -> str:
    """Pick the fastest available device; mps is Apple GPU, cpu is the safe default."""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("mps requested but torch.backends.mps.is_available() is False")
    return requested


def count_architectures(space: SearchSpace) -> int:
    """Exact number of distinct specs admitted by the space."""
    choices = len(space.ffn_ratios) * len(space.ffn_kernels) * len(space.activations)
    pairs = len(space.valid_dim_pairs())
    return sum(
        pairs * choices ** depth
        for depth in range(space.min_depth, space.max_depth + 1)
    )


def head_dim_for(space: SearchSpace, embed_dim: int) -> int:
    candidates = [item for item in space.head_dims if embed_dim % item == 0]
    if not candidates:
        raise ValueError(f"no head_dim divides {embed_dim}")
    return min(candidates)


def quality_spec(space: SearchSpace, depth: int, embed_dim: int) -> SubnetSpec:
    """Fix the quality-maximising block choice; vary only depth or width."""
    layer = LayerChoice(max(space.ffn_ratios), max(space.ffn_kernels), space.activations[0])
    return SubnetSpec(
        depth=depth,
        embed_dim=embed_dim,
        head_dim=head_dim_for(space, embed_dim),
        layers=tuple(layer for _ in range(depth)),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = Path(args.out)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def mark(name: str, since: float) -> float:
        elapsed = time.monotonic() - since
        timings[name] = round(elapsed, 2)
        return elapsed

    device = resolve_device(args.device)
    deterministic = False
    if args.deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            deterministic = True
        except Exception as error:  # backend refused: keep going, but record it
            print(f"  warning: deterministic algorithms unavailable ({error})")
    # Measured on an M5 Pro: batch 64 and 128 cost almost the same per step
    # (100 vs 134 ms) while 128 carries twice the images, so 128 is the better default.
    batch_size = args.batch_size if args.batch_size is not None else (
        128 if device == "mps" else 8
    )
    args.batch_size = batch_size
    if args.standalone_steps is None:
        # The audit compares candidates under an equal training budget, so keep the
        # *image* budget fixed and let the step count follow the batch size.
        audit_images = 16000
        args.standalone_steps = max(20, round(audit_images / batch_size))
    say("=" * 68)
    say(f"超网 / 三明治采样教学 Demo    预算 {args.budget:.0f}s   数据集 {args.dataset}")
    say(f"设备 {device}   batch {batch_size}   "
        f"审计: {args.candidates} 候选 x {args.standalone_steps} 步 "
        f"= {args.candidates * args.standalone_steps * batch_size:,} 图")
    say(f"渐进收缩 shrink_stages={args.shrink_stages}"
        f"{'（关闭）' if args.shrink_stages <= 1 else f'（{args.shrink_stages} 级）'}   "
        f"子网蒸馏 distill_weight={args.distill_weight}"
        f"{'（关闭）' if args.distill_weight <= 0 else ''}")
    say("=" * 68)

    # ---------------------------------------------------------------- data
    phase = time.monotonic()
    say("\n[1/7] 加载数据")
    task = make_task(args.dataset, verbose=True, device=device)
    say(f"      {task.name}: 输入 {task.input_features} 维 x {task.seq_len} 行, "
        f"{task.num_classes} 类 ({mark('data', phase):.1f}s)")
    for key, value in task.describe().items():
        if key not in {"name", "num_classes", "input_features", "seq_len"}:
            say(f"      {key} = {value}")

    # --------------------------------------------------------------- model
    phase = time.monotonic()
    say("\n[2/7] 构建超网")
    space = SearchSpace(
        num_classes=task.num_classes,
        input_features=task.input_features,
        max_seq_len=task.seq_len,
    )
    trainer = SandwichTrainer(
        space,
        SandwichConfig(
            lr=args.lr,
            n_random=args.n_random,
            seed=args.seed,
            shrink_stages=args.shrink_stages,
            distill_weight=args.distill_weight,
        ),
    )
    trainer.model.to(device)
    max_spec = space.max_subnet()
    min_spec = space.min_subnet()
    supernet_params = sum(parameter.numel() for parameter in trainer.model.parameters())
    max_params = trainer.model.subnet_parameter_count(max_spec)
    min_params = trainer.model.subnet_parameter_count(min_spec)
    say(f"      空间候选数          : {count_architectures(space):,}")
    say(f"      超网参数量          : {supernet_params / 1e6:.2f}M")
    say(f"      max 子网 (d{space.max_depth}/{space.max_embed_dim}) : {max_params / 1e6:.2f}M")
    say(f"      min 子网 (d{space.min_depth}/{space.min_embed_dim})  : {min_params / 1e6:.3f}M")
    say(f"      一份权重供应        : {supernet_params / max_params:.1f}x 于最大子网 "
        f"({mark('model', phase):.2f}s)")

    # --------------------------------------------------- warm up and plan
    phase = time.monotonic()
    say("\n[3/7] 热身测速，规划步数")
    batch_fn = task.batch_fn(args.batch_size, seed=args.seed)
    warmup_steps = 5
    timed_steps = 5
    # Warm-up steps are thrown away on purpose: the first steps on mps pay kernel
    # compilation, and averaging them in makes the planner believe every step costs
    # ~20x what it really does.
    for step in range(warmup_steps):
        inputs, labels = batch_fn(step)
        trainer.train_step(inputs, labels)
    timings_per_step = []
    for step in range(timed_steps):
        inputs, labels = batch_fn(warmup_steps + step)
        tick = time.perf_counter()
        trainer.train_step(inputs, labels)
        timings_per_step.append(time.perf_counter() - tick)
    timings_per_step.sort()
    step_seconds = max(timings_per_step[len(timings_per_step) // 2], 1e-3)
    remaining = args.budget - (time.monotonic() - started)
    train_budget = max(remaining - args.report_reserve, remaining * 0.25) * args.train_share
    # Progressive shrinking makes the first stages cost more per step: all four legs are
    # the largest subnet (measured 1.75x). 1.4x is the stage-weighted average.
    cost_factor = 1.4 if args.shrink_stages > 1 else 1.0
    planned = int(train_budget / (step_seconds * cost_factor))
    planned = max(args.min_steps, min(args.steps, planned))
    say(f"      单步耗时（中位数）  : {step_seconds * 1000:.0f} ms "
        f"(一步 = 1 max + 1 min + {args.n_random} random, "
        f"{args.batch_size / step_seconds:.0f} 图/秒)")
    say(f"      计划训练步数        : {planned} "
        f"(~{planned * args.batch_size} 张图, ~{planned * step_seconds:.0f}s)")

    # ------------------------------------------------------------ training
    phase = time.monotonic()
    say("\n[4/7] 三明治训练")
    log_every = max(1, planned // 12)
    history = []

    def on_log(record) -> None:
        fraction = record.step / planned
        bar_length = 24
        filled = int(bar_length * fraction)
        bar = "#" * filled + "." * (bar_length - filled)
        random_loss = sum(
            value for role, value in record.role_losses.items() if role.startswith("random")
        ) / max(1, args.n_random)
        random_acc = sum(
            value for role, value in record.role_accuracy.items()
            if role.startswith("random")
        ) / max(1, args.n_random)
        say(f"      [{bar}] {record.step:4d}/{planned} "
            f"max loss={record.role_losses['max']:5.3f} acc={record.role_accuracy['max']:.3f}  "
            f"min loss={record.role_losses['min']:5.3f} acc={record.role_accuracy['min']:.3f}  "
            f"rand loss={random_loss:5.3f} acc={random_acc:.3f}")

    history = trainer.train(
        batch_fn,
        steps=max(0, planned - warmup_steps - timed_steps),
        log_every=log_every,
        on_log=on_log,
        deadline=started + max(args.budget - args.report_reserve, args.budget * 0.6),
    )
    train_seconds = mark("train", phase)
    say(f"      训练完成            : {planned} 步 / "
        f"{planned * (1 + 1 + args.n_random)} 个子网次 ({train_seconds:.0f}s)")

    # --------------------------------------------------------- evaluation
    phase = time.monotonic()
    say("\n[5/7] 评估子网族（不再训练）")
    batches = task.fixed_batches(args.batch_size, args.eval_batches, seed=args.seed + 101)
    chance_loss = math.log(task.num_classes)
    chance_accuracy = 1.0 / task.num_classes
    max_loss = average_subnet_loss(trainer.model, max_spec, batches)
    max_acc = subnet_accuracy(trainer.model, max_spec, batches)
    min_loss = average_subnet_loss(trainer.model, min_spec, batches)
    min_acc = subnet_accuracy(trainer.model, min_spec, batches)
    say(f"      随机猜测基线        : loss {chance_loss:.3f} / acc {chance_accuracy:.3f}")
    say(f"      max 子网            : loss {max_loss:.3f} / acc {max_acc:.3f}")
    say(f"      min 子网            : loss {min_loss:.3f} / acc {min_acc:.3f}")

    depth_sweep = []
    for depth in range(space.min_depth, space.max_depth + 1):
        spec = quality_spec(space, depth, space.max_embed_dim)
        depth_sweep.append(
            {
                "depth": depth,
                "arch_id": spec.arch_id,
                "parameters": trainer.model.subnet_parameter_count(spec),
                "loss": average_subnet_loss(trainer.model, spec, batches),
                "accuracy": subnet_accuracy(trainer.model, spec, batches),
            }
        )
    width_sweep = []
    for embed_dim in space.embed_dims:
        spec = quality_spec(space, space.max_depth, embed_dim)
        width_sweep.append(
            {
                "embed_dim": embed_dim,
                "arch_id": spec.arch_id,
                "parameters": trainer.model.subnet_parameter_count(spec),
                "loss": average_subnet_loss(trainer.model, spec, batches),
                "accuracy": subnet_accuracy(trainer.model, spec, batches),
            }
        )
    for row in depth_sweep:
        say(f"      depth {row['depth']:2d}  {row['parameters'] / 1e6:5.2f}M  "
            f"loss={row['loss']:.3f}  acc={row['accuracy']:.3f}")
    for row in width_sweep:
        say(f"      width {row['embed_dim']:3d}  {row['parameters'] / 1e6:5.2f}M  "
            f"loss={row['loss']:.3f}  acc={row['accuracy']:.3f}")

    # A fixed generator instead of the global random module: this row and the example
    # spec must be reproducible from --seed alone.
    intermediate = space.sample(random.Random(args.seed + 7))
    example_spec = intermediate.to_dict()
    eval_rows = [
        {
            "label": "max",
            "arch_id": max_spec.arch_id,
            "parameters": max_params,
            "loss": max_loss,
            "accuracy": max_acc,
        },
        {
            "label": "min",
            "arch_id": min_spec.arch_id,
            "parameters": min_params,
            "loss": min_loss,
            "accuracy": min_acc,
        },
        {
            "label": "random sample",
            "arch_id": intermediate.arch_id,
            "parameters": trainer.model.subnet_parameter_count(intermediate),
            "loss": average_subnet_loss(trainer.model, intermediate, batches),
            "accuracy": subnet_accuracy(trainer.model, intermediate, batches),
        },
    ]

    preview_images, preview_labels = task.preview()
    preview_inputs = normalize_images(preview_images).to(device)
    predictions = {
        "max": trainer.model.predict(preview_inputs, max_spec).tolist(),
        "min": trainer.model.predict(preview_inputs, min_spec).tolist(),
    }
    say("\n      预测示例（第一张是 max 子网，第二张是 min 子网）")
    say(f"      真实    : {preview_labels.tolist()}")
    say(f"      max     : {predictions['max']}")
    say(f"      min     : {predictions['min']}")
    mark("eval", phase)

    # ------------------------------------------------------------- ranking
    phase = time.monotonic()
    say("\n[6/7] 超网内排序 vs 独立训练排序")
    ranking_config = RankingExperimentConfig(
        num_candidates=args.candidates,
        standalone_steps=args.standalone_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    deadline = started + args.budget - 20.0

    def on_candidate(index: int, total: int, spec) -> None:
        say(f"      独立训练候选 {index + 1}/{total}  "
            f"depth={spec.depth} width={spec.embed_dim} id={spec.arch_id}")

    ranking = run_ranking_experiment(
        trainer.model, space, task, ranking_config,
        progress=on_candidate, deadline=deadline, min_candidates=3,
    )
    say(f"      Kendall tau = {ranking['kendall_tau']:+.3f}   "
        f"Spearman rho = {ranking['spearman_rho']:+.3f}   "
        f"top1 一致 = {ranking['top1_agreement']}   "
        f"regret = {ranking['regret']:+.3f}")
    mark("ranking", phase)

    # ---------------------------------------------------- figures + report
    phase = time.monotonic()
    figure_paths = {}
    if not args.no_figures:
        say("\n[7/7] 绘图与报告")
        from supernet import viz

        figure_paths = {
            "slicing": viz.plot_slicing_schematic(
                example_spec,
                figures_dir / "slicing.png",
                max_embed_dim=space.max_embed_dim,
                ffn_ratios=space.ffn_ratios,
                ffn_kernels=space.ffn_kernels,
            ),
            "sandwich": viz.plot_sandwich_curves(
                history, figures_dir / "sandwich_curves.png", chance_loss
            ),
            "coverage": viz.plot_coverage(trainer.stats, figures_dir / "coverage.png"),
            "family": viz.plot_family_curve(depth_sweep, width_sweep, figures_dir / "family.png"),
            "ranking": viz.plot_ranking_scatter(ranking, figures_dir / "ranking.png"),
            "predictions": viz.plot_predictions(
                preview_images, preview_labels.tolist(),
                {"max": predictions["max"], "min": predictions["min"]},
                figures_dir / "predictions.png",
            ),
        }
        mark("figures", phase)

    metrics = {
        "run": {
            "started_at": started_at,
            "wall_seconds": round(time.monotonic() - started, 1),
            "timings": timings,
            "seed": args.seed,
            "deterministic": deterministic,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_state(),
            "budget_seconds": args.budget,
            "steps_planned": planned,
            "steps_done": trainer.step_index,
            "step_ms": round(step_seconds * 1000, 1),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_random": args.n_random,
            "shrink_stages": args.shrink_stages,
            "distill_weight": args.distill_weight,
            "shrink_schedule": [stage.to_dict() for stage in trainer.schedule],
            "stage_reached": max((record.stage for record in history), default=0),
            "eval_batches": args.eval_batches,
            "standalone_steps": args.standalone_steps,
            "device": device,
            "chance_loss": chance_loss,
            "chance_accuracy": chance_accuracy,
            "space_size": count_architectures(space),
            "dataset": task.describe(),
            "space": space.to_dict(),
            "params": {
                "supernet": supernet_params,
                "max": max_params,
                "min": min_params,
                "shared_ratio": supernet_params / max_params,
            },
            "command": "PYTHONPATH=. ../.venv/bin/python examples/run_demo.py "
            + " ".join(sys.argv[1:]),
        },
        "sandwich": trainer.stats.to_dict(),
        "history": [record.to_dict() for record in history[::10]],
        "evals": eval_rows,
        "depth_sweep": depth_sweep,
        "width_sweep": width_sweep,
        "ranking": ranking,
        "predictions": {
            "true": preview_labels.tolist(),
            **predictions,
        },
        "example_spec": example_spec,
    }
    weights_path = None
    if args.save_weights:
        weights_path = out_dir / "supernet.pt"
        torch.save(trainer.model.state_dict(), weights_path)
        say(f"\n      supernet 权重已保存 : {weights_path} "
            f"({weights_path.stat().st_size / 1e6:.0f} MB)")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path = out_dir / "sandwich_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    report_path = None
    if not args.no_report and figure_paths:
        from supernet.report import build_report

        context = {
            "run": metrics["run"],
            "stats": trainer.stats.to_dict(),
            "evals": eval_rows,
            "depth_sweep": depth_sweep,
            "width_sweep": width_sweep,
            "ranking": ranking,
            "predictions": metrics["predictions"],
            "figures": figure_paths,
            "example_spec": json.dumps(example_spec, indent=2, ensure_ascii=False),
        }
        report_path = build_report(context, out_dir / "supernet_demo.html")
        mark("report", phase)

    # ------------------------------------------------------------ summary
    say("\n" + "=" * 68)
    say("结果")
    say("=" * 68)
    say(f"  总耗时              : {time.monotonic() - started:.0f}s   (明细: {timings})")
    say(f"  超网 / max / min     : {supernet_params / 1e6:.2f}M / "
        f"{max_params / 1e6:.2f}M / {min_params / 1e3:.0f}K 参数")
    say(f"  准确率 猜测 / max / min: {chance_accuracy:.3f} / {max_acc:.3f} / {min_acc:.3f}")
    say(f"  深度扫描            : " + "  ".join(
        f"d{row['depth']}={row['accuracy']:.3f}" for row in depth_sweep))
    say(f"  宽度扫描            : " + "  ".join(
        f"w{row['embed_dim']}={row['accuracy']:.3f}" for row in width_sweep))
    say(f"  排序 Kendall tau    : {ranking['kendall_tau']:+.3f}  "
        f"(regret {ranking['regret']:+.3f})")
    if weights_path is not None:
        say(f"  超网权重            : {weights_path}")
    say(f"  指标 JSON           : {metrics_path}")
    say(f"  训练日志            : {log_path}")
    if figure_paths:
        say(f"  图                  : {figures_dir}")
    if report_path is not None:
        say(f"  教学报告 (HTML)     : {report_path}")
    say("")

    if args.open and report_path is not None:
        import webbrowser

        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
