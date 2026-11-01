#!/usr/bin/env python3
"""End-to-end supernet demo: train with the sandwich rule, then teach from the result.

The run is organised around a wall-clock budget (default 300 s):

    load data -> build supernet -> warm up and measure step time -> plan the number of
    sandwich steps -> train -> evaluate the subnet family -> compare the supernet's
    ranking with from-scratch training -> draw figures -> render the HTML report

Run it with the interpreter that has torch installed, e.g.

    PYTHONPATH=. ../.venv/bin/python examples/run_demo.py

Fast mode (about a minute):

    PYTHONPATH=. ../.venv/bin/python examples/run_demo.py --budget 60 --steps 150 --candidates 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    run_ranking_experiment,
)
from supernet.search_space import LayerChoice


def say(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="shakespeare",
                        help="shakespeare (public text) or induction (synthetic, offline)")
    parser.add_argument("--budget", type=float, default=300.0,
                        help="wall-clock budget in seconds for the whole demo")
    parser.add_argument("--steps", type=int, default=2400,
                        help="upper bound on sandwich steps (the planner may use fewer)")
    parser.add_argument("--min-steps", type=int, default=60, help="lower bound on sandwich steps")
    parser.add_argument("--seq-len", type=int, default=32, help="tokens per training sequence")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-random", type=int, default=2,
                        help="random legs of the sandwich (max and min are always trained)")
    parser.add_argument("--candidates", type=int, default=10,
                        help="candidates in the supernet-vs-standalone ranking experiment")
    parser.add_argument("--standalone-steps", type=int, default=250,
                        help="training steps per standalone candidate")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--train-share", type=float, default=0.6,
                        help="fraction of the remaining budget spent on sandwich training")
    parser.add_argument("--report-reserve", type=float, default=45.0,
                        help="seconds reserved for the ranking experiment, figures and report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="report", help="output directory for figures and HTML")
    parser.add_argument("--samples", type=int, default=110, help="characters to generate per subnet")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--open", action="store_true", help="open the HTML report when finished")
    return parser.parse_args(argv)


def count_architectures(space: SearchSpace) -> int:
    """Exact number of distinct specs admitted by the space."""
    choices = len(space.ffn_ratios) * len(space.ffn_kernels) * len(space.activations)
    pairs = len(space.valid_dim_pairs())
    return sum(pairs * choices ** depth for depth in range(space.min_depth, space.max_depth + 1))


def head_dim_for(space: SearchSpace, embed_dim: int) -> int:
    candidates = [item for item in space.head_dims if embed_dim % item == 0]
    if not candidates:
        raise ValueError(f"no head_dim divides {embed_dim}")
    return min(candidates)


def uniform_quality_spec(space: SearchSpace, depth: int, embed_dim: int) -> SubnetSpec:
    """Fix the quality-maximising block choice; vary only depth or width."""
    layer = LayerChoice(max(space.ffn_ratios), max(space.ffn_kernels), space.activations[0])
    return SubnetSpec(
        depth=depth,
        embed_dim=embed_dim,
        head_dim=head_dim_for(space, embed_dim),
        layers=tuple(layer for _ in range(depth)),
    )


def generation_samples(model, task, space, specs: list[tuple[str, SubnetSpec]], count: int) -> list[dict]:
    """Qualitative check: sample text from different subnets inside the same weights."""
    if not hasattr(task, "stoi") or not hasattr(task, "decode"):
        return []
    prompt_text = "ROMEO:\n"
    prompt = torch.tensor(
        [task.stoi[char] for char in prompt_text if char in task.stoi], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(1234)
    samples = []
    for label, spec in specs:
        output = model.generate(spec, prompt, count, temperature=0.8, generator=generator)
        samples.append(
            {
                "label": label,
                "arch_id": spec.arch_id,
                "text": task.decode(output[0]),
            }
        )
    return samples


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

    say("=" * 68)
    say(f"超网 / 三明治采样教学 Demo    预算 {args.budget:.0f}s   数据集 {args.dataset}")
    say("=" * 68)

    # ---------------------------------------------------------------- data
    phase = time.monotonic()
    say("\n[1/7] 加载数据")
    task = make_task(args.dataset, seq_len=args.seq_len, verbose=True)
    say(f"      {task.name}: vocab={task.vocab_size} seq_len={task.seq_len} "
        f"({mark('data', phase):.1f}s)")
    for key, value in task.describe().items():
        if key not in {"name", "vocab_size", "seq_len"}:
            say(f"      {key} = {value}")

    # --------------------------------------------------------------- model
    phase = time.monotonic()
    say("\n[2/7] 构建超网")
    space = SearchSpace(vocab_size=task.vocab_size, max_seq_len=task.seq_len)
    trainer = SandwichTrainer(
        space, SandwichConfig(lr=args.lr, n_random=args.n_random, seed=args.seed)
    )
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
    warmup_steps = min(5, args.steps)
    for step in range(warmup_steps):
        tokens, targets, mask = batch_fn(step)
        trainer.train_step(tokens, targets, mask)
    warmup_seconds = time.monotonic() - phase
    step_seconds = max(warmup_seconds / warmup_steps, 1e-3)
    remaining = args.budget - (time.monotonic() - started)
    train_budget = max(remaining - args.report_reserve, remaining * 0.25) * args.train_share
    planned = int(train_budget / step_seconds)
    planned = max(args.min_steps, min(args.steps, planned))
    say(f"      单步耗时            : {step_seconds * 1000:.0f} ms "
        f"(一步 = 1 max + 1 min + {args.n_random} random)")
    say(f"      计划训练步数        : {planned} (~{planned * step_seconds:.0f}s)")

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
        random_mean = sum(
            value for role, value in record.role_losses.items() if role.startswith("random")
        ) / max(1, args.n_random)
        say(f"      [{bar}] {record.step:4d}/{planned} "
            f"total={record.total_loss:6.3f} "
            f"max={record.role_losses['max']:6.3f} "
            f"min={record.role_losses['min']:6.3f} "
            f"rand={random_mean:6.3f}")

    history = trainer.train(batch_fn, steps=planned - warmup_steps, log_every=log_every,
                            on_log=on_log)
    train_seconds = mark("train", phase)
    say(f"      训练完成            : {planned} 步 / {planned * (1 + 1 + args.n_random)} 个子网次 "
        f"({train_seconds:.0f}s)")

    # --------------------------------------------------------- evaluation
    phase = time.monotonic()
    say("\n[5/7] 评估子网族（不再训练）")
    batches = task.fixed_batches(args.batch_size, args.eval_batches, seed=args.seed + 101)
    uniform_loss = float(torch.log(torch.tensor(float(space.vocab_size))))
    max_loss = average_subnet_loss(trainer.model, max_spec, batches)
    min_loss = average_subnet_loss(trainer.model, min_spec, batches)
    say(f"      均匀猜测损失        : {uniform_loss:.3f} (log {space.vocab_size})")
    say(f"      max 子网损失        : {max_loss:.3f}")
    say(f"      min 子网损失        : {min_loss:.3f}")

    depth_sweep = []
    for depth in range(space.min_depth, space.max_depth + 1):
        spec = uniform_quality_spec(space, depth, space.max_embed_dim)
        depth_sweep.append(
            {
                "depth": depth,
                "arch_id": spec.arch_id,
                "parameters": trainer.model.subnet_parameter_count(spec),
                "loss": average_subnet_loss(trainer.model, spec, batches),
            }
        )
    width_sweep = []
    for embed_dim in space.embed_dims:
        spec = uniform_quality_spec(space, space.max_depth, embed_dim)
        width_sweep.append(
            {
                "embed_dim": embed_dim,
                "arch_id": spec.arch_id,
                "parameters": trainer.model.subnet_parameter_count(spec),
                "loss": average_subnet_loss(trainer.model, spec, batches),
            }
        )
    for row in depth_sweep:
        say(f"      depth {row['depth']:2d}  {row['parameters'] / 1e6:5.2f}M  "
            f"loss={row['loss']:.3f}")
    for row in width_sweep:
        say(f"      width {row['embed_dim']:3d}  {row['parameters'] / 1e6:5.2f}M  "
            f"loss={row['loss']:.3f}")

    intermediate = space.sample()
    eval_rows = [
        {"label": "max", "arch_id": max_spec.arch_id, "parameters": max_params, "loss": max_loss},
        {"label": "min", "arch_id": min_spec.arch_id, "parameters": min_params, "loss": min_loss},
        {
            "label": "random sample",
            "arch_id": intermediate.arch_id,
            "parameters": trainer.model.subnet_parameter_count(intermediate),
            "loss": average_subnet_loss(trainer.model, intermediate, batches),
        },
    ]

    say("\n      生成文本（同一套权重里的两个子网）")
    samples = generation_samples(
        trainer.model, task, space, [("max 子网", max_spec), ("min 子网", min_spec)], args.samples
    )
    for sample in samples:
        preview = sample["text"].replace("\n", " / ")[:110]
        say(f"      {sample['label']}: {preview}")
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
            "slicing": viz.plot_slicing_schematic(figures_dir / "slicing.png"),
            "sandwich": viz.plot_sandwich_losses(
                history, figures_dir / "sandwich_losses.png", uniform_loss
            ),
            "coverage": viz.plot_coverage(trainer.stats, figures_dir / "coverage.png"),
            "family": viz.plot_family_curve(depth_sweep, width_sweep, figures_dir / "family.png"),
            "ranking": viz.plot_ranking_scatter(ranking, figures_dir / "ranking.png"),
        }
        mark("figures", phase)

    metrics = {
        "run": {
            "started_at": started_at,
            "wall_seconds": round(time.monotonic() - started, 1),
            "timings": timings,
            "budget_seconds": args.budget,
            "steps_planned": planned,
            "steps_done": trainer.step_index,
            "step_ms": round(step_seconds * 1000, 1),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_random": args.n_random,
            "eval_batches": args.eval_batches,
            "standalone_steps": args.standalone_steps,
            "uniform_loss": uniform_loss,
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
        # The full per-step trace lives in sandwich_log.jsonl; keep the JSON light.
        "history": [record.to_dict() for record in history[::10]],

        "evals": eval_rows,
        "depth_sweep": depth_sweep,
        "width_sweep": width_sweep,
        "ranking": ranking,
        "samples": samples,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log_path = out_dir / "sandwich_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    report_path = None
    if not args.no_report and figure_paths:
        from supernet.report import build_report

        example = intermediate.to_dict()
        context = {
            "run": metrics["run"],
            "stats": trainer.stats.to_dict(),
            "evals": eval_rows,
            "depth_sweep": depth_sweep,
            "width_sweep": width_sweep,
            "ranking": ranking,
            "samples": samples,
            "figures": figure_paths,
            "example_spec": json.dumps(example, indent=2, ensure_ascii=False),
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
    say(f"  损失  均匀 / max / min: {uniform_loss:.3f} / {max_loss:.3f} / {min_loss:.3f}")
    say(f"  深度扫描            : " + "  ".join(
        f"d{row['depth']}={row['loss']:.2f}" for row in depth_sweep))
    say(f"  排序 Kendall tau    : {ranking['kendall_tau']:+.3f}  "
        f"(regret {ranking['regret']:+.3f})")
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
