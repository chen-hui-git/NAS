#!/usr/bin/env python3
"""端到端的超网实践：用三明治规则训练，然后把整族数字读出来。

任务：MNIST 数字分类，28 行图像各是一个 token。整轮运行以墙钟预算（默认 300 秒）为轴：

    加载数据 -> 构建超网 -> 热身测速 -> 规划三明治步数 -> 训练 -> 评估子网族 ->
    与"从零独立训练"对比排序 -> 出图 -> 渲染 HTML 报告

用法（用装了 torch 的解释器）：

    PYTHONPATH=. ../.venv/bin/python examples/run_demo.py

快速模式（约一分钟）：

    PYTHONPATH=. ../.venv/bin/python examples/run_demo.py --budget 60 --steps 300 --candidates 4
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supernet.evaluation import audit_ranking, evaluate_family, fine_tune_report
from supernet import (
    SandwichConfig,
    SandwichTrainer,
    SearchSpace,
    make_task,
    parameter_breakdown,
)


LOGGER = logging.getLogger("supernet")


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """把运行进度交给 logging 输出。

    控制台保持原有的分栏排版（只有消息本身），同时另开一个文件 handler 记录带时间戳的
    完整日志；日志级别仍是常规过滤器，--log-level WARNING 会静默进度行。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
        )
        handlers.append(file_handler)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


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
                        help="progressive shrinking stages; 0 = off (joint training). "
                             "N = N equal-length segments unlocking ALL dims at once; "
                             "OFA's own scheme is 5 stages, one dimension at a time "
                             "(see docs/ofa_gap.md)")
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
    parser.add_argument("--report-reserve", type=float, default=70.0,
                        help="seconds reserved for ranking audit, fine-tuning, figures and report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="report", help="output directory for figures and HTML")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--fine-tune-steps", type=int, default=150,
                        help="steps to specialize each exported subnet (0 = skip); measured "
                             "to cost ~15-20 s and to lift small subnets a few points")
    parser.add_argument("--save-weights", action="store_true",
                        help="write the trained supernet to <out>/supernet.pt (~74 MB) "
                             "so subnets can be exported and fine-tuned later")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="logging verbosity for the run's progress output")
    parser.add_argument("--log-file", default="run.log",
                        help="file name inside --out that receives the timestamped log "
                             "(empty string disables it)")
    parser.add_argument("--open", action="store_true", help="open the HTML report when finished")
    return parser.parse_args(argv)


def git_state() -> str:
    """返回短 HEAD 哈希（工作区有改动时带 dirty 标记），让报告能对应到真实代码。"""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        # 只看源码文件：每次运行都会重写 report/ 下的产物，那些不应该让代码显得被改动过。
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--",
             "supernet", "tests", "examples", "docs", "README.md", ".gitignore"],
            cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def resolve_device(requested: str) -> str:
    """选出最快的可用设备；mps 是 Apple GPU，cpu 是稳妥的默认值。"""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("mps requested but torch.backends.mps.is_available() is False")
    return requested


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = Path(args.out)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / args.log_file if args.log_file else None
    configure_logging(args.log_level, log_file)
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
            LOGGER.warning("deterministic algorithms unavailable (%s)", error)
    # 在 M5 Pro 上实测：batch 64 与 128 的单步耗时几乎相同（100 vs 134 ms），
    # 但 128 每步喂两倍图片，所以默认取 128。
    batch_size = args.batch_size if args.batch_size is not None else (
        128 if device == "mps" else 8
    )
    args.batch_size = batch_size
    if args.standalone_steps is None:
        # 审计是在"相同训练预算"下比较候选，所以固定的是**图片数**预算，步数随 batch 反推。
        audit_images = 16000
        args.standalone_steps = max(20, round(audit_images / batch_size))
    LOGGER.info("=" * 68)
    LOGGER.info(f"超网实践：三明治规则 + 子网族    预算 {args.budget:.0f}s   数据集 {args.dataset}")
    LOGGER.info(f"设备 {device}   batch {batch_size}   "
        f"审计: {args.candidates} 候选 x {args.standalone_steps} 步 "
        f"= {args.candidates * args.standalone_steps * batch_size:,} 图")
    LOGGER.info(f"渐进收缩 shrink_stages={args.shrink_stages}"
        f"{'（关闭）' if args.shrink_stages <= 1 else f'（{args.shrink_stages} 级）'}   "
        f"子网蒸馏 distill_weight={args.distill_weight}"
        f"{'（关闭）' if args.distill_weight <= 0 else ''}")
    LOGGER.info("=" * 68)

    # ---------------------------------------------------------------- 数据
    phase = time.monotonic()
    LOGGER.info("\n[1/7] 加载数据")
    task = make_task(args.dataset, verbose=True, device=device)
    LOGGER.info(f"      {task.name}: 输入 {task.input_features} 维 x {task.seq_len} 行, "
        f"{task.num_classes} 类 ({mark('data', phase):.1f}s)")
    for key, value in task.describe().items():
        if key not in {"name", "num_classes", "input_features", "seq_len"}:
            LOGGER.info(f"      {key} = {value}")

    # --------------------------------------------------------------- 模型
    phase = time.monotonic()
    LOGGER.info("\n[2/7] 构建超网")
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
    LOGGER.info(f"      空间候选数          : {space.architecture_count():,}")
    LOGGER.info(f"      超网参数量          : {supernet_params / 1e6:.2f}M")
    LOGGER.info(f"      max 子网 (d{space.max_depth}/{space.max_embed_dim}) : {max_params / 1e6:.2f}M")
    LOGGER.info(f"      min 子网 (d{space.min_depth}/{space.min_embed_dim})  : {min_params / 1e6:.3f}M")
    LOGGER.info(f"      一份权重供应        : {supernet_params / max_params:.1f}x 于最大子网 "
        f"({mark('model', phase):.2f}s)")

    # --------------------------------------------------- 热身测速与步数规划
    phase = time.monotonic()
    LOGGER.info("\n[3/7] 热身测速，规划步数")
    batch_fn = task.batch_fn(args.batch_size, seed=args.seed)
    warmup_steps = 5
    timed_steps = 5
    # 热身步是故意丢弃的：mps 上最初几步要付 kernel 编译成本，
    # 把它们算进平均值会让规划器误以为每步贵出约 20 倍。
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
    # 渐进收缩会让前几个阶段每步更贵：四条腿都是最大子网（实测 1.75 倍）。
    # 这里取按阶段加权的平均值 1.4 倍作为估计。
    cost_factor = 1.4 if args.shrink_stages > 1 else 1.0
    planned = int(train_budget / (step_seconds * cost_factor))
    planned = max(args.min_steps, min(args.steps, planned))
    LOGGER.info(f"      单步耗时（中位数）  : {step_seconds * 1000:.0f} ms "
        f"(一步 = 1 max + 1 min + {args.n_random} random, "
        f"{args.batch_size / step_seconds:.0f} 图/秒)")
    LOGGER.info(f"      计划训练步数        : {planned} "
        f"(~{planned * args.batch_size} 张图, ~{planned * step_seconds:.0f}s)")

    # ------------------------------------------------------------ 训练
    phase = time.monotonic()
    LOGGER.info("\n[4/7] 三明治训练")
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
        LOGGER.info(f"      [{bar}] {record.step:4d}/{planned} "
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
    LOGGER.info(f"      训练完成            : {planned} 步 / "
        f"{planned * (1 + 1 + args.n_random)} 个子网次 ({train_seconds:.0f}s)")

    # --------------------------------------------------------- 评估子网族
    phase = time.monotonic()
    LOGGER.info("\n[5/7] 评估子网族（不再训练）")
    report = evaluate_family(
        trainer.model,
        space,
        task,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        seed=args.seed + 101,
    )
    max_row, min_row = report.evals[0], report.evals[1]
    example_spec = report.example_dict
    LOGGER.info(f"      随机猜测基线        : loss {report.chance_loss:.3f} / acc {report.chance_accuracy:.3f}")
    for row in report.evals:
        LOGGER.info(f"      {row['label']:12s} 子网      : loss {row['loss']:.3f} / acc {row['accuracy']:.3f}")
    for row in report.depth_sweep:
        LOGGER.info(f"      depth {row['depth']:2d}  {row['parameters'] / 1e6:5.2f}M  "
                    f"loss={row['loss']:.3f}  acc={row['accuracy']:.3f}")
    for row in report.width_sweep:
        LOGGER.info(f"      width {row['embed_dim']:3d}  {row['parameters'] / 1e6:5.2f}M  "
                    f"loss={row['loss']:.3f}  acc={row['accuracy']:.3f}")

    LOGGER.info(f"\n      微调导出子网（各 {args.fine_tune_steps} 步，lr=3e-4）")
    fine_tune_rows = fine_tune_report(
        trainer.model, space, task, report,
        steps=args.fine_tune_steps, batch_size=args.batch_size, seed=args.seed + 31,
        log=LOGGER.info,
    )

    LOGGER.info("\n      预测示例（第一张是 max 子网，第二张是 min 子网）")
    LOGGER.info(f"      真实    : {report.predictions['true']}")
    LOGGER.info(f"      max     : {report.predictions['max']}")
    LOGGER.info(f"      min     : {report.predictions['min']}")
    depth_sweep, width_sweep, eval_rows = report.depth_sweep, report.width_sweep, report.evals
    mark("eval", phase)

    # ------------------------------------------------------------- 排序审计
    phase = time.monotonic()
    LOGGER.info("\n[6/7] 超网内排序 vs 独立训练排序")
    ranking = audit_ranking(
        trainer.model, space, task,
        candidates=args.candidates,
        standalone_steps=args.standalone_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        deadline=started + args.budget - 20.0,
        log=LOGGER.info,
    )
    LOGGER.info(f"      Kendall tau = {ranking['kendall_tau']:+.3f}   "
                f"Spearman rho = {ranking['spearman_rho']:+.3f}   "
                f"top1 一致 = {ranking['top1_agreement']}   "
                f"regret = {ranking['regret']:+.3f}")
    mark("ranking", phase)

    # ---------------------------------------------------- 出图与报告
    phase = time.monotonic()
    figure_paths = {}
    if not args.no_figures:
        LOGGER.info("\n[7/7] 绘图与报告")
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
                history, figures_dir / "sandwich_curves.png", report.chance_loss
            ),
            "coverage": viz.plot_coverage(trainer.stats, figures_dir / "coverage.png"),
            "family": viz.plot_family_curve(depth_sweep, width_sweep, figures_dir / "family.png"),
            "ranking": viz.plot_ranking_scatter(ranking, figures_dir / "ranking.png"),
            "predictions": viz.plot_predictions(
                report.preview_images,
                report.predictions["true"],
                {"max": report.predictions["max"], "min": report.predictions["min"]},
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
            "chance_loss": report.chance_loss,
            "chance_accuracy": report.chance_accuracy,
            "space_size": space.architecture_count(),
            "dataset": task.describe(),
            "space": space.to_dict(),
            "params": {
                "breakdown": parameter_breakdown(space),
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
        "predictions": report.predictions,
        "fine_tune": {
            "steps": args.fine_tune_steps,
            "lr": 3e-4,
            "rows": fine_tune_rows,
        },
        "example_spec": example_spec,
    }
    weights_path = None
    if args.save_weights:
        weights_path = out_dir / "supernet.pt"
        torch.save(trainer.model.state_dict(), weights_path)
        LOGGER.info(f"\n      supernet 权重已保存 : {weights_path} "
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
            "fine_tune": metrics["fine_tune"],
            "breakdown": metrics["run"]["params"].get("breakdown")
            or parameter_breakdown(space),
            "figures": figure_paths,
            "example_spec": json.dumps(example_spec, indent=2, ensure_ascii=False),
        }
        report_path = build_report(context, out_dir / "supernet_demo.html")
        mark("report", phase)

    # ------------------------------------------------------------ 汇总
    LOGGER.info("\n" + "=" * 68)
    LOGGER.info("结果")
    LOGGER.info("=" * 68)
    LOGGER.info(f"  总耗时              : {time.monotonic() - started:.0f}s   (明细: {timings})")
    LOGGER.info(f"  超网 / max / min     : {supernet_params / 1e6:.2f}M / "
        f"{max_params / 1e6:.2f}M / {min_params / 1e3:.0f}K 参数")
    LOGGER.info(f"  准确率 猜测 / max / min: {report.chance_accuracy:.3f} / "
                f"{max_row['accuracy']:.3f} / {min_row['accuracy']:.3f}")
    LOGGER.info(f"  深度扫描            : " + "  ".join(
        f"d{row['depth']}={row['accuracy']:.3f}" for row in depth_sweep))
    LOGGER.info(f"  宽度扫描            : " + "  ".join(
        f"w{row['embed_dim']}={row['accuracy']:.3f}" for row in width_sweep))
    LOGGER.info(f"  排序 Kendall tau    : {ranking['kendall_tau']:+.3f}  "
        f"(regret {ranking['regret']:+.3f})")
    if weights_path is not None:
        LOGGER.info(f"  超网权重            : {weights_path}")
    LOGGER.info(f"  指标 JSON           : {metrics_path}")
    LOGGER.info(f"  训练日志            : {log_path}")
    if figure_paths:
        LOGGER.info(f"  图                  : {figures_dir}")
    if report_path is not None:
        LOGGER.info(f"  报告 (HTML)         : {report_path}")
    LOGGER.info("")

    if args.open and report_path is not None:
        import webbrowser

        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())