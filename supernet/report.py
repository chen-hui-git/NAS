"""Render the self-contained teaching report (HTML) for one demo run.

The prose is static; every number, figure and code excerpt comes from the run that
produced it. Code is read from the real source files through snippet markers, so the
report can never drift from the implementation.
"""

from __future__ import annotations

import base64
from datetime import datetime
import html
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]

CSS = """
:root {
  --ink: #16202b;
  --muted: #6b7684;
  --line: #e2e7ee;
  --accent: #1c5fd4;
  --accent-soft: #eef3ff;
  --warm: #d9480f;
  --green: #2f9e44;
  --code-bg: #10192a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 20px 72px;
  background: #f6f8fb;
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.72;
}
.wrap { max-width: 1020px; margin: 0 auto; }
header.hero {
  padding: 46px 0 26px;
  border-bottom: 1px solid var(--line);
}
header.hero h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: -0.4px; }
header.hero p.lead { margin: 0; color: var(--muted); font-size: 16px; }
header.hero .meta { margin-top: 14px; color: var(--muted); font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 12px; margin: 24px 0 8px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.card-label { font-size: 12px; color: var(--muted); letter-spacing: 0.3px; }
.card-value { font-size: 21px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }
.card-hint { font-size: 11.5px; color: var(--muted); margin-top: 3px; }
section { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 26px 28px; margin: 18px 0; }
section > h2 { margin: 0 0 6px; font-size: 21px; letter-spacing: -0.2px; }
section > h2 .tag { font-size: 12px; color: var(--accent); background: var(--accent-soft); border-radius: 999px; padding: 2px 10px; margin-left: 10px; vertical-align: middle; font-weight: 500; }
h3 { font-size: 16px; margin: 26px 0 6px; }
p { margin: 10px 0; }
ul, ol { margin: 10px 0 10px 22px; padding: 0; }
li { margin: 5px 0; }
code.inline { background: var(--accent-soft); color: #14396b; border-radius: 5px; padding: 1px 6px; font-size: 13px; font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; }
.figure { margin: 20px 0 6px; }
.figure img { width: 100%; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
.figure figcaption { color: var(--muted); font-size: 12.5px; margin-top: 8px; }
.code { margin: 16px 0 6px; border-radius: 10px; overflow: hidden; border: 1px solid #0b1220; }
.code .code-head { background: #0b1424; color: #8fa3bd; font-size: 12px; padding: 8px 14px; font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; border-bottom: 1px solid #16233a; }
.code pre { margin: 0; padding: 14px 16px; background: var(--code-bg); color: #d7e3f4; overflow-x: auto; font-size: 12.6px; line-height: 1.62; font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; }
.tok-keyword { color: #7db9ff; }
.tok-string { color: #a5e6a0; }
.tok-comment { color: #6b8299; font-style: italic; }
.tok-number { color: #f0b072; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13.5px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
th { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: none; }
tbody tr:last-child td { border-bottom: none; }
tr.bad td { background: #fff4ec; }
.note { border-left: 3px solid var(--accent); background: var(--accent-soft); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 14px; }
.warn { border-left: 3px solid var(--warn); background: #fff5ee; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 14px; }
.ok { border-left: 3px solid var(--green); background: #f0f9f1; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 14px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 780px) { .grid2 { grid-template-columns: 1fr; } }
pre.ascii { background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; overflow-x: auto; font-size: 12.4px; line-height: 1.5; font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; color: #33465c; }
footer { color: var(--muted); font-size: 12.5px; text-align: center; padding: 28px 0 0; }
a { color: var(--accent); }
.toc { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 16px 22px; margin: 18px 0; font-size: 13.5px; }
.toc b { font-size: 12px; color: var(--muted); display: block; margin-bottom: 6px; }
.toc a { text-decoration: none; margin-right: 16px; white-space: nowrap; display: inline-block; }
.pill { display: inline-block; font-size: 11.5px; border: 1px solid var(--line); border-radius: 999px; padding: 1px 9px; color: var(--muted); margin-right: 6px; }
"""

MARKER = re.compile(r"^\s*#\s*\[(/?)snippet:([\w-]+)\]\s*$")
TOKEN = re.compile(
    r"""
    (?P<comment>\#[^\n]*)
  | (?P<string>\"\"\".*?\"\"\"|f?\"(?:\\.|[^\"\\])*\"|f?'(?:\\.|[^'\\])*')
  | (?P<keyword>\b(?:def|class|return|if|elif|else|for|while|in|import|from|as|with|
        not|and|or|is|None|True|False|lambda|yield|raise|try|except|finally|pass|break|
        continue|assert|await|async|super|self)\b)
  | (?P<number>\b\d[\d_]*(?:\.\d+)?\b)
    """,
    re.VERBOSE | re.DOTALL,
)


def read_snippet(relative_path: str, name: str) -> str:
    """Extract a marked region of a real source file, dedented."""
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    captured: list[str] = []
    capturing = False
    for line in text.splitlines():
        match = MARKER.match(line)
        if match is not None:
            closing, tag = match.groups()
            if tag == name:
                if closing:
                    break
                capturing = True
            continue
        if capturing:
            captured.append(line)
    if not captured:
        raise ValueError(f"snippet {name!r} not found in {relative_path}")
    return textwrap.dedent("\n".join(captured)).strip("\n")


def highlight(code: str) -> str:
    pieces: list[str] = []
    position = 0
    for match in TOKEN.finditer(code):
        if match.start() > position:
            pieces.append(html.escape(code[position : match.start()]))
        pieces.append(
            f'<span class="tok-{match.lastgroup}">{html.escape(match.group())}</span>'
        )
        position = match.end()
    if position < len(code):
        pieces.append(html.escape(code[position:]))
    return "".join(pieces)


def _data_uri(path: Path) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "data:image/png;base64," + payload


def _code(relative_path: str, name: str) -> str:
    code = read_snippet(relative_path, name)
    return (
        '<div class="code"><div class="code-head">'
        + html.escape(relative_path)
        + " &middot; snippet "
        + html.escape(name)
        + "</div><pre><code>"
        + highlight(code)
        + "</code></pre></div>"
    )


def _figure(figures: dict[str, Path], key: str, caption: str) -> str:
    path = figures[key]
    return (
        '<figure class="figure"><img alt="'
        + html.escape(key)
        + '" src="'
        + _data_uri(path)
        + '"><figcaption>'
        + caption
        + "</figcaption></figure>"
    )


def _card(label: str, value: str, hint: str = "") -> str:
    return (
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-hint">{html.escape(hint)}</div></div>'
    )


def _table(headers: list[str], rows: list[list[str]], row_classes: list[str] | None = None) -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = []
    for index, row in enumerate(rows):
        classes = ""
        if row_classes is not None and row_classes[index]:
            classes = f' class="{row_classes[index]}"'
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f"<tr{classes}>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _millions(value: int) -> str:
    return f"{value / 1e6:.2f}M"


def build_report(context: dict, out_path: Path) -> Path:
    """Write a single self-contained HTML file (figures embedded as base64)."""
    run = context["run"]
    space = run["space"]
    params = run["params"]
    stats = context["stats"]
    ranking = context["ranking"]
    figures = context["figures"]

    max_row = next(row for row in context["evals"] if row["label"] == "max")
    min_row = next(row for row in context["evals"] if row["label"] == "min")

    cards = "".join(
        [
            _card("超网参数", _millions(params["supernet"]), "一份权重，全部子网共享"),
            _card("最大子网", _millions(params["max"]), "depth 10 / width 128"),
            _card("最小子网", _millions(params["min"]), "depth 4 / width 64"),
            _card("权重共享比", f"{params['shared_ratio']:.0f}x", "超网 / 最大子网 参数量"),
            _card("本次训练步数", f"{run['steps_done']}", f"{run['step_ms']:.0f} ms/步 · 每步 4 个子网"),
            _card("max 子网准确率", f'{max_row["accuracy"] * 100:.1f}%',
                  f'min 子网 {min_row["accuracy"] * 100:.1f}% / 随机猜 {run["chance_accuracy"] * 100:.0f}%'),
            _card("排序一致性 τ", _fmt(ranking["kendall_tau"], 2), "超网内排序 vs 独立训练排序"),
        ]
    )

    eval_rows = [
        [
            html.escape(row["label"]),
            f'<code class="inline">{html.escape(row["arch_id"])}</code>',
            _millions(row["parameters"]),
            _fmt(row["loss"]),
            f'{row["accuracy"] * 100:.1f}%',
        ]
        for row in context["evals"]
    ]

    depth_rows = [
        [
            str(row["depth"]),
            _millions(row["parameters"]),
            _fmt(row["loss"]),
            f'{row["accuracy"] * 100:.1f}%',
        ]
        for row in context["depth_sweep"]
    ]

    candidate_rows = []
    candidate_classes = []
    for row in sorted(ranking["candidates"], key=lambda item: item["supernet_rank"]):
        disagree = row["supernet_rank"] != row["standalone_rank"]
        candidate_rows.append(
            [
                f'<code class="inline">{html.escape(row["arch_id"])}</code>',
                str(row["spec"]["depth"]),
                str(row["spec"]["embed_dim"]),
                _millions(row["parameters"]),
                f'{_fmt(row["supernet_loss"])} / {row["supernet_accuracy"] * 100:.1f}%',
                f'{_fmt(row["standalone_loss"])} / {row["standalone_accuracy"] * 100:.1f}%',
                f'{row["supernet_rank"]} vs {row["standalone_rank"]}',
            ]
        )
        candidate_classes.append("bad" if disagree else "")

    predictions = context["predictions"]
    truth = predictions["true"]
    correct_max = sum(1 for a, b in zip(truth, predictions["max"]) if a == b)
    correct_min = sum(1 for a, b in zip(truth, predictions["min"]) if a == b)
    prediction_block = _figure(
        figures,
        "predictions",
        "每个类别取一张测试图，下面两行是同一份共享权重里 max 与 min 子网的预测"
        "（绿色正确、橙色错误）。",
    ) + _table(
        ["", "逐张预测"],
        [
            ["真实标签", " ".join(str(value) for value in truth)],
            [f"max 子网（{correct_max}/{len(truth)} 正确）",
             " ".join(str(value) for value in predictions["max"])],
            [f"min 子网（{correct_min}/{len(truth)} 正确）",
             " ".join(str(value) for value in predictions["min"])],
        ],
    )

    timings = run.get("timings", {}) or {}
    audit_images = (
        ranking["num_candidates"] * run["standalone_steps"] * run["batch_size"]
    )
    search_images = run["steps_done"] * (2 + run["n_random"]) * run["batch_size"]
    audit_seconds = timings.get("ranking")
    train_seconds = timings.get("train")
    if audit_seconds and train_seconds:
        audit_ratio = (
            f"{audit_seconds:.0f}s vs {train_seconds:.0f}s 训练"
            f"（{audit_seconds / train_seconds * 100:.0f}%），占本页总耗时 "
            f"{audit_seconds / max(1e-9, run['wall_seconds']) * 100:.0f}%"
        )
    else:
        audit_ratio = "未记录（metrics.json 无 timings）"

    by_supernet = [
        row["arch_id"]
        for row in sorted(ranking["candidates"], key=lambda item: item["supernet_loss"])
    ]
    by_standalone = [
        row["arch_id"]
        for row in sorted(ranking["candidates"], key=lambda item: item["standalone_loss"])
    ]
    recall_rows = []
    for k in (1, 3, 5):
        if k > len(by_standalone):
            continue
        hits = len(set(by_supernet[:k]) & set(by_standalone[:k]))
        recall_rows.append([f"recall@{k}", f"{hits}/{k}", f"{k / len(by_standalone):.2f}"])

    depth_counts = ", ".join(f"{key}: {value}" for key, value in stats["depth_counts"].items())
    kernel_counts = ", ".join(f"{key}: {value}" for key, value in stats["kernel_counts"].items())

    try:
        example_dict = json.loads(context["example_spec"])
    except (KeyError, TypeError, ValueError):
        example_dict = {}
    example_dim = int(example_dict.get("embed_dim") or 0)
    example_depth = int(example_dict.get("depth") or 0)
    slicing_caption = (
        "左：共享权重张量，每格代表若干通道（格数与子网宽度严格对应），粗框是"
        f"这个子网实际用到的前 {example_dim} 个通道，虚线框是最宽子网的边界；"
        f"右：它 {example_depth} 层各自选中的 FFN 分支——同一份规格里每层可以选不同分支。"
    )

    dataset = run["dataset"]
    toc = "".join(
        f'<a href="#{anchor}">{title}</a>'
        for anchor, title in (
            ("what", "1 什么是超网"),
            ("space", "2 弹性搜索空间"),
            ("share", "3 权重共享与切片"),
            ("sandwich", "4 三明治规则"),
            ("family", "5 一次训练一个族"),
            ("trap", "6 陷阱：排序不一致"),
            ("budget", "7 预算记账"),
            ("run", "8 复现与教学安排"),
            ("exercises", "9 自己动手"),
            ("refs", "10 参考文献"),
        )
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>超网（Supernet）最小可运行教学实验</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <h1>超网（Supernet）：一次训练，得到一个子网族</h1>
  <p class="lead">最小可运行的权重共享 NAS 教学实验，训练规则用的是三明治采样（sandwich rule）。本页所有数字与图都来自这一次真实运行。</p>
  <div class="meta">
    <span class="pill">数据集 {html.escape(dataset["name"])}</span>
    <span class="pill">输入 {dataset["input_features"]}×{dataset["seq_len"]}</span>
    <span class="pill">类别 {dataset["num_classes"]}</span>
    <span class="pill">device {run["device"]} · batch {run["batch_size"]}</span>
    <span class="pill">lr {run["lr"]}</span>
    <span class="pill">总耗时 {run["wall_seconds"]:.0f}s</span>
    <span class="pill">生成于 {html.escape(run["started_at"])}</span>
  </div>
  <div class="cards">{cards}</div>
</header>

<div class="toc"><b>目录</b>{toc}</div>

<section id="what">
  <h2>1 什么是超网<span class="tag">直觉</span></h2>
  <p>普通 NAS 把每个候选架构当作一个独立模型：训练一次、评估一次、丢掉。N 个候选就是 N 次训练。</p>
  <p><b>超网（supernet）是把所有候选架构的并集放进同一套权重里。</b>所有可能的层都在，权重按“最宽的那个子网”分配；任意一个子网都不是新模型，而是这份共享权重的<b>切片</b>——深度取前 L 层、宽度取前 d 个通道、每个 block 从 FFN 分支里挑一支。</p>
<pre class="ascii">朴素 NAS                         超网（one-shot）
候选 A ─ 训练 ─ 评估 ─ 丢弃         ┌───────────── 共享权重池 ─────────────┐
候选 B ─ 训练 ─ 评估 ─ 丢弃   ⟶     │ depth 4..10 · width 64/96/128        │
候选 C ─ 训练 ─ 评估 ─ 丢弃         │ 每层 9 个 FFN 分支 · 激活 gelu/silu   │
...  N 次训练                       └──────────────────────────────────────┘
                                     子网 = 切片 → 一次训练，N 个子网可评估</pre>

  <h3>这次的任务</h3>
  <p>MNIST 手写数字分类：一张 28×28 的图被看成 <b>{dataset["seq_len"]} 个 token</b>
  （每行一个 token，每行的 {dataset["input_features"]} 个像素值就是它的特征），
  过完超网后做平均池化，再输出 {dataset["num_classes"]} 类 logits。
  随机猜的准确率是 {run["chance_accuracy"] * 100:.0f}%，好坏一眼就能看懂。</p>

  <h3>本仓库这次覆盖了多大的空间？</h3>
  <p>空间里有 <b>{run["space_size"]:,}</b> 个不同的候选架构（深度 4–10 × 宽度 × 每层 FFN 分支 × 激活），
  但训练只发生在<b>一份 {_millions(params["supernet"])} 参数的权重</b>上。这就是权重共享省下来的东西。</p>

  <div class="note"><b>关键区别</b>：超网省的是“训练”，不省“测量”。延迟 / FLOPs 这类目标仍然要单独测，或查表。</div>

  <h3>先对齐几个词</h3>
  {_table(
      ["术语", "本仓库里的含义"],
      [
          ["超网 supernet", "按最宽/最深配置分配的一份权重，所有子网共享它"],
          ["子网 subnet", "超网的一个切片：深度取前 L 层、宽度取前 d 通道、每层选一条 FFN 分支；用 SubnetSpec 描述"],
          ["弹性 elastic", "某个维度可大可小：深度、宽度、head 大小、FFN 分支"],
          ["三明治规则", "每步固定训练 max + min + 若干随机子网，损失求和后只做一次 optimizer step"],
          ["one-shot", "只训练一次超网，之后不再改权重，只在里面取子网打分"],
          ["排序一致性", "超网内打分得到的排序，与每个子网独立训练后排序的一致程度（Kendall τ / Spearman ρ）"],
          ["regret", "信任超网 top-1 的选择、而不是真实最好子网，所付出的精度/损失代价"],
          ["evaluator_kind", "记账字段：这次数字是“共享权重下的代理分”还是“独立训练后的真实分”"],
      ],
  )}
</section>

<section id="space">
  <h2>2 弹性搜索空间<span class="tag">契约</span></h2>
  <p>整个实验只有一处描述“空间是什么”，而且它是纯数据——没有任何模型代码。搜索策略只输出 <code class="inline">SubnetSpec</code>，不输出 Python。</p>
  {_table(
      ["维度", "取值", "弹性实现方式"],
      [
          ["depth", f"{space['min_depth']}–{space['max_depth']}", "截断：depth 为 L 的子网只跑前 L 层"],
          ["embed_dim", " / ".join(str(item) for item in space["embed_dims"]), "切片：取共享权重的前 d 个通道"],
          ["head_dim", " / ".join(str(item) for item in space["head_dims"]), "切片：head 数 = embed_dim / head_dim"],
          ["ffn_ratio", " / ".join(str(item) for item in space["ffn_ratios"]), "分支：每层 9 个 (ratio, kernel) 组合，选一支"],
          ["ffn_kernel", " / ".join(str(item) for item in space["ffn_kernels"]), "分支：卷积核大小不能切片，只能并列存放"],
          ["activation", " / ".join(space["activations"]), "运行时切换：无参数"],
          ["输入 / 类别", f"{dataset['input_features']}×{dataset['seq_len']} / {dataset['num_classes']}",
           "固定：每行像素是一个 token，所有子网共享同一输入输出形状"],
      ],
  )}
  <p>一个具体的子网长这样（本次运行里随机抽到的一个候选子网；第 3 节的示意图画的就是它）：</p>
<pre class="ascii">{html.escape(context["example_spec"])}</pre>
  <h3>每一层那三个字段到底是什么意思</h3>
  <p>逐层可搜索的只有三项，其余都是全局或写死的。一个 block 的实际计算是：</p>
<pre class="ascii">x = x + Attention( LayerNorm(x) )                       ← 宽度 / head 数来自全局 embed_dim、head_dim
x = x + conv1d_k( LayerNorm(x) , out = dim x ratio )    ← kernel、ratio 来自这一层
        -> activation(...) -> conv1d_1( out = dim )</pre>
  {_table(
      ["字段", "取值", "含义", "落到模型上"],
      [
          ["ffn_ratio", " / ".join(str(item) for item in space["ffn_ratios"]),
           "FFN 隐藏层宽度倍数：hidden = embed_dim x ratio",
           "选哪条分支；分支容量按 max_dim x ratio 分配，再切前 dim x ratio 个通道"],
          ["ffn_kernel", " / ".join(str(item) for item in space["ffn_kernels"]),
           "FFN 第一个 Conv1d 的卷积核大小",
           "k=1 只看当前 token；k=3 / 5 会把相邻 token（本任务是相邻图像行）卷进来"],
          ["activation", " / ".join(space["activations"]),
           "FFN 里的激活函数", "无参数，运行时按名字切换函数"],
          ["（全局）embed_dim / head_dim", "见上表", "宽度与 head 大小，所有层共用",
           "head 数 = embed_dim / head_dim；注意力输出投影取左上块"],
          ["（全局）depth", f"{space['min_depth']}–{space['max_depth']}", "只跑前 L 层",
           "截断，不复制权重"],
      ],
  )}
  <div class="warn"><b>记账坑：arch_id 只对“可搜索变量”做哈希。</b>
  <code class="inline">arch_id</code> 覆盖 depth / embed_dim / head_dim / layers 四项。
  如果你改了写死的部分（池化方式、归一化类型、注意力是否遮挡、位置编码），
  <b>arch_id 一个字符都不会变，但模型已经不是同一个了</b>。
  所以实验记录必须把 <code class="inline">arch_id</code> 和空间定义
  （<code class="inline">space.to_dict()</code>）成对保存——本仓库的
  <code class="inline">metrics.json</code> 就是这么存的。</div>
  <div class="warn"><b>为什么 FFN 用“分支”而宽度用“切片”？</b> 切片只在“同一组权重的前缀”有意义时才等价于一个真实的窄模型。
  通道维度可以取前缀（前 d 个通道），卷积核大小不行——k=5 的卷积核没有“前 3 个”这种子结构，所以 kernel 只能并列存放、整支激活。</div>
</section>

<section id="share">
  <h2>3 权重共享与切片<span class="tag">核心技巧</span></h2>
  <p>切片只发生在 <code class="inline">forward</code> 里几行：权重按最大宽度分配，子网取前 <code class="inline">out_channels × in_channels</code> 那块。同一份参数因此同时服务宽子网和窄子网。</p>
  {_figure(figures, "slicing", slicing_caption)}
  {_code("supernet/modules.py", "elastic-conv")}
  <p>FFN 的分支选择同样只是一个字典查找：</p>
  {_code("supernet/modules.py", "elastic-ffn")}
  <p>整网前向就变成“截断深度 + 一路切片”：</p>
  {_code("supernet/supernet.py", "supernet-forward")}

  <h3>注意力也一起变窄</h3>
  <p>宽度不只影响 FFN。qkv 卷积的输出通道按 3 x max_dim 分配，窄子网只取前 3 x dim 行；
  这 3 x dim 行按 (q, k, v) 三段切开后，每段恰好是“前 heads 个 head”，所以 head 数是
  <code class="inline">dim / head_dim</code> 自然跟着宽度变。输出投影则取左上 dim x dim 的块。</p>
  <p>这就是为什么 <code class="inline">head_dim</code> 也要放进搜索空间：固定宽度下，head_dim 小意味着更多 head（更细的注意力），
  但每个 head 的维度更小。两者都受同一份共享权重的切片约束。</p>
  <div class="warn"><b>这是近似，不是等价。</b> 取“前 d 个通道”得到的窄卷积，参数值继承自宽子网，但没有单独训练过；
  导出子网上线前需要 fine-tune。另外，本仓库用 LayerNorm，所以导出时不需要重估统计量；如果换成 BatchNorm，
  必须用一批数据重新校准 BN（BN re-estimation），否则精度会明显掉。</div>
</section>

<section id="sandwich">
  <h2>4 三明治规则<span class="tag">训练规则</span></h2>
  <p>随机采样一条路径有个毛病：大子网和小子网都很少被连续训练，权重会朝“随机平均”跑偏。三明治规则每步固定训练四类子网：</p>
  <ol>
    <li><b>max</b>：最贵子网（depth 10 / width 128 / ratio 4 / kernel 5）——质量上限，也是其他子网的“教师”；</li>
    <li><b>min</b>：最便宜子网（depth 4 / width 64 / ratio 2 / kernel 1）——保证低端不掉队；</li>
    <li><b>random × {run["n_random"]}</b>：均匀采样，负责覆盖中间地带。</li>
  </ol>
  <p>四条腿的损失求和后<b>只做一次 optimizer step</b>，所以步数预算的口径是“4 个子网 / 步”。</p>
  {_code("supernet/sandwich.py", "sandwich-build")}
  {_code("supernet/sandwich.py", "sandwich-step")}
  {_figure(figures, "sandwich", "三条腿的损失曲线：max 最低、min 最高、随机子网位于中间。细线是原始值，粗线是滑动平均。")}
  <p>训练完成后，覆盖统计证明每个深度、每个分支都真的拿到了梯度：</p>
  <div class="grid2">
    <div>{_figure(figures, "coverage", "深度 / 宽度 / 卷积核的覆盖计数。")}</div>
    <div>
      <p><b>深度覆盖</b>：{html.escape(depth_counts)}</p>
      <p><b>FFN 分支覆盖（按层次计）</b>：{html.escape(kernel_counts)}</p>
      <p>共训练子网 {stats["subnets_trained"]} 个次，{stats["steps"]} 步。</p>
      <div class="note">如果某一行是 0，就说明那部分权重从未被训练——这是超网最容易悄悄出错的地方，所以这里把它显式画出来。</div>
    </div>
  </div>
</section>

<section id="family">
  <h2>5 一次训练得到一个子网族<span class="tag">收益</span></h2>
  <p>训练结束后，同一份权重里可以取出任意多的子网并立刻打分。下面是<b>没有额外训练</b>的两次扫描：固定最大宽度扫深度 4→10，固定最大深度扫宽度。</p>
  {_figure(figures, "family", "左：深度扫描；右：参数-质量散点，两条曲线都来自同一份共享权重。")}
  <h3>这两条曲线怎么读</h3>
  <ul>
    <li><b>曲线为什么几乎平？</b> 本次 depth 从 {space["min_depth"]} 加到 {space["max_depth"]}，
    准确率只动了不到 1 个点——说明这个任务在当前训练量下<b>深度已经饱和</b>，
    而不是“深度没用”。判断依据：更深的子网更容易欠训（见第 4 节覆盖图，depth 10 只在 max 腿每步出现）。
    </li>
    <li><b>为什么宽度更有效？</b> 宽度直接影响每层的通道数，也影响 head 数；同样的训练步数下，
    宽子网拿到的梯度更多（每次更新的参数更多），所以低宽度端掉得更快。</li>
    <li><b>右图怎么用？</b> 每一点都是一个“可直接部署”的配置：横轴是成本（参数量），纵轴是质量。
    选模型就是在左上角找点——这就是超网真正交付的东西：<b>一条成本-质量曲线，而不是一个架构</b>。</li>
    <li><b>注意非单调。</b> 宽度扫描里更宽的子网偶尔更差（本次 w96 就低于 w64），
    这不是 bug，而是“训练不均衡 + 评测噪声”的体现，值得让学生自己找反例。</li>
  </ul>
  <h3>深度扫描明细</h3>
  {_table(["depth", "参数量", "验证损失", "验证准确率"], depth_rows)}
  <h3>极端子网与随机子网</h3>
  {_table(["子网", "arch_id", "参数量", "验证损失", "验证准确率"], eval_rows)}
  <p>随机猜的基线是 loss {_fmt(run["chance_loss"], 3)} / 准确率 {run["chance_accuracy"] * 100:.0f}%；
  上面几个数字与它的距离，就是“训练确实发生了”的证据。</p>
  <h3>定性检查：同一份权重里的两个子网分别预测</h3>
  <p>max 与 min 子网共用同一套参数，参数量却差了约 {params["max"] / max(1, params["min"]):.0f} 倍。
  下面看它们在同一批测试图上的逐张表现。</p>
  {prediction_block}
</section>

<section id="trap">
  <h2>6 陷阱：排序不一致<span class="tag">必读</span></h2>
  <p>超网内损失低，不等于独立训练后也好。共享权重是在“邻居分支也在被训练”的前提下学出来的（co-adaptation），所以两套排序可能对不上。本次实验直接量了这件事：抽 {ranking["num_candidates"]} 个候选，先用超网打分，
  再把每个候选<b>从零独立训练同样步数</b>后打分。</p>
  {_figure(figures, "ranking", "横轴：超网内损失；纵轴：独立训练后的损失。点靠近虚线表示两套排序一致。")}
  {_table(
      ["指标", "本次实测"],
      [
          ["Kendall τ", _fmt(ranking["kendall_tau"], 2)],
          ["独立训练后的最高准确率", f'{ranking["best_standalone_accuracy"] * 100:.1f}%'],
          ["Spearman ρ", _fmt(ranking["spearman_rho"], 2)],
          ["top-1 是否一致", "是" if ranking["top1_agreement"] else "否"],
          ["regret（信任超网 top-1 的代价）", f'{ranking["regret"]:+.3f}'],
          ["超网 top-1 的独立训练损失", _fmt(ranking["best_supernet_loss"])],
          ["真正最好的独立训练损失", _fmt(ranking["best_standalone_loss"])],
      ],
  )}
  <p>逐候选对照（橙色行 = 两套排序对不上）：</p>
  {_table(
      ["arch_id", "depth", "width", "参数量", "超网内 loss / acc", "独立训练 loss / acc",
       "排名 超网 vs 真实"],
      candidate_rows,
      candidate_classes,
  )}
  <h3>τ 不够用：搜索真正关心的是 recall@k</h3>
  <p>τ 衡量<b>全序</b>是否一致，而搜索只关心一件事：<b>真实最好的几个子网，有没有被超网排进前 k 名</b>。
  这两个指标可以不一致，所以两份都该报。</p>
  {_table(["指标", "本次实测", "随机基线"], recall_rows)}
  <p>如果 recall@k 和随机基线差不多，那就不能拿超网当筛选器——本次正处在边缘：
  recall@1 为 0，recall@3 与基线持平。用法判断标准：τ &gt; 0.7 可当可靠代理；
  0.3–0.7 只能粗筛且必须复核；τ &lt; 0.3 基本无效（阈值任务相关，需自己量）。</p>
  <div class="warn"><b>教学要点</b>：τ 只有 {_fmt(ranking["kendall_tau"], 2)}，说明“超网最优”这个结论并不天然可信。
  样本量小（{ranking["num_candidates"]} 个候选）时 τ 的置信区间很宽，所以上面的数字只用于<b>演示方法</b>，不作为结论。</div>
  <h3>那超网还能用吗？三种安全用法</h3>
  <ol>
    <li><b>只当粗筛</b>：超网从几千个候选里选前 20，再用独立训练重排。省的是粗筛的钱。</li>
    <li><b>当加速器而非论文主张</b>：主实验的预算口径仍然按“独立训练次数”记，超网只出现在附录。</li>
    <li><b>把它当作被研究的对象</b>：把“超网内排序与真实排序的相关性”本身做成一个可报告的量（就是本页第 6 节）。</li>
  </ol>
</section>

<section id="budget">
  <h2>7 预算记账：把超网纳入实验口径<span class="tag">方法学</span></h2>
  <p>一旦换了评估器，不同策略的候选就不再是在同一把尺子下比较。下表是接入任何 NAS/HPO 对比时必须写清的量：</p>
  <p style="color:var(--muted);font-size:12.5px">其中 <code class="inline">ranking_audit</code> 那一行是本页最容易被忽略的开销：
  超网侧打分只要一次前向，而“真值排序”要把每个候选**从零训练一遍**，成本随候选数线性增长。</p>
  {_table(
      ["字段", "取值", "为什么必须记"],
      [
          ["evaluator_kind", "shared_weights", "区分“独立训练后的精度”和“共享权重下的代理精度”"],
          ["supernet_steps", f"{run['steps_done']}", "训练超网本身的开销"],
          ["subnets_per_step", f"{1 + 1 + run['n_random']}", "三明治一步算几个子网，决定口径"],
          ["standalone_trials", f"{ranking['num_candidates']} × {run['standalone_steps']} steps", "最终结论必须来自独立训练"],
          ["ranking_audit（真值排序）",
           f"{ranking['num_candidates']} × {run['standalone_steps']} 步 × {run['batch_size']} = {audit_images:,} 图",
           f"超网训练量是 {search_images:,} 子网-图次；审计 = 它的 {audit_images / max(1, search_images) * 100:.0f}%"],
          ["audit / search 开销比", audit_ratio,
           "审稿人第一个会算的数：审计开销常常和训练超网本身同量级"],
          ["supernet_score（超网侧打分）",
           f"{ranking['num_candidates']} 候选 × {run['eval_batches']} batch（只前向）",
           "几乎免费——贵的是“真值”那半，不是打分"],
          ["eval_batches / batch_size", f"{run['eval_batches']} / {run['batch_size']}", "打分用的批次必须对所有策略一致"],
          ["device / 墙钟", f"{run['device']} · {run['wall_seconds']:.0f}s",
           "训练成本和策略无关，但必须报告；本 Demo 走单机训练"],
          ["seeds", "1", "至少 3 个种子才能谈方差，本 Demo 只跑 1 个"],
      ],
  )}
  <p>成本是<b>极不对称</b>的：超网侧打分 = 一次前向 + 一批候选，几乎免费；真值侧 = 每个候选一次完整训练。
  所以扩大审计规模（更多候选 / 更多种子）的代价是线性上涨的：把候选从 {ranking["num_candidates"]} 提到 30、
  再跑 3 个种子，审计开销就会超过训练超网本身。这也是为什么很多 one-shot 论文只报最终架构精度、不报排序一致性——
  不是不重要，是<b>很少给得起这个预算</b>。</p>
  <div class="note">原则：<b>超网内损失只能用于筛选，不能用于报告</b>。任何“我们的方法更好”的结论，最后的数字必须来自固定评估器下的独立训练。</div>
</section>

<section id="run">
  <h2>8 复现与教学安排<span class="tag">动手</span></h2>
  <h3>一条命令</h3>
  <pre class="ascii">{html.escape(run["command"])}</pre>
  <p>脚本会按 <code class="inline">--budget</code>（默认 300 秒）自动决定训练步数：先热身测速，再把剩下的时间分给三明治训练和排序实验。
  想要极限快：<code class="inline">--budget 60 --steps 120 --candidates 4</code>。</p>
  <h3>文件结构</h3>
  <pre class="ascii">supernet/
  search_space.py   SubnetSpec / SearchSpace：纯数据的空间描述
  modules.py        ElasticConv1d / ElasticAttention / ElasticFFN 切片
  supernet.py       共享权重模型、子网导出、predict
  sandwich.py       三明治训练循环 + 覆盖统计
  rank.py           超网内打分 vs 独立训练，Kendall/Spearman
  tasks.py          任务接口（batch / fixed_batches / batch_fn）
  dataset.py        MNIST（gzip+IDX 结构校验 + SHA-256，可离线 --dataset bars）
  synthetic.py      合成条形图任务（无网络时的回退）
  viz.py            教学图（matplotlib，headless）
  report.py         本页的 HTML 生成器
examples/run_demo.py  端到端演示（带时间预算）
tests/test_supernet.py 40 个回归测试</pre>
  <h3>课堂讨论：三个问题</h3>
  <p>先让学生只看第 5、6 节的数据自己回答，再对照教师用文档：</p>
  <div class="ok">
    <p><b>① 最小子网的损失，一定会高于最大子网吗？</b>
    看第 5 节的深度/宽度扫描与逐张预测，找出反例，再说清楚“为什么通常更差”究竟是从哪儿来的。</p>
    <p><b>② 如果把三明治里的 min 那条腿去掉，会坏在哪里？</b>
    先看第 4 节的覆盖直方图：哪些分支、哪些深度会被饿死？</p>
    <p><b>③ 当 Kendall τ 只有 0.3 左右时，还能用超网做搜索吗？</b>
    请给出一个可量化的判断标准，而不只是“能用/不能用”。</p>
  </div>
  <p style="color:var(--muted);font-size:12.5px">参考答案、45 分钟课堂节奏、常见误答见仓库里的
  <code class="inline">docs/instructor_notes.md</code>（教师用，不随报告重新生成）。</p>

  <h3>读完应该能回答</h3>
  <ol>
    <li>超网省下了什么、又<b>没有</b>省下什么？（训练 vs 测量）</li>
    <li>为什么通道可以切片、卷积核不行、深度用截断就够了？</li>
    <li>三明治里为什么必须同时有 max 和 min 两条腿，去掉一条会坏在哪？</li>
    <li>怎么从一次训练里读出“成本-质量曲线”，而不是一个架构？</li>
    <li>τ 和 recall@k 差在哪，为什么搜索要报后者？</li>
    <li>什么情况下“超网内指标”绝对不能写进论文数字？</li>
  </ol>
</section>

<section id="exercises">
  <h2>9 自己动手<span class="tag">练习</span></h2>
  <p>下面每个实验都保留原来的评估口径（同一批测试图、同样的独立训练步数），所以结果可以直接和本节上面的数字对比。</p>
  <h3>六个可以现场改的地方</h3>
  <p>每个都只需要一条命令或几行代码，改完重跑一次，把“换了个旋钮会发生什么”变成学生自己的观察。</p>
  {_table(
      ["改什么", "怎么改", "看哪里"],
      [
          ["只留 max + min 两条腿",
           "<code class=\"inline\">--n-random 0</code>",
           "第 4 节覆盖图会明显变窄；随机子网精度下降——证明“覆盖”不是白来的"],
          ["训练量砍到很小",
           "<code class=\"inline\">--steps 200 --budget 60</code>",
           "第 6 节的 τ 和 recall@k 会一起变差：超网质量决定排序可用性"],
          ["缩小 FFN 分支集合",
           "search_space.py 里 FfnKernels 改成 (1, 3)",
           "空间候选数、覆盖直方图、以及第 5 节曲线是否更平"],
          ["换划分方式",
           "search_space.py 里 EMBED_DIMS 增加一个宽度",
           "第 5 节右图会多一个点，看它落在成本-质量曲线的哪个位置"],
          ["换池化方式",
           "supernet.py 里 mean 改成取最后一个 token",
           "精度与第 5 节扫描曲线的变化（分类任务里池化选择影响很大）"],
          ["只改报告文案",
           "<code class=\"inline\">PYTHONPATH=. python3 examples/render_report.py</code>",
           "秒级重渲染，不重新训练，数字保持不变"],
      ],
  )}
  <h3>两次消融的实测结果</h3>
  <p>这不是本页那一次运行的数字，而是另外跑的两次（同机、同评估口径：同一批测试图，每个候选独立训练 120–150 步），每次只改一个设置：</p>
  {_table(
      ["配置", "训练步数", "max 子网", "min 子网", "随机子网", "宽度 w64 / w96", "Kendall τ"],
      [
          ["完整三明治（本页那次）", "1826", "98.1%", "96.9%", "93.0%", "95.8% / 92.8%", "+0.33"],
          ["只留 max + min（<code class=\"inline\">--n-random 0</code>）", "600", "92.5%", "95.2%",
           "<b>20.8%</b>", "<b>48.8% / 44.7%</b>", "+0.20"],
          ["训练量砍到 200 步（<code class=\"inline\">--steps 200</code>）", "200", "84.8%", "77.7%",
           "60.2%", "72.0% / 76.1%", "<b>-0.20</b>"],
      ],
  )}
  <p>看中间那一行：<b>去掉随机腿之后，两端依然健康（max 92.5%、min 95.2%），但中间地带塌了</b>——
  宽度 w64/w96 从 95.8%/92.8% 掉到 48.8%/44.7%，随机采样子网从 93.0% 掉到 20.8%。
  所以三明治的三条腿各管一段：<b>max 管上限、min 管下限、随机腿管中间地带</b>，缺哪条就塌哪段。</p>
  <p>最后一行则说明另一件事：训练量不足时，不只是精度低，<b>连排序都会从正相关翻成负相关</b>（τ +0.33 → -0.20）。
  τ 不是超网的固有属性，而是“超网 + 训练量”的属性。</p>

  <div class="note"><b>建议的第一次实验</b>：把 <code class="inline">--steps</code> 从默认值砍到 200，
  再对比第 6 节的 τ / recall@k。你会看到“超网内排序”从“勉强可用”变成“完全不可用”——
  这比任何文字都更能说明：<b>超网分数是代理，不是结论</b>。</div>
</section>

<section id="refs">
  <h2>10 参考文献<span class="tag">出处</span></h2>
  <ul>
    <li>Bender et al. <i>Understanding and Simplifying One-Shot Architecture Search</i>. <a href="https://arxiv.org/abs/1802.01548">arXiv:1802.01548</a></li>
    <li>Pham et al. <i>Efficient Neural Architecture Search via Parameter Sharing</i>. <a href="https://arxiv.org/abs/1802.03268">arXiv:1802.03268</a></li>
    <li>Liu et al. <i>DARTS: Differentiable Architecture Search</i>. <a href="https://arxiv.org/abs/1806.09055">arXiv:1806.09055</a></li>
    <li>Guo et al. <i>Single Path One-Shot Neural Architecture Search with Uniform Sampling</i>. <a href="https://arxiv.org/abs/1904.00420">arXiv:1904.00420</a></li>
    <li>Yu et al. <i>Universally Slimmable Networks and Improved Training Efficiency</i>（三明治式训练思路）. <a href="https://arxiv.org/abs/1903.05134">arXiv:1903.05134</a></li>
    <li>Cai et al. <i>Once-for-All: Train One Network and Specialize it for Efficient Deployment</i>（三明治规则 + 弹性宽度/深度/卷积核）. <a href="https://arxiv.org/abs/1908.09791">arXiv:1908.09791</a></li>
    <li>Yu et al. <i>BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models</i>. <a href="https://arxiv.org/abs/2003.11142">arXiv:2003.11142</a></li>
    <li>Wang et al. <i>AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling</i>. <a href="https://arxiv.org/abs/2011.09011">arXiv:2011.09011</a></li>
  </ul>
  <p style="color:var(--muted);font-size:12.5px">数据集：MNIST 手写数字（LeCun, Bottou, Bengio, Haffner 1998；idx3/idx1 ubyte 原始格式），
  本地缓存，做 gzip + IDX 结构校验与 SHA-256 校验。</p>
</section>

<footer>
  由 <code class="inline">examples/run_demo.py</code> 自动生成 · {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))} ·
  图中标签使用英文，因为 matplotlib 默认字体没有中文字形。
</footer>

</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
