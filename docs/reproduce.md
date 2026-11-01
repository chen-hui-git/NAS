# 复现指南

本仓库的"复现"分三级，需求不同做法不同。**默认配置（按墙钟预算自适应）能给到第 3 级；
想要第 1 级的逐位一致，必须钉死步数。**

| 级别 | 含义 | 需要什么 |
| --- | --- | --- |
| **1. 权重级** | `supernet.pt` 与参考实现**逐位一致** | 固定 seed + **固定步数** + 同一设备/版本 |
| **2. 数值级** | 精度/τ 等指标落在 ±1pt 内 | 固定 seed（步数可以不钉死） |
| **3. 结论级** | 结论方向一致（切片近似、覆盖不均、排序不可靠） | 什么都不用固定，任何一次运行都成立 |

## 一、实测证据：步数钉死后是逐位可复现的

同一台机器（M5 Pro，MPS）、同 `--steps 400 --seed 0`、跑三次（其中一次加了 `--deterministic`）：

| 运行 | `supernet.pt` 的 sha256（前 16 位） | 参数量 |
| --- | --- | --- |
| A | `882371bf8b5d8994` | 73,819,270 B |
| B | `882371bf8b5d8994` | 同上 |
| C（`--deterministic`） | `882371bf8b5d8994` | 同上 |

逐张量比较 A vs B、A vs C：**447 个张量全部逐位相同，最大绝对差 0**。
也就是说这套模型在 MPS + float32 上没有非确定性算子；`--deterministic` 不会改变任何一位（但换环境时建议带上）。

## 二、唯一的非确定源：步数由墙钟决定

```python
planned = int(train_budget / (step_seconds * cost_factor))   # step_seconds 来自热身实测
```

`--steps` 只是**上界**。机器快慢、后台负载、散热都会改变实测耗时，从而改变步数——
所以默认配置下每次运行的步数不同（本仓库记录过 1058 / 1326 / 1788 / 1826 等），权重自然不同。

**要复现权重，把步数钉死**：

```bash
cd NAS
PYTHONPATH=. python3 examples/run_demo.py \
  --seed 0 \
  --steps 400 --budget 120 --report-reserve 20 \   # 步数写死，预算给足，避免被 deadline 截断
  --candidates 4 --standalone-steps 60 \           # 审计部分同样钉死
  --save-weights \                                 # 落盘 report/supernet.pt
  --out /tmp/reproduce
```

五个条件缺一不可：**同一份代码（`metrics.json` 的 `git_commit`）、同 seed、同步数、同设备/后端、同 torch/python 版本**。
后两项已经记在 `metrics.json` 的 `torch_version` / `python_version` / `platform` 里。

## 三、超网权重怎么拿、怎么校验

权重用 **Git LFS** 跟踪（`.gitattributes` 里 `report/*.pt filter=lfs`）：

```bash
git lfs install            # 第一次用，需系统里有 git-lfs
git lfs pull               # 拉取真正的 70 MB 对象（否则只有 133 B 的指针）
```

**校验**（指针里就带着 sha256）：

```bash
shasum -a 256 report/supernet.pt
# 期望：d5c50e45f3a835cbeec9afb13bcfc73fa908c619bb22f7391a75e70434093776
```

这份权重对应本仓库提交的那次运行：**batch 128、KD λ=0.5、1058 步、195 秒**。
对应的全部指标在 `report/metrics.json`。

> 注意：如果机器上**没有** git-lfs，`git add *.pt` 会把它当普通二进制存进 git（历史里就会多 70 MB）。
> 提交前用 `git cat-file -s $(git rev-parse :report/supernet.pt)` 检查：**应该只有约 133 字节**（指针），
> 如果是 7000 万级别，说明 LFS 没生效。
> GitHub 的 LFS 免费额度是 1 GB 存储 / 1 GB 月流量，70 MB 的单文件没问题，但别往里塞多个。

## 四、一次运行会留下哪些日志

| 文件 | 内容 | 是否入库 |
| --- | --- | --- |
| `<out>/run.log` | **带时间戳的完整控制台日志**（每个阶段、进度条、逐候选审计） | 否（gitignore） |
| `<out>/sandwich_log.jsonl` | **逐步结构化记录**：每步 4 条腿的 loss / acc / arch_id / stage | 否（gitignore） |
| `<out>/metrics.json` | 汇总指标 + 环境（`git_commit` / `torch` / `python` / `platform` / `seed`）+ `timings` | 是 |
| `<out>/figures/*.png` | 报告用图 | 否（已内嵌进 HTML） |
| `<out>/supernet.pt` | 超网权重 | **是（Git LFS）** |

`run.log` 与 `sandwich_log.jsonl` 都在 `--out` 下，默认路径是 `report/`。
想要别的位置：`--out /tmp/run1`；想换日志文件名：`--log-file my.log`；想只留逐步 jsonl：`--log-file ""`。

控制台只打消息本体（保持分栏对齐），文件里每行额外带 `时间戳 + 级别`，方便事后按时间定位慢在哪一段
（与 `metrics.json` 的 `timings` 对照）。

## 五、复现后怎么自查

```bash
../.venv/bin/python - <<'PY'
import json
a = json.load(open('report/metrics.json'))      # 仓库里提交的那次
b = json.load(open('/tmp/reproduce/metrics.json'))  # 你刚跑的
for key in ('steps_done', 'batch_size', 'distill_weight', 'shrink_stages', 'seed'):
    print(f"{key:16s} {a['run'][key]!r:>12} -> {b['run'][key]!r}")
print("max acc:", a['evals'][0]['accuracy'], "->", b['evals'][0]['accuracy'])
print("tau    :", a['ranking']['kendall_tau'], "->", b['ranking']['kendall_tau'])
PY
```

- **权重级**：`shasum` 与上面的 sha256 一致；
- **数值级**：`steps_done` 与 `evals[0].accuracy` 对齐到 ±1pt 即可；
- **结论级**：只要 `steps_done` 在千步量级，报告第 5/6 节的结论方向就会复现。

## 六、已知的不确定源与边界

| 项 | 影响 | 说明 |
| --- | --- | --- |
| 墙钟→步数映射 | 权重完全不同 | 唯一的"大"不确定源，靠 `--steps` 钉死 |
| 设备/后端（mps vs cpu） | 可能有末位差异 | 本次只在 MPS 上验证过逐位一致 |
| torch / python 版本 | 可能有数值差异 | 已记录在 `metrics.json` |
| MNIST 数据源 | 无影响（有 sha256 校验） | 下载即校验，坏文件不会进入训练 |
| 排序审计的候选 | 结论的样本量 | `--candidates` 少时 τ/recall 的置信区间很宽 |
