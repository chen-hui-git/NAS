# NAS — 超网（Supernet）+ 三明治采样 教学 Demo

一个最小、可运行、可在笔记本上 5 分钟跑完的**权重共享 NAS（one-shot supernet）**实验：

- 一个 Conv1d Transformer 超网，弹性维度为 **深度 4–10 / 宽度 64·96·128 / head_dim 16·32 / 每层 9 个 FFN 分支 / gelu·silu**；
- 用**三明治规则**训练：每一步固定训练 `max` + `min` + 2 个随机子网，损失求和后只做一次 optimizer step；
- 训练完在**同一份权重**里直接取出任意子网打分（不再训练），得到一整条“成本–质量”曲线；
- 顺手量一个关键陷阱：**超网内的排序 vs 每个子网独立训练的排序**（Kendall τ / Spearman ρ / regret）；
- 自动生成一份**中文教学报告 HTML**（原理、代码、图、本次运行的实测数字）。

数据集用公开的 **Tiny Shakespeare**（char-level，1.1 MB，SHA-256 校验后本地缓存），
没有网络时可以用 `--dataset induction` 跑内置的合成 induction 任务。

---

## 快速开始

```bash
cd NAS

# 完整版（默认 300 秒预算，自动决定训练步数）
PYTHONPATH=. python3 examples/run_demo.py --open

# 一分钟速览（课堂现场演示用）
PYTHONPATH=. python3 examples/run_demo.py --budget 60 --steps 150 --candidates 4

# 完全离线
PYTHONPATH=. python3 examples/run_demo.py --dataset induction
```

需要 Python 3.10+ 和 PyTorch 2.0+（`uv`/`venv` 均可），绘图用 matplotlib，其余只用标准库。
脚本会按 `--budget` 自适应：先热身测速，再把剩余时间分给三明治训练、排序实验和出图，
所以慢机器上也是“大约 5 分钟”，而不是“跑一夜”。

跑完输出：

| 产物 | 说明 |
| --- | --- |
| `report/supernet_demo.html` | 中文教学报告（自包含，图片内嵌 base64，可直接发学生） |
| `report/figures/*.png` | 5 张图：切片示意、三条腿损失、空间覆盖、子网族曲线、排序散点 |
| `report/metrics.json` | 本次运行的全部数字（空间、参数、损失、排序相关性、耗时明细） |
| `report/sandwich_log.jsonl` | 每一步的四条腿损失与对应 arch_id，可逐行复现/画图 |

---

## 文件结构

```
supernet/
  search_space.py   SubnetSpec / SearchSpace：纯数据的空间描述；采样、max/min 角点
  modules.py        ElasticConv1d / ElasticAttention / ElasticFFN：权重切片
  supernet.py       共享权重模型、子网导出(export_subnet)、文本生成
  sandwich.py       三明治训练循环 + 覆盖统计（谁真的拿到了梯度）
  rank.py           超网内打分 vs 独立训练，Kendall τ / Spearman ρ / regret
  tasks.py          任务接口：batch / fixed_batches / batch_fn
  dataset.py        tiny-shakespeare 下载+校验；CharLMTask
  data.py           合成 induction 任务（离线回退）
  viz.py            教学图（matplotlib，headless，英文标签）
  report.py         HTML 报告生成器（代码片段从源码里按标记抽取）
examples/run_demo.py  端到端演示，带墙钟时间预算
tests/test_supernet.py 37 个回归测试
```

代码片段不是复制粘贴的：`report.py` 用 `# [snippet:name]` 标记从真实源文件里抽取，
所以报告里的代码永远不会和实现走样。

---

## 三个核心设计

**1. 一份权重，子网是切片。** 权重按最宽的配置分配，窄子网只用前 `d` 个输入/输出通道：

```python
weight = self.weight[:out_channels, :in_channels]
return F.conv1d(x, weight, self.bias[:out_channels], padding=self.padding)
```

通道可以取前缀，卷积核不行 —— 所以 kernel 用“并列分支 + 选一支”，宽度用“切片”，深度用“截断”。

**2. 三明治规则。** 每步四类子网，损失求和，一次 optimizer step：

```python
legs = [("max", space.max_subnet(), w), ("min", space.min_subnet(), w)]
legs += [(f"random_{i}", space.sample(rng), w) for i in range(n_random)]
for role, spec, weight in legs:
    (masked_next_token_loss(model(tokens, spec), targets, mask) * weight).backward()
optimizer.step()
```

max 子网是质量上限兼“教师”，min 子网防止低端掉队，随机子网覆盖中间地带。

**3. 排序一致性要显式量。** 超网内损失低 ≠ 独立训练后也好（co-adaptation）。
`rank.py` 抽 K 个候选，先用超网打分，再把每个候选**从零独立训练同样步数**打分，
报告 τ / ρ / top-1 一致性 / regret。这是超网最容易被忽略的失效模式。

---

## 教学报告

`report/supernet_demo.html` 是本仓库的主要教学材料，包含：

1. 什么是超网（为什么省训练不省测量）；
2. 弹性搜索空间（表格 + 具体 spec）；
3. 权重共享与切片（示意图 + 真实代码 + “切片是近似”的警告）；
4. 三明治规则（代码 + 三条腿损失曲线 + 覆盖直方图）；
5. 一次训练得到一个子网族（深度/宽度扫描曲线 + 参数表 + 生成文本）；
6. 陷阱：排序不一致（散点图 + τ/ρ/regret + 逐候选对照表）；
7. 预算记账（`evaluator_kind`、`subnets_per_step`、独立训练次数怎么记）；
8. 45 分钟课堂节奏建议、复现命令、参考文献。

图中标签用英文，因为 matplotlib 默认字体没有中文字形。

---

## 局限（重要）

- 这是**教学 Demo**，不是 SOTA 实现：模型很小、只跑 1 个种子、排序实验的候选数少，
  τ 的置信区间很宽，第 6 节的数字只用于演示方法，不作为结论。
- 切片是近似：窄子网继承宽子网的权重值但没有单独训练过，导出上线前需要 fine-tune。
- 本实现用 LayerNorm，因此导出子网时不需要 BN 重估；换成 BatchNorm 必须做 BN re-estimation。
- 超网只省训练，**不省测量**：延迟 / FLOPs / 显存仍然要单独测或查表。
- 任何“方法更好”的结论，最终数字必须来自固定评估器下的独立训练；
  超网内的损失只能用于筛选，不能用于报告。

## 测试

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

覆盖：空间校验、极端子网前向、未用分支不会泄漏、浅子网忽略深层、导出子网与共享前向逐位一致、
参数计数与导出模型一致、三明治四条腿、深度/分支覆盖、训练降低 max 子网损失、排序指标、
任务接口、生成接口、报告代码片段抽取与高亮。

## 参考文献

- Bender et al. *Understanding and Simplifying One-Shot Architecture Search* — arXiv:1802.01548
- Pham et al. *Efficient Neural Architecture Search via Parameter Sharing* — arXiv:1802.03268
- Liu et al. *DARTS: Differentiable Architecture Search* — arXiv:1806.09055
- Guo et al. *Single Path One-Shot NAS with Uniform Sampling* — arXiv:1904.00420
- Yu et al. *Universally Slimmable Networks and Improved Training Efficiency* — arXiv:1903.05134
- Cai et al. *Once-for-All: Train One Network and Specialize it for Efficient Deployment* — arXiv:1908.09791
- Yu et al. *BigNAS* — arXiv:2003.11142
- Wang et al. *AttentiveNAS* — arXiv:2011.09011
