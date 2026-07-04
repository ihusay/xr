# Budgeted Active Experimentation for Treatment Effect Estimation from Observational and Randomized Data

> 论文：[Budgeted Active Experimentation for Treatment Effect Estimation from Observational and Randomized Data](https://arxiv.org/abs/2602.22021)，Didi 等，2026

## TL;DR

1. **解决了什么问题**：Observational data 规模大但 treatment 来自历史策略，存在 selection bias 和 overlap deficit；RCT 数据因果信号干净，但实验预算小，不能全量做。
2. **怎么解决的**：用 OBS 训练初始 CATE/uplift 模型，并为候选样本计算 acquisition score：
   - 选择模型多次预测分歧大的样本，优先降低 CATE 估计的不确定性。
   - 选择更像目标候选池、但不像当前训练集的样本，补齐已有 OBS/RCT 没覆盖到的分布区域。
   - 选择历史 propensity 接近 0 或 1 的样本，在历史策略缺少 treatment/control 对照的区域用 RCT 补充反事实信息。

## 背景

在 Uplift/CATE 场景里，常见的数据来源有两类：

- **Observational data**：历史日志，规模大，包含用户特征、历史 treatment、结果反馈。
- **RCT data**：随机实验数据，因果信号更干净，但实验成本高、预算小，难以全量覆盖。

核心矛盾是：OBS 数据便宜但有偏，RCT 数据可信但稀缺。

## Observational Data 的问题

历史日志里的 treatment 分配通常来自线上策略，而不是随机分配。例如营销系统会优先把券、短信、电话等资源给模型认为更可能转化或更值得干预的人群。

这会带来两个问题：

- **Selection bias**：被干预人群和未被干预人群本身就不同，直接比较结果会混入策略选择偏差。
- **Overlap/positivity violation**：某些用户群几乎只收到某一种 treatment，缺少另一种 treatment 下的 counterfactual 样本。

因此，直接用 OBS 训练 CATE/uplift 模型，容易把历史策略偏差学进去，而不是学到真实 treatment effect。

## RCT Data 的价值与限制

RCT 通过随机分配 treatment，可以提供更可靠的因果监督信号：

- treatment assignment 与用户潜在结果独立；
- 能缓解 selection bias；
- 能补齐 OBS 中缺失的 counterfactual 信息。

但 RCT 的问题是成本高，尤其在工业场景里，实验样本会带来预算、用户体验、机会成本和业务风险。因此不能简单地扩大 RCT 到全量用户。

## 整体思路

论文的核心思路是：

> 不直接相信 OBS 的因果标签，而是用 OBS 判断“哪些地方最缺因果信息”，再把有限 RCT 预算花在这些样本上。

它把实验设计看成 active learning：

1. 先用大规模 OBS 训练初始 CATE/uplift 模型。
2. 从目标候选池中为每个样本打分。
3. 选择最值得实验的一批样本做 RCT。
4. 将新 RCT 数据加入训练集，更新模型。
5. 重复直到实验预算耗尽。

关键不是随机抽样做 RCT，而是主动挑选最有信息增益的样本。

## Acquisition Score

对候选样本 `u`，论文使用三个信号打分：

### 1. Epistemic uncertainty

模型对该样本的 uplift/CATE 预测越不确定，越值得实验。

可用 MC Dropout 或 ensemble 多次预测：

```text
v_u = Var({tau_j(phi(u))})
```

### 2. Domain discrepancy

训练一个 domain classifier，区分样本来自：

- 目标候选池 `D_pool`
- 当前已有训练数据 `D_obs ∪ D_rct`

如果某个样本更像目标池、但不像已有训练数据，说明当前数据覆盖不足：

```text
d_u = P(domain = pool | phi(u))
```

### 3. Overlap deficit

用 OBS 训练 propensity model：

```text
e_obs(u) = P_obs(T = 1 | phi(u))
```

如果 `e_obs(u)` 接近 0 或 1，说明历史策略几乎只给一种 treatment，缺 counterfactual：

```text
o_u = 2 * |e_obs(u) - 0.5|
```

### 总分

三个分数先做 rank normalization，再加权：

```text
S(u) = alpha * rank(v_u) + beta * rank(d_u) + gamma * rank(o_u)
```

选 `S(u)` 最高的一批样本进入 RCT。

## 与 Pseudo-Sample Matching 的区别

| 方法 | OBS 的作用 | RCT 的作用 |
|---|---|---|
| Pseudo-Sample Matching | 从 OBS 里找相似样本，扩充已有 RCT | 修补已有小规模、有偏 RCT |
| Budgeted Active Experimentation | 用 OBS 判断哪里缺信息 | 主动决定下一批 RCT 做谁 |

前者更像 **用 OBS 扩增 RCT**，后者更像 **用 OBS 指导 RCT 采样**。

## 业务启发

在营销/uplift 场景中，有限实验预算不应平均撒，也不应只投给模型最不确定的 outlier。更合理的实验样本应该同时满足：

- 模型当前不确定；
- 当前训练数据覆盖不足；
- 历史策略下 treatment/control 严重不平衡。

也就是优先实验那些“最可能修复模型盲区和反事实缺口”的用户群。
