# Online Conversion Rate Prediction via Multi-Interval Screening and Synthesizing under Delayed Feedback

**作者**：Qiming Liu, Xiang Ao, Yuyao Guo, Qing He
**机构**：中科院计算所
**会议**：AAAI 2024

---

## TL;DR

DFSN 用"主模型 + 卫星"的主次结构解决延迟反馈，MISS 进一步细化：用多个等待窗口训练多个平等的 head（共享底层），再用一个轻量 synthesizing model 在 assembled pipeline 上学习动态权重聚合各 head 预测，同时用全局正样本加权减少 CVR 低估。比 DFSN 粒度更细、聚合更灵活、可解释性更强。

---

## 问题背景

与 DFSN 相同：延迟反馈导致假负例，等待窗口长短与数据新鲜度之间存在不可调和的矛盾。

现有方法的问题：
- 单窗口方法（FNW、ES-DFM 等）：只能取一个 trade-off 点，Missing-Not-At-Random 偏差难以消除
- 多窗口方法（FTP 等）：聚合框架粗糙，缺乏有效的融合策略

---

## 方法：MISS（Multi-Interval Screening and Synthesizing）

整体分三个模块：

```
Multi-Interval Screening Model  →  Assembled Pipeline Synthesizing Model
（多 head 筛选模型）                    （轻量聚合模型）
         ↕
  Global Positive Weighting
  （全局正样本加权）
```

---

### 1. Multi-Interval Screening Modeling

**结构**：共享底层（embedding + hidden layers）+ N 个独立 output head

```
Feature Input
     ↓
[Shared Layers]
     ↓
 h1    h2    h3  ...  hN
 d1    d2    d3  ...  dN       d1 > d2 > ... > dN > 0
```

- 每个 head hᵢ 独立训练在 D⁺_{τ,dᵢ} 上（带复制机制）
- 梯度只更新对应 head 和共享层，各 head 保持独立"个性"
- 真实正样本反复更新共享层，减少假负例对底层表示的污染

**各 head 的语义**：
- 长窗口 head：标签准，假负例少，但感知不到最新趋势
- 短窗口 head：数据新鲜，能捕捉即时转化激增，但假负例多

这种多样性是后续动态聚合的前提。

损失函数：
```
L_heads = Σᵢ Σ_{(x,y) ∈ D⁺_{τ,dᵢ}} ℓ(y, hᵢ(s(x)))
```

**与 DFSN 的区别**：DFSN 是"一个主模型 + 卫星辅助"的主次结构，MISS 是多个平等 head 共用底层，没有主次之分，粒度更细（实验用5个窗口）。

---

### 2. Assembled Pipeline Aggregation

**聚合方式**：将各 head 预测值拼接，加上归一化版本，送入轻量 synthesizing model：

```
x_pred = [y_h1, y_h2, ..., y_hN]       # 各 head 预测拼接
x_norm = normalize(x_pred)              # 归一化，感知相对大小关系
x      = [x_pred, x_norm]              # 合并输入

→ Light Dense Layers + Softmax → 动态权重 w = [w1, w2, ..., wN]

y_s = Σ wᵢ · y_hᵢ                      # 加权聚合最终预测
```

归一化的作用：让模型感知各 head 预测的**相对大小关系**，例如短窗口 head 预测值偏高时暗示近期即时转化激增。

**关键：Assembled Training Pipeline M_τ**

synthesizing model 的训练数据设计是核心：

| | 旧方案（D_{τ,d_max}）| MISS（M_τ）|
|---|---|---|
| 负样本 | 真实负例 N_τ | 相同 |
| 正样本 | 旧正样本（τ - d_max 附近点击） | **最新正样本**（τ 附近点击、已确认转化）|

正样本替换逻辑：保持相同的延迟时长 d，但把点击时间从 τ-d_max 替换为 τ-d，使正样本更贴近当前分布。用 KL 散度验证，最新正样本分布比旧正样本更接近理想分布。

**与 DFSN Strategy Aggregation 的对比**：

| | MISS | DFSN |
|---|---|---|
| 聚合方式 | softmax 动态权重加权求和 | logit 相加 + max 取下界 |
| 训练数据 | assembled pipeline（新正例 + 真实负例）| 主模型训练数据 D⁺_{τ,d_m} |
| 额外模型 | 轻量 synthesizing model | 无 |

---

### 3. Global Positive Weighting

**动机**：各 head 训练数据中仍有假负例，导致预测值系统性偏低，聚合后仍会低估 CVR。

**方法**：用重要性采样全局放大所有正样本的权重：

```
w(x, y, d) = 1 + α · P(delay > dᵢ | y=1)
           = 1 + α · #{(x,y,d) ∈ Pτ : d > dᵢ} / |Pτ|
```

- α 是预定义超参，代表全局 CVR 水平，控制放大程度
- P(delay > dᵢ | y=1) 用 Pτ 中延迟分布统计近似，不需要额外模型
- 同一 batch 内正样本权重相同（静态设置），避免过度影响 head 的排序能力

**与之前方法的区别**：FNW、ES-DFM 等为每个样本精确计算权重，需要辅助模型；MISS 只做全局校正，依靠 synthesizing model 来提升精度，轻量得多。

---

## 实验

### 数据集

与 DFSN 相同（Criteo 60天/30天归因，Tencent 9天/5天归因）。

MISS 和 FTP 使用相同的等待窗口：
- Criteo：[1D, 7D, 14D, 21D, 30D]
- Tencent：[1H, 6H, 24H, 48H, 120H]

### 主实验结果（RQ1）

MISS 在 Criteo/Tencent 上分别比最强 baseline 提升 RI-AUC **16.8% / 6.1%**（对比 DFSN 的 6.1% / 5.4%，提升更大）。

### 消融实验（RQ2/RQ3）

| 变体 | 说明 | 结论 |
|------|------|------|
| MISS_O | 去掉复制机制 | 性能明显下降，复制机制关键 |
| MISS_L | 去掉 synthesizing model，只用最长窗口 head | 最差，单 head 不够 |
| MISS_A | 去掉 synthesizing model，用平均值 | 优于 MISS_L，但不如动态权重 |
| MISS_R | synthesizing model 训练在 D_{τ,d_max} 上 | 给最长窗口 head 过高权重，忽略其他 head |
| MISS_H | synthesizing model 加入隐层中间结果作为输入 | 与 MISS 相当，说明 head 预测值已足够 |

---

## 关键思路总结

> 各等待窗口对应不同的 bias-freshness 取舍，每个窗口都是一个独特的"观察视角"。MISS 让多个 head 在共享底层上各自学习，再用轻量 synthesizing model 在新鲜的 assembled pipeline 上学习动态聚合权重，同时用全局正样本加权消除系统性低估。

---

## 与 DFSN 的整体对比

| | DFSN | MISS |
|---|---|---|
| 模型结构 | 主模型 + 2个卫星（主次） | N个平等 head（共享底层）|
| 聚合策略 | logit 相加 + max 下界 | softmax 动态权重 |
| 知识迁移 | Embedding Transfer + Strategy Aggregation | 直接用 head 预测值作为聚合输入 |
| 偏差校正 | 无偏卫星重要性采样 | 全局正样本加权（轻量）|
| 数据新鲜度 | 卫星用最新数据，embedding 迁移给主模型 | assembled pipeline 引入最新正样本 |
