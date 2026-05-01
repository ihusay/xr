# Wukong: Towards a Scaling Law for Large-Scale Recommendation

> 论文：[Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545)，Meta，2024
> 代码：官方未开源；社区复现：[clabrugere/wukong-recommendation](https://github.com/clabrugere/wukong-recommendation)（PyTorch + TensorFlow，MIT 协议）

## 整体思路

推荐模型长期依赖**稀疏缩放**（扩大 Embedding 表）来提升效果，但这条路有三个根本问题：无法增强特征交互能力、无法充分利用 GPU/TPU 算力、没有可量化的缩放律。

Wukong 的答案是转向**密集缩放**——把模型复杂度预算用在堆叠特征交互层上，而非 Embedding 表的宽度。核心工具是**堆叠 FM（Factorization Machine）**：第 $i$ 层可以捕捉 $1$ 到 $2^i$ 阶交互，层数线性增加，交互阶数指数增长。

```
输入特征（稀疏 + 密集）
    │
    ▼
Embedding Layer              ← 统一映射为 X₀ ∈ ℝⁿˣᵈ
    │
    ▼
Interaction Stack (l 层)     ← FMB（高阶交叉）+ LCB（低阶保留）并行，残差连接
    │
    ▼
MLP → 预测输出
```

在内部 146B 样本的数据集上，Wukong 在跨越两个数量级的模型复杂度范围（~1 → 100+ GFLOP/example）内保持稳定缩放律，是推荐领域首次建立类 LLM 的 scaling law。

---

## 技术方案

### 1. Embedding Layer

#### 输入处理

| 输入类型 | 处理方式 |
|---------|---------|
| 稀疏类别特征（one-hot） | Embedding 表查找，得到 1 个向量 |
| 稀疏类别特征（multi-hot） | Embedding 表查找后 **sum pooling**，聚合成 1 个向量 |
| 密集连续特征 | MLP 映射成 $d$ 维向量 |

**multi-hot 的 sum pooling**：特征有多个取值（如用户历史点击商品列表），对每个值分别查表后逐位求和，压缩为一个固定长度向量。选择 sum 而非 attention/sequence 建模，是刻意把交互复杂度集中在后续 Interaction Stack，保持 Embedding Layer 简洁。

#### 统一维度 d

所有特征最终都变为 **d 维向量**，设计上针对特征重要性差异做了处理：
- **重要特征** → 分配多个 Embedding（在输出矩阵中占多行）
- **次要特征** → 分配较小底层维度，多个小 Embedding 分组拼接后经 MLP 升维到 $d$

#### 输出格式

$$X_0 \in \mathbb{R}^{n \times d}$$

- $n$：所有稀疏 + 密集 Embedding 的总数
- $d$：统一的 Embedding 维度

以**矩阵**而非扁平向量的形式输出（区别于 DCNv2 等模型的 $\mathbb{R}^{nd}$ 表示），每一行是一个独立的 Embedding 向量。这是 FMB 能做 $XX^\top$ 运算的基础——FM 在"特征向量"粒度上做内积，矩阵表示比扁平向量更自然。

#### 设计哲学

Embedding Layer 刻意保持简单，**不在这里做复杂交互**，把所有特征交叉的任务完全交给后续的 Interaction Stack（FMB + LCB 堆栈）来完成，使整体架构更清晰，也更易于缩放。

---

### 2. Interaction Stack

#### 整体结构

堆叠 $l$ 层相同的交互层，每层并行两个块：

$$X_{i+1} = \text{LayerNorm}\bigl([\text{FMB}(X_i) \| \text{LCB}(X_i)] + W_{\text{res}} X_i\bigr)$$

其中 $\|$ 表示在 embedding 数量维度上拼接，$W_{\text{res}} X_i$ 为残差投影（当 $n_F + n_L \neq n_i$ 时将 $X_i$ 投影到匹配形状，否则退化为恒等映射）。

**关键性质**：第 $i$ 层能捕捉 $1$ 到 $2^i$ 阶特征交互，层数线性增加，交互阶数指数增长。

#### Factorization Machine Block（FMB）

FMB 用 FM 捕捉两两特征交叉，再经 MLP 将交叉结果编码为新的 embedding 集合：

$$X \xrightarrow{XX^\top} [n, n] \xrightarrow{\text{flatten}} [n^2] \xrightarrow{\text{LN + MLP}} [n_F \cdot d] \xrightarrow{\text{reshape}} [n_F,\ d]$$

输入 $n$ 个 embedding，输出 $n_F$ 个新 embedding，每层 $n_F$ 固定，MLP 负责把交叉信息压缩重组而非检测交叉。

**低秩优化**：$XX^\top$ 是 $O(n^2 d)$，实际用 $X(X^\top Y)$（$Y \in \mathbb{R}^{n \times k},\ k \ll n$）等价替代，复杂度降至 $O(nkd)$，$Y$ 为固定可学习参数。

#### Linear Compression Block（LCB）

$$\text{LCB}(X_i) = W_l X_i$$

线性变换，在不增加交互阶数的情况下重新组合 Embedding，保留低阶信息，确保交互阶数的单调性。

#### 缩放策略

可调超参数优先级：

1. **$l$**（层数）：最优先，每增加一层交互阶数翻倍
2. **MLP 大小**（FMB 内部）：次之
3. **$k$、$n_x$**（压缩维度和输出嵌入数）：边际收益较小
4. **$n_l$**（LCB 输出嵌入数）：在基础配置中效果有限

---

## 实验结论

- **6 个公开数据集**（Frappe、Criteo Terabyte 等）：所有数据集 AUC 均达 SOTA
- **内部大规模数据集**（146B 样本，720 个特征）：
  - 跨两个数量级保持稳定缩放律：$y = -100 + 99.56 \times x^{0.00071}$
  - 竞品（DLRM、FinalMLP、DCNv2 等）在 30~40 GFLOP 后均饱和或训练不稳定
  - 达到相同质量，DCNv2 需要 **40 倍**计算量
