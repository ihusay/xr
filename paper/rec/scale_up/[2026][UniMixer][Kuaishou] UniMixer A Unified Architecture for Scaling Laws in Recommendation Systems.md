# UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems

> 论文：[UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems](https://arxiv.org/abs/2604.00590)，Kuaishou，2026

## TL;DR

1. **解决了什么问题**：Heterogeneous Attention 虽然可学习，但在异构推荐特征上用 $QK^T$ 内积决定交互强度，训练早期容易被输入 token 数值主导，产生不稳定的 attention pattern；TokenMixer 虽然高效稳定，但 mixing 规则固定、不可学习，难以适配不同业务场景。

2. **怎么解决的**：UniMixer 把 TokenMixer 的固定置换矩阵参数化，拆成 block 内 local mixing $W_B^i$ 和 block 间 global mixing $W_G$，让原本规则化的 mixing pattern 变成可学习的 heterogeneous feature interaction。

## 整体思路

推荐系统的 scaling block 目前主要有三类：Heterogeneous Attention、TokenMixer/RankMixer、FM/Wukong。三者都能扩大 dense ranking model 的容量，但建模假设不同：

- Attention-based：通过 token-specific Q/K/V 建模异构特征交互，表达力强但计算重，异构 token 间直接做相似度也可能训练不稳。
- TokenMixer-based：用规则化、无参数 token mixing 替代 attention，相对高效，但 mixing pattern 固定，缺少场景自适应能力。
- FM-based：显式建模特征交叉，效率和解释性好，但低阶交互先验限制 scaling 后的表达力。

UniMixer 的核心是把 TokenMixer 的规则置换操作参数化，得到可学习的 local/global mixing 矩阵，并进一步把 attention、TokenMixer 和 FM 都写进同一个"Local Mixing Pattern + Global Mixing Pattern"框架。

```
原始异构特征
    │
    ▼
Feature Tokenization
    │
    ▼
[UniMixing / UniMixing-Lite + Pertoken SwiGLU] × M
    │
    ▼
Task Towers
```

作者进一步提出 UniMixing-Lite，用低秩 global mixing 和 basis-composed local mixing 降低参数量与 FLOPs，在快手广告留存任务上获得更好的 scaling exponent。

---

## 技术方案

### 4.2 Feature Tokenization

原文表述：

> Based on the semantic categories of the input feature fields, the input features X is first divided into N disjoint feature domains

也就是先按输入 feature field 的语义类别，把特征划成互不重叠的 domain，例如 User Profile、Item Features、Behavior Sequence、Query Features 等。

但这个 domain 划分不是贯穿全模型的长期隔离结构。论文很快对每个 domain 分别做 embedding，然后马上 concat：

$$e_n = \text{Embedding}(X_{\text{domain}}) \in \mathbb{R}^{d_{\text{domain}}}$$

$$E = [e_1, e_2, \ldots, e_N]$$

随后将拼接后的大向量 $E$ 均匀切成若干 block，每个 block 通过 token-specific linear layer 投影成统一维度 token：

$$x_i = W_i^{proj} E_{di:di+d} + b_i^{proj}$$

最终输入 UniMixer 的 hidden states 为：

$$X = [x_1; x_2; \ldots; x_T] \in \mathbb{R}^{T \times D}$$

关键细节：这里的"按语义类别分组"主要服务于 embedding 阶段的组织方式，而不是说后续 block 仍保持 domain-isolated processing。domain embedding 被 concat 后，特征已经进入一个统一长向量。

不过 concat 不等于完全丢失 domain 信息。由于 $E = [e_1, e_2, \ldots, e_N]$ 的排列顺序仍保留 domain 布局，后续等长切 block 时，相邻维度通常仍来自同一个或相近 domain；只有 block 边界附近可能跨 domain。因此 UniMixer 的 tokenization 更准确地说是：

> 先利用 domain 语义组织 embedding，再把有语义顺序的长 embedding 切成规则 token 序列，让后续 UniMixing 在位置化 block 上学习 local/global feature mixing。

这和 Zenith 的"人工语义分组后形成 Prime Token"不同。UniMixer 的 token 不是严格语义纯净 token，而是 concat 后按位置切片得到的 block token；但它又不像完全随机切分，因为 concat 顺序保留了 domain 的弱语义布局。

### 4.3 UniMixing

#### Heterogeneous Feature Interactions

这一段先说明为什么不能直接沿用 Heterogeneous Attention，也不能满足于固定规则的 TokenMixer。

Heterogeneous Attention 用 token-specific $W_Q^i/W_K^i/W_V^i$ 处理推荐特征的异构性，形式上比 vanilla attention 更合理。但它的 global mixing pattern 仍然来自内积相似度：

$$
\text{Attention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt d}\right)V
$$

作者指出，这种由 inner-product similarity 得到的 attention pattern 通常带有 diagonally dominant prior。训练早期 $W_Q^h$ 和 $W_K^h$ 随机初始化时，attention weight 的大小主要被输入 token $X$ 的数值分布主导，而不是被已经学好的特征交互模式主导。

直觉上，第 $i,j$ 个 attention score 为：

$$
\text{score}(i,j) = (x_i W_Q)(x_j W_K)^T
$$

当 Q/K 还没有学到稳定投影时，$x_i$ 与自身或少数数值尺度相近 token 的内积更容易偏大，softmax 后形成 sharp/sparse attention。论文认为这会带来两个风险：

- attention 过度集中在少数 token 或对角线附近，梯度回传变困难，Q/K 的训练可能卡住；
- 在大规模异构特征输入下，内积相似度可能产生不可靠或接近均匀的 interaction pattern，让噪声掩盖关键交叉信号。

这里的关键不是说 attention 一定不能用于推荐，而是说"异构特征之间用 $QK^T$ 相似度决定交互强度"本身不是很稳的归纳偏置。

TokenMixer 则走向另一端：它不用输入相关的 Q/K 相似度，而是用无参数、规则化的 token mixing 做交互，因此计算便宜且避免异构语义空间之间直接比较。但问题是 mixing pattern 固定、不可学习，业务场景适配能力不足；同时原始 TokenMixer 还要求 token 数 $T$ 与 head 数 $H$ 相等，限制了结构选择。

UniMixer 的动机就是折中两者：保留 TokenMixer 的高效结构先验，但把固定规则参数化，使 heterogeneous feature interaction 可以在训练中学习。

TokenMixer 的规则 mixing 可以写成一个置换矩阵作用在 flatten 后的输入上：

$$\text{TokenMixer}(X) = \text{reshape}(W^{perm}\text{flatten}(X))$$

作者观察到这个置换矩阵具有可压缩性，可以分解成类似 Kronecker product 的形式：

$$W^{perm} = G \otimes I$$

UniMixer 将原本固定的 $G$ 和 block 内矩阵参数化，使 mixing pattern 可学习：

$$
\text{UniMixing}(X) =
\text{reshape}
\left(
W_G \cdot \text{reshape}
\left(
[x_1 W_B^1, x_2 W_B^2, \ldots, x_{L/B} W_B^{L/B}], L/B, B
\right),
1, L
\right)
$$

其中：

- $W_B^i$：block 内 local mixing，控制 intra-block feature interaction。
- $W_G$：block 间 global mixing，控制 inter-block interaction。

PyTorch 风格实现如下，重点是沿用论文公式 (13)(14) 的优化计算路径，不显式构造完整的 $W^{perm} \in \mathbb{R}^{L \times L}$：

```python
import torch
import torch.nn as nn


def sinkhorn_knopp(logits, tau=1.0, iters=8, eps=1e-8):
    # logits: (..., n, n)
    w = torch.exp(logits / tau)
    for _ in range(iters):
        w = w / (w.sum(dim=-1, keepdim=True) + eps)
        w = w / (w.sum(dim=-2, keepdim=True) + eps)
    return w


class UniMixing(nn.Module):
    def __init__(
        self,
        input_dim: int,
        block_size: int,
        tau: float = 0.05,
        sinkhorn_iters: int = 8,
        use_constraints: bool = True,
    ):
        super().__init__()
        assert input_dim % block_size == 0
        self.input_dim = input_dim
        self.block_size = block_size
        self.num_blocks = input_dim // block_size
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters
        self.use_constraints = use_constraints

        # W_G: inter-block global mixing, shape (L/B, L/B)
        self.W_G = nn.Parameter(torch.empty(self.num_blocks, self.num_blocks))

        # W_B^i: block-specific local mixing, shape (L/B, B, B)
        self.W_B = nn.Parameter(torch.empty(self.num_blocks, block_size, block_size))

        nn.init.xavier_uniform_(self.W_G)
        nn.init.xavier_uniform_(self.W_B)

    def _constrained_weights(self):
        W_G = self.W_G
        W_B = self.W_B

        if not self.use_constraints:
            return W_G, W_B

        # Symmetry constraint: (W + W^T) / 2
        W_G = 0.5 * (W_G + W_G.transpose(-1, -2))
        W_B = 0.5 * (W_B + W_B.transpose(-1, -2))

        # Doubly stochastic + sparsity controlled by temperature tau.
        W_G = sinkhorn_knopp(W_G, tau=self.tau, iters=self.sinkhorn_iters)
        W_B = sinkhorn_knopp(W_B, tau=self.tau, iters=self.sinkhorn_iters)
        return W_G, W_B

    def forward(self, x):
        """
        x:
          - (batch, input_dim), or
          - (batch, T, D), where T * D == input_dim
        return: same shape as input
        """
        original_shape = x.shape
        x = x.reshape(x.shape[0], self.input_dim)

        W_G, W_B = self._constrained_weights()

        # Split flatten(X) into L/B blocks: (batch, L/B, B)
        x = x.view(x.shape[0], self.num_blocks, self.block_size)

        # Local mixing:
        # [x_1 W_B^1, x_2 W_B^2, ..., x_{L/B} W_B^{L/B}]
        h = torch.einsum("bni,nij->bnj", x, W_B)

        # Global mixing:
        # W_G @ reshape(H, L/B, B)
        y = torch.einsum("nm,bmj->bnj", W_G, h)

        return y.reshape(original_shape)
```

对应论文里的两步：

```text
local:  x_i W_B^i       # block 内交互
global: W_G H           # block 间交互
```

完整矩阵写法是 $(W_G \otimes \{W_B^i\})\text{flatten}(X)$，但代码里只保留两个小矩阵乘法，因此避免了 $L \times L$ 中间矩阵。

直接构造完整 $W^{perm}$ 会产生 $L \times L$ 级别的大矩阵。作者通过上述计算重排，把复杂度从 $O(L^2)$ 降为：

$$O(L^2/B + LB)$$

同时避免训练和推理时产生巨大的中间变量。

### 统一视角

UniMixer 将不同方法统一为：

$$
\text{UniMixing}(X) =
\text{reshape}
\left(
\underbrace{G(X, W_G)}_{\text{Global Mixing Pattern}}
\underbrace{
\begin{bmatrix}
x_1 W_B^1 \\
\vdots \\
x_{L/B} W_B^{L/B}
\end{bmatrix}
}_{\text{Local Mixing Pattern}},
1, L
\right)
$$

在这个框架下：

| 方法 | Local Mixing | Global Mixing |
|---|---|---|
| Self-Attention | $XW_V$ | $\text{softmax}((XW_Q)(XW_K)^T / \sqrt d)$ |
| Heterogeneous Attention | token-specific $X\tilde W_V$ | token-specific Q/K 相似度 |
| TokenMixer | $X$ | 固定规则矩阵 $G$ |
| FM/Wukong | projection matrix $Y$ | $XI(XI)^T$ |
| UniMixer | block-specific $W_B^i$ | learnable $W_G$ |

为了让学习到的 mixing 矩阵保持 TokenMixer 置换矩阵的有用归纳偏置，作者对 $W_G$ 和 $W_B^i$ 加了约束：

- doubly stochastic：行列归一；
- sparsity：通过 temperature coefficient 控制稀疏度；
- symmetry：使用 $(W + W^T)/2$。

实现上使用 Sinkhorn-Knopp iteration 做归一化。

### UniMixing-Lite

标准 UniMixing 中，每个 block 都有独立 local mixing 矩阵，global mixing 矩阵也可能较大。UniMixing-Lite 做两处压缩：

**1. Global mixing 低秩化**

$$W_G \approx A_G B_G$$

**2. Local mixing 用 basis 组合生成**

$$W_B^{*i} = \text{Sinkhorn-Knopp}\left(\sum_{\ell=1}^b \omega_\ell^i Z_\ell\right)$$

也就是用少量 basis matrices $\{Z_\ell\}$ 生成每个 block 的 local mixing，减少 block-specific 参数冗余。

UniMixing-Lite 保留了 TokenMixer 的低成本 global mixing，又引入 attention-like 的可学习 local heterogeneous interaction，是论文中 scaling efficiency 最好的版本。

### SiameseNorm 与训练策略

UniMixer 使用 SiameseNorm 提升深层堆叠稳定性。它维护两条耦合流，试图兼顾 Pre-Norm 的训练稳定性和 Post-Norm 的表达效果。

稀疏 mixing 矩阵对性能很重要，但低温度会导致梯度稀疏和训练困难。论文采用两类策略：

- temperature annealing：从高温逐步降到低温；
- warm-up/cold-start：先用高温训练，再用低温继续训练。

消融中，去掉 temperature coefficient 和 model warm-up 的损失最大。

---

## 实验结论

数据来自快手广告投放场景，任务是用户次日留存预测，样本规模超过 0.7B。指标包括 AUC、UAUC、dense 参数量和 FLOPs。

### 与 SOTA 对比

约 100M 参数级别结果：

| 模型 | AUC | UAUC | Params | FLOPs/Batch |
|---|---:|---:|---:|---:|
| Heterogeneous Attention | 0.744577 | 0.733829 | 132.7M | 1.68T |
| Wukong | 0.744477 | 0.733849 | 107.1M | 1.40T |
| RankMixer | 0.749329 | 0.738938 | 135.5M | 1.68T |
| TokenMixer-Large | 0.748410 | 0.737940 | 103.3M | 1.27T |
| UniMixer 101.5M | 0.750238 | 0.739983 | 101.5M | 2.50T |
| UniMixer-Lite 84.5M | **0.752718** | **0.742530** | 84.5M | 4.24T |

Scaling law 拟合中，UniMixer-Lite 的参数和 FLOPs scaling exponent 都高于 RankMixer 与标准 UniMixer，说明模型容量增加时收益更高。

### 消融

| 变体 | 影响 |
|---|---|
| w/o Temperature Coefficient | AUC 明显下降，说明稀疏 mixing 重要 |
| w/o Symmetry Constraint | 下降 |
| w/o Block-Specific Local Mixing Weight | 下降 |
| w/o Model Warm-Up | 下降明显 |
| SiameseNorm → PostNorm | 小幅下降 |

### UniMixing-Lite

- basis number 增大通常带来收益；
- low-rank rank 增大也带来收益，但参数效率不如增加 basis；
- UniMixer-Lite 加深 block 数仍能提升；
- RankMixer 从 2 blocks 加到 4 blocks 反而退化，说明规则 TokenMixer 不适合简单 depth scaling。

### 线上 A/B

UniMixer / UniMixing-Lite 已部署在快手多个广告投放场景。论文报告 30 天观察窗口内，累计活跃天数 CAD 平均提升超过 15%。

---

## 评价

UniMixer 的贡献在于把推荐 scaling block 从经验模块对比推进到统一结构视角：不同方法都可以被理解为 local mixing 与 global mixing 的不同参数化。它的工程价值也比较明确：从 TokenMixer 的低成本结构出发，引入可学习性和可约束稀疏性，再用 Lite 版本控制参数与计算。

需要注意的是，实验主要基于快手内部广告留存数据，外部可复现性有限；线上收益很大，但论文没有充分拆解收益来自结构、训练策略、业务特征工程还是系统调参。更可靠的结论是：在该工业场景下，UniMixing-Lite 相比 RankMixer 更适合参数、FLOPs 和深度 scaling。
