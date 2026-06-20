# Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation

> 论文：[Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation](https://arxiv.org/abs/2511.06077)，ByteDance / Douyin，WWW 2026
> 代码：官方未开源；社区复现：[soaprockets/rank-recall/STCA.py](https://github.com/soaprockets/rank-recall/blob/main/STCA.py)（TensorFlow/Keras，非官方）

## 整体思路

工业推荐中的长行为序列模型通常卡在两个矛盾上：

- **效果上**：用户历史越长，能覆盖的兴趣越完整，尤其是低频兴趣和长期偏好
- **工程上**：Transformer self-attention 复杂度为 $O(L^2)$，10K 行为序列在训练、显存、在线延迟上都不可接受

已有两阶段方法（如 TWIN 类 GSU + ESU）先检索再精排，可以降成本，但会打断端到端学习：检索阶段的召回偏差会限制下游模型看到的信息，且难以让排序目标直接反向影响长序列建模。

STCA 的核心判断是：在排序场景里，最重要的信号不是历史行为之间的二阶关系，而是**候选 target 与每条历史行为的相关性**。因此它去掉 history-history self-attention，只保留 target-to-history cross-attention，将复杂度从 $O(L^2)$ 降为 $O(L)$，从而支持端到端建模 10K 长度用户行为序列。

```
target item embedding ─┐
                       ├→ [Target-to-History Cross Attention] × M → history summary
user history sequence ─┘
                                          │
target embedding + multi-layer summaries ─┘ → RankMixer / ranking head → score
```

---

## 技术方案

### 1. STCA：Stacked Target-to-History Cross Attention

#### 1.1 单层结构

每一层 STCA 用候选 target 作为唯一 query，完整用户历史作为 key/value。论文先把 query 和 history 各自过一个**保维的 SwiGLUFFN + LayerNorm** 做输入编码，再做多头 cross-attention。

输入编码（第 $i$ 层）：

$$
\tilde{X}^{(i)} = \text{LN}\big(\text{SwiGLUFFN}^{(i)}(X)\big) \in \mathbb{R}^{L \times d},\qquad
q^{(1)} = \text{LN}\big(\text{SwiGLUFFN}^{(1)}(x_t)\big) \in \mathbb{R}^{d}
$$

多头 target-to-history cross-attention，令 $d_h = d/h$，对每个 head $j$：

$$
\alpha^{(i,j)} = \text{softmax}\left(\frac{q^{(i)} W_Q^{(i,j)} \big(\tilde{X}^{(i)} W_K^{(i,j)}\big)^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{1 \times L},\qquad
o^{(i,j)} = \alpha^{(i,j)} \,\tilde{X}^{(i)} W_V^{(i,j)} \in \mathbb{R}^{1 \times d_h}
$$

$$
o^{(i)} = \big[o^{(i,1)} \| \cdots \| o^{(i,h)}\big] W_O^{(i)} \in \mathbb{R}^{d}
$$

因为 query 只有 1 个，attention score 是 $1 \times L$，每层成本是 $O(L d h)$，随历史长度**线性增长**（对比 self-attention 的 $O(L^2 d h)$）。这里还用了一个计算重排：把 $\alpha(\tilde{X} W_V)$ 改成 $(\alpha \tilde{X}) W_V$——两者因线性可交换而等价，但后者先用 attention 权重把长历史压成一个 $d$ 维向量再做 value 投影，避免为每个历史 token materialize value projection，对 10K 级序列直接降低显存和长度相关 FLOPs。

```python
import torch, torch.nn as nn, torch.nn.functional as F


class SwiGLUFFN(nn.Module):  # 保维 FFN: (xW_u ⊙ swish(xW_v)) W_o
    def __init__(self, d, r=2.0):
        super().__init__()
        h = int(r * d)
        self.w_u, self.w_v, self.w_o = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w_o(self.w_u(x) * F.silu(self.w_v(x)))  # silu = swish


class STCALayer(nn.Module):  # 单层 cross attn: q[B,d] + X[B,L,d] -> o[B,d]
    def __init__(self, d, n_heads, r=2.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q_ffn, self.q_ln = SwiGLUFFN(d, r), nn.LayerNorm(d)  # query 输入编码
        self.k_ffn, self.k_ln = SwiGLUFFN(d, r), nn.LayerNorm(d)  # history 输入编码
        self.w_q, self.w_k, self.w_v, self.w_o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, q, X, mask=None):
        B, L, d = X.shape
        q, Xc = self.q_ln(self.q_ffn(q)), self.k_ln(self.k_ffn(X))    # [B,d], [B,L,d]
        Q = self.w_q(q).view(B, self.h, 1, self.dh)                   # [B,h,1,dh]
        K = self.w_k(Xc).view(B, L, self.h, self.dh).transpose(1, 2)  # [B,h,L,dh]
        attn = (Q @ K.transpose(-1, -2)) / self.dh ** 0.5            # [B,h,1,L]
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :], float("-inf"))  # True=padding
        attn = attn.softmax(-1)
        ctx = (attn @ Xc.unsqueeze(1)).squeeze(2).mean(1)            # (αX) 聚合多头 -> [B,d]
        return self.w_o(self.w_v(ctx))                              # 本层 summary o^(i)
```

> 关于 value 重排：标准多头是 `α(XW_V)`，论文用的是 `(αX)W_V`——两者因线性可交换而等价，但后者在 10K 长序列下不必为每个历史 token materialize value 投影，显存更省。上面按等价形式（先聚合 `αX` 再投影）实现。

#### 1.2 Stacking 与 target-conditioned fusion

STCA 不是只做一层，而是堆叠 $M$ 层，并用 **target-conditioned fusion** 逐层更新 query：第 $i+1$ 层的 query 由原始 target $x_t$ 和前面所有层的输出拼接、压缩、再编码得到：

$$
q^{(i+1)} = \text{LN}\Big(\text{SwiGLUFFN}^{(i+1)}\big([o^{(1)} \| \cdots \| o^{(i)} \| x_t]\, W_C^{(i+1)}\big)\Big),\quad W_C^{(i+1)} \in \mathbb{R}^{(i+1)d \times d}
$$

每层都重新以 target 为中心读取历史，但 query 已经累积了前面各层抽取的兴趣摘要，相当于逐层做 target-conditioned refinement。注意这里**没有加法残差**：原始 target $x_t$ 和各层输出 $o^{(i)}$ 通过**拼接 + 投影**（DenseNet 风格的稠密跳连）一直保留到深层，承担了残差的信息/梯度直通作用。

堆叠完 $M$ 层后，把所有层摘要与 target 再压缩成最终的 target-aware token，送入排序主干：

$$
z = \text{SwiGLUFFN}\big([o^{(1)} \| \cdots \| o^{(M)} \| x_t]\, W_Z\big),\quad W_Z \in \mathbb{R}^{(M+1)d \times d}
$$

```python
class STCA(nn.Module):  # 堆叠 M 层 + target-conditioned fusion: x_t[B,d] + X[B,L,d] -> z[B,d]
    def __init__(self, d, n_heads, n_layers, r=2.0):
        super().__init__()
        self.M = n_layers
        self.layers = nn.ModuleList(STCALayer(d, n_heads, r) for _ in range(n_layers))
        # 第 i+1 层 query 融合：concat((i+1)*d) -> proj -> SwiGLUFFN -> LN
        self.fuse_proj = nn.ModuleList(nn.Linear((i + 1) * d, d, bias=False) for i in range(n_layers - 1))
        self.fuse_ffn = nn.ModuleList(SwiGLUFFN(d, r) for _ in range(n_layers - 1))
        self.fuse_ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers - 1))
        # 预测头：所有层摘要 + target -> z
        self.out_proj, self.out_ffn = nn.Linear((n_layers + 1) * d, d, bias=False), SwiGLUFFN(d, r)

    def forward(self, x_t, X, mask=None):
        q, outs = x_t, []
        for i, layer in enumerate(self.layers):
            outs.append(layer(q, X, mask))                              # o^(i)
            if i < self.M - 1:                                         # -> q^(i+1)
                q = self.fuse_ln[i](self.fuse_ffn[i](self.fuse_proj[i](torch.cat(outs + [x_t], -1))))
        return self.out_ffn(self.out_proj(torch.cat(outs + [x_t], -1)))  # z
```

最终把 $z$ 送入预测头输出分数（见 1.3）。

#### 1.3 预测头与输出

STCA 产出的 $z$ 只是「长序列在 target 条件下的摘要」，**它本身不直接打分**。真正输出 CTR/互动概率前，还要把 $z$ 和另外两路特征**平级拼接**，再过排序主干 RankMixer：

$$
X_{\text{mix}} = \text{concat}\big(z,\ \{u_k\}_{k=1}^{K},\ \{c_\ell\}_{\ell=1}^{C}\big)
$$

$$
h = \text{RankMixer}(X_{\text{mix}};\Theta),\qquad \hat{y} = \text{sigmoid}(w^\top h + b)
$$

其中：

- $z$：STCA 抽出的 target-aware 长序列摘要（1.2 的输出）
- $\{u_k\}$：**user-side 辅助 token**，如 profile / context 特征
- $\{c_\ell\}$：**candidate-side token**，同一 target 的 content / creator 等多模态特征

注意 $u_k$、$c_\ell$ **不参与前面的 cross-attention**，它们到这一步才第一次进入模型，和 $z$ 一起交给 RankMixer 做最终特征交叉。训练用 binary cross-entropy：

$$
\mathcal{L}_{\text{BCE}} = -y\log\hat{y} - (1-y)\log(1-\hat{y})
$$

```python
class STCARanker(nn.Module):  # STCA + 辅助特征 -> 打分
    def __init__(self, d, n_heads, n_layers, n_user, n_cand, ranker):
        super().__init__()
        self.stca = STCA(d, n_heads, n_layers)
        self.ranker = ranker                       # RankMixer，输入 [B, T, d] -> h [B, d]
        self.head = nn.Linear(d, 1)                # w^T h + b
        self.n_user, self.n_cand = n_user, n_cand  # u_k / c_l 的 token 数

    def forward(self, x_t, X, u, c, mask=None):
        # x_t[B,d]  X[B,L,d]  u[B,n_user,d]  c[B,n_cand,d]
        z = self.stca(x_t, X, mask)                          # [B, d]
        x_mix = torch.cat([z[:, None], u, c], dim=1)         # [B, 1+n_user+n_cand, d]
        h = self.ranker(x_mix)                               # [B, d]
        return self.head(h).squeeze(-1)                      # logit；外部接 BCEWithLogitsLoss
```

> 这样 STCA 是个「长序列特征抽取器」，和 RankMixer 解耦：换 RankMixer 或增减 $u_k$/$c_\ell$ 都不影响 STCA 结构。线上推理时配合 RLB，$z$ 的用户侧部分可在同一请求的多个候选间复用。

### 2. RLB：Request-Level Batching

线上排序通常一次请求会给同一用户打分多个候选 item。普通 batching 会把 `(user history, target)` 当作独立样本，导致同一段用户历史被重复传输、重复编码。

RLB 改成以请求为单位组织计算：

```
同一用户历史 H
    ├── target_1
    ├── target_2
    ├── ...
    └── target_m
```

用户侧历史只加载和编码一次，多个 target 共享 history representation，再分别做 target-to-history cross-attention。这个优化不改变模型目标，但显著减少 CPU-GPU 传输、显存占用和重复计算，是 STCA 能在线部署的关键工程部分。

### 3. Train Sparsely, Infer Densely

训练时直接用 10K 历史成本仍然很高，因此论文采用长度外推策略：

- **训练阶段**：随机采样较短历史窗口，降低训练成本
- **推理阶段**：输入完整长历史，如 10K 行为

这个策略成立的前提是 STCA 的 attention 结构对长度是线性的，且 target-to-history 的匹配方式不依赖固定序列长度。论文实验显示，模型在更长推理长度上仍能获得稳定收益。

---
