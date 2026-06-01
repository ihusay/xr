# MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

> 论文：[MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)，ByteDance，2026

## 整体思路

传统推荐架构将序列建模（用户行为历史）和 dense 特征交叉分为独立模块，两者**竞争同一计算预算**：序列 Transformer 的计算量随序列长度 $O(L^2)$ 增长，dense 交叉随特征维度 $O(d^2)$ 增长，增长速度不同，分开参数化导致无法协同 scaling。即使用边际收益分配预算，两模块在固定接口处单向传递信息，梯度不互通，联合表征能力存在结构性天花板。

MixFormer 的答案是在**每个 block 内部**让 dense 特征 token 直接 attend 序列，打破接口壁垒，实现序列和 dense 的深度融合与协同 scaling。

```
NS 特征 → embed + 切分为 N 个 head ─┐
                                    ├→ [Query Mixer → Cross Attention → Output Fusion] × L → MLP → 预测
行为序列 → per-action embed ─────────┘
```

在抖音和抖音极速版验证，线上 A/B 相比 STCA→RankMixer 基线：Finish Rate +0.39%，Comments +0.70%（抖音极速版 Comments +1.91%），结果未收敛。

---

## 技术方案

### 3.2 Feature Embedding and Splitting

**序列特征（S-tokens）**：用户行为序列 $S = [s_1, s_2, \ldots, s_T]$，每个 action 包含 item id、行为类型、时间戳等，各自 embed 后形成序列。

**非序列特征（NS-tokens）**：user/item/context 属性特征分别 embed 后直接 concat 成大向量 $e_{ns} \in \mathbb{R}^{D_{ns}}$，再按**等长切片**划分为 $N$ 个 subvector（维度 $d = D_{ns}/N$），每段过独立线性投影 $W_j \in \mathbb{R}^{D \times d}$ 升维：

$$x_j = W_j \cdot e_{ns}[d(j-1):dj], \quad j = 1, \ldots, N$$

> 分头方式是位置等分而非语义分组，依赖后续 Query Mixer 的跨 head 信息混合来弥补初始分组的语义模糊性。

```python
class NonSeqEncoder(nn.Module):
    def __init__(self, feature_vocab_sizes, embed_dim, num_heads, output_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, embed_dim) for vocab in feature_vocab_sizes
        ])
        self.num_heads = num_heads
        feats_per_head = len(feature_vocab_sizes) // num_heads
        self.projections = nn.ModuleList([
            nn.Linear(feats_per_head * embed_dim, output_dim)
            for _ in range(num_heads)
        ])

    def forward(self, feature_ids):  # feature_ids: list of (B,)
        # 各特征 embed 后 concat: (B, M*embed_dim)
        e_ns = torch.cat([emb(x) for emb, x in zip(self.embeddings, feature_ids)], dim=-1)
        # 等长切分为 N 份，每份独立投影到 output_dim
        chunks = e_ns.chunk(self.num_heads, dim=-1)
        return torch.stack([proj(h) for proj, h in zip(self.projections, chunks)], dim=1)
        # → (B, N, D)，N 个 NS-token 输入后续 MixFormer Block
```

---

### 3.3 MixFormer Block

每层三个串行模块：

#### Query Mixer（NS 特征内部交叉）

用**无参数 HeadMixing** 替代 self-attention，在 N 个 head 之间做信息混合：将输入 reshape 成 $\mathbb{R}^{N \times N \times D/N}$，转置前两维后 flatten，再接 per-head SwiGLU FFN。

$$q_i = \text{SwiGLUFFN}_i(\text{HeadMixing}(X_{ns}))_i$$

用无参数转置代替 attention 的 $QK^\top$ 计算，消除 $O(N^2)$ 的相似度计算开销。消融实验显示，用 self-attention 替换 HeadMixing **无可见性能提升**，但引入显著计算开销。

灵感来自 MLP-Mixer（视觉领域 token mixing 思路在推荐特征交叉上的移植）。

```python
def query_mixer(X, per_head_ffn):
    # X: (B, N, D)

    # --- HeadMixing（无参数）---
    # 把每个 head 的 D 维拆成 N 段，每段 D/N
    X = X.reshape(B, N, N, D // N)   # (B, N, N, D/N)
    # 转置前两个维度：让原来"第 j 段"流向"第 j 号 head"
    X = X.transpose(1, 2)            # (B, N, N, D/N)
    X = X.flatten(start=2)           # (B, N, D)

    # --- Per-head SwiGLU FFN ---
    # 每个 head 有独立参数，不跨 head 共享
    out = []
    for i in range(N):
        out.append(per_head_ffn[i](X[:, i, :]))  # (B, D)
    Q = stack(out, dim=1)            # (B, N, D)

    return Q  # 作为 Cross Attention 的 query
```

HeadMixing 的直觉：reshape 后每个 head $i$ 持有的 $D/N$ 段数据，transpose 之后被分发给所有其他 head——相当于每个 head 在自己的子空间里读取了所有其他 head 的一片信息，实现跨 head 混合，无可学习参数和相似度计算。

#### Cross Attention（序列信息注入）

每个 NS 特征 head 作为独立 query，去 attend 用户行为序列：

$$h_t = \text{SwiGLUFFN}^{(l)}(\text{Norm}(s_t)) + s_t$$
$$z_i = \sum_t \text{softmax}\!\left(\frac{q_i^\top k_t^i}{\sqrt{D}}\right) v_t^i + q_i$$

每个 head 用独立的 $K/V$ 投影矩阵，不同特征从序列中抽取各自关注的行为信号。这是联合处理的核心——**dense 特征直接 condition 在序列上，梯度双向流通**。

#### Output Fusion（深度融合）

每个 head 独立过 SwiGLU FFN，保持 head 间异质性，完成序列信号与 dense 特征的非线性融合：

$$o_i = \text{SwiGLUFFN}_i(\text{Norm}(z_i)) + z_i$$

整体使用 Pre-RMSNorm，消融实验显示优于 Post-LayerNorm。

---

### 3.4 UI-MixFormer（User-Item Decoupling）

将 NS 特征 head 按来源拆分为 user 侧（$N_u$ 个）和 item 侧（$N_g$ 个），HeadMixing 中加入 mask 矩阵 $\mathcal{M}$ 使信息只从 user → item 单向流动。

工程收益：支持 **Request-Level Batching**——同一请求的多个候选 item 共享 user 侧序列计算，序列-候选 cross-attention 计算量大幅下降，**serving 延迟降低 30%+**，FLOPs 从 3,503 降至 2,242（约 $-36\%$）。

---

## 实验结论

### Scaling 分析

- **Dense Scaling**：相比仅序列或仅非序列模型，MixFormer 在各 FLOP 预算下均有更大截距和竞争性斜率
- **Sequence Scaling**：序列长度从 512 扩展到 10000，MixFormer 与 SOTA STCA 持平，验证序列侧 scaling 未被联合架构压制

### 消融

| 去掉的模块 | 影响 |
|---|---|
| HeadMixing | 性能显著下降 |
| HeadMixing → self-attention | 无提升，计算开销增加 |
| Query Mixer per-head FFN | 有损 |
| Output Fusion per-head FFN | 有损 |
| Pre-RMSNorm → Post-LN | 有损 |

### 线上 A/B（抖音，2 周）

| 指标 | 提升 |
|---|---|
| Active Days | +0.0415% |
| Duration | +0.2799% |
| Finish Rate | +0.3897% |
| Comments | +0.7035% |

抖音极速版 Comments +1.91%，结果未收敛。

---

## 附录：与 Zenith 的特征分组设计对比

### 为什么不按语义分组

Zenith 语义分组的目的是让每个 Prime Token 语义纯净，因为后续 RSA/TMHSA 做的是 token 间 self-attention，token 内混入不相关特征会引入 attention 噪声，分组是为了让 token 成为"有意义的查询单元"。

MixFormer 的 NS-token 定位不同——它是 Cross Attention 的 query，去 attend 用户行为序列。Query Mixer 里的 HeadMixing 在进入 Cross Attention 之前已经做了跨 head 信息混合，语义模糊性在此被弥补。语义分组对 MixFormer 没有必要，且 Zenith 有 4552 个特征，人工维护分组本身是工程负担。消融实验显示 HeadMixing 效果充分，按位置等分足够。

### 特征多了 token 数会不会爆炸

不会。MixFormer 的 token 数 N 是固定超参，与特征数 M 无关：

```
特征数 M 增加 → e_ns 变长（D_ns 增大）→ 每个 head 的切片 d = D_ns/N 变大
                                          W_j 输入维度变大，但 N 不变
```

| | Zenith | MixFormer |
|---|---|---|
| token 数 | T，由语义分组决定（如 32） | N，固定超参（如 8~16） |
| 特征增加时 | token 数可能随之增加 | token 数不变，每个 head 切片更宽 |
| Cross Attention 复杂度 | O(T × L) | O(N × L) |

代价是特征越多每个 head 的 `Linear(d→D)` 需压缩更多信息，但 N 始终可控，serving 延迟不会因特征增加而爆炸。
