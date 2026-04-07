# Zenith: Scaling up Ranking Models for Billion-scale Livestreaming Recommendation

> 论文：[Zenith: Scaling up Ranking Models for Billion-scale Livestreaming Recommendation](https://arxiv.org/abs/2601.21285)，ByteDance/TikTok，2025

## 整体思路

传统排序模型面临两个矛盾：特征数量极多（TikTok Live 有 4,552 个特征）vs. 直接对所有特征做 attention 计算量爆炸。Zenith 的解法是**先压缩特征为少量 Prime Token，再在 token 层面做高质量交叉**。

```
原始特征 (K=4552)
    │
    ▼
Prime Tokenization          ← 压缩：K → T=32 个高维 token
    │
    ▼
Token Fusion (跨 token 交叉) ← RSA + TMHSA
    │
    ▼
Token Boost (token 内增强)   ← TSwiGLU + TSMoE
    │
    ▼
预测输出
```

---

## Prime Tokenization（特征压缩）

**目标**：将 K 个原始特征压缩为 T 个 Prime Token（T ≪ K），每个 token 是一个高维向量（256~1024 维）。

**三条分组原则**：

1. **ID 特征单独建 token**：user_id、item_id 等对几乎所有交互对都关键，各占独立 token
2. **单个特征 embedding 不拆分**：一个特征的 embedding 必须完整放入同一 token，防止语义碎裂
3. **非 ID 特征按语义分组，均衡信息量**：语义相近的特征放同一 token，各 token 含特征数尽量均衡

**具体操作（三步）**：

```python
# Step 1: embedding lookup，每个特征独立
embeds = [emb_table[i](x[:, i]) for i in range(K)]   # K × (B, d)

# Step 2: 按语义分组 concat（人工预定义分组）
group_raw = torch.cat(embeds[group_idx], dim=-1)       # (B, d * n)

# Step 3: 每组独立 MLP，统一投影到 token_dim
prime_token = MLP_i(group_raw)                         # (B, token_dim)

# 所有组的输出拼成 token 序列
prime_tokens = stack([id_token, *group_tokens])        # (B, T, token_dim)
```

ID 类特征跳过 Step 2/3，embedding lookup 后**直接作为 Prime Token**。

---

## Token Fusion（跨 token 交叉）

对标 DCN-V2 的 cross network，建模 token 间的 **pairwise 协同信息**。有两个版本：RSA（Zenith）和 TMHSA（Zenith++）。

### RSA（Retokenized Self-Attention）— Zenith

**动机**：Prime Token 是多个特征压缩而来，同一 token 内部的特征彼此还没有充分交叉，RSA 专门解决这个问题。

**Step 1：Self-Attention**

$$O_1 = X X^\top X W_R, \quad W_R \in \mathbb{R}^{D \times k}, \quad O_1 \in \mathbb{R}^{T \times k}$$

**Step 2：Retokenization（核心技巧）**

$$O_1 \in \mathbb{R}^{T \times k} \xrightarrow{\text{reshape}} O_{TF} \in \mathbb{R}^{\hat{T} \times d}, \quad \text{满足} \; T \cdot k = \hat{T} \cdot d$$

零额外计算的 reshape，token 数从 T 扩展到 $\hat{T}$，维度从 k 变为 d。

**Step 3：残差归一化**

$$X_{out} = \text{Norm}(O_{TF} + \text{MLP}(X))$$

**直觉**：

```
attention 前：[token_1(D维), ..., token_T(D维)]
                    ↓ attention，维度压到 k
attention 后：[token_1(k维), ..., token_T(k维)]
                    ↓ reshape（免费！）
retokenize后：[token_1(d维), ..., token_T̂(d维)]   T̂ > T
```

reshape 后，原来同一 Prime Token 的"片段"被切成新的独立 token，下一层可与其他 token 交叉，实现 **token 内部特征的跨层交叉**。副作用是 token 数变化，需辅助 MLP 补齐以做残差连接。

### TMHSA（Tokenwise Multi-Head Self-Attention）— Zenith++

抛弃 retokenization，改用每个 token **独立的投影矩阵**：

$$Q_h = \text{concat}(\{t_i \, q_{(i,h)}\}_{i=1}^T), \quad K_h, V_h \text{ 同理}$$

$q_{(i,h)}, k_{(i,h)}, v_{(i,h)}$ 是 token × head 专属权重，不跨 token 共享。

| | RSA | TMHSA |
|--|--|--|
| token 内交叉 | reshape 拆 token | token-specific 投影 |
| 额外开销 | reshape 免费，但需辅助 MLP | 参数更多，无需 reshape |
| 设计复杂度 | 较高 | 更简洁 |

---

## Token Boost（token 内增强）

独立处理每个 token，保留并增强其自身语义：

- **TSwiGLU（Tokenwise SwiGLU）**：token 级 FFN，激活函数用 SwiGLU
- **TSMoE（Tokenwise Sparse MoE）**：token 级稀疏混合专家，扩容时不等比增加推理开销；用 z-loss 防止专家负载不均衡

---

## 为什么有效（对比 Vanilla Attention）

| | Vanilla Attention | Zenith (TMHSA + TSMoE) |
|--|--|--|
| 投影矩阵 | 所有 token 共享 | 每个 token 独立 |
| 多层后 token 相似度 | 0.5 ~ 0.68 | **0.06 ~ 0.47** |
| Scaling 效果 | 参数增加但性能不再提升 | 持续有效提升 |

**核心结论**：token 异质性是 scaling 有效的前提，Zenith 通过 tokenwise 参数化主动维持异质性。

---

## 附录：RSA 中的简化 Linear Attention

RSA 的 attention 公式 $O_1 = X X^\top X W_R$ 初看奇怪，本质是标准 attention 的极简化版本。

**按维度拆解**（$X \in \mathbb{R}^{T \times D}$）：

| 子表达式 | 输入维度 | 输出维度 | 含义 |
|----------|----------|----------|------|
| $X X^\top$ | $(T, D) \times (D, T)$ | $(T, T)$ | token 两两内积，即 attention score（无 softmax） |
| $X W_R$ | $(T, D) \times (D, k)$ | $(T, k)$ | token 作为 value，线性压缩到 k 维 |
| $(X X^\top)(X W_R)$ | $(T, T) \times (T, k)$ | $(T, k)$ | 用 score 对 value 加权求和 |

**与标准 Self-Attention 的对比**：

$$\text{标准 Attention:} \quad \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

$$\text{RSA:} \quad X X^\top \cdot X W_R$$

区别：
- Q、K、V **均为同一个** $X$，无独立投影
- 无 softmax，无 $\sqrt{d}$ 缩放 → **线性 attention**
- $W_R$ 仅作用于 value 侧做维度压缩

RSA 的创新点不在 attention 本身，而在其后的 retokenization reshape，attention 部分刻意保持简单以控制计算量。
