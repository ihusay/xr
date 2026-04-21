# 特征交叉（Feature Interaction）

## 本质

捕捉特征间的**协同信息**（non-additive effect）——一个特征对目标的影响取决于另一个特征的取值。

$$\hat{y} = f_1(x_1) + f_2(x_2) + \underbrace{f_{12}(x_1, x_2)}_{\text{协同项}} + \cdots$$

例：`年轻用户 + 游戏直播` 的 CTR 远高于各自基线之和，单看任何一个特征都看不出来。

> 特征交叉 ≠ 学联合分布，是提取异构特征间的协同信息。

---

## 推荐系统的特殊性

推荐特征高度异构（user_id、item_id、品类、时段…），不像 NLP/CV 天然同质。MLP 能隐式学习交叉但效率低，需要**显式建模**高阶协同信息。

---

## 演进路线

### FM（Factorization Machines，2010）

经典的二阶显式交叉：每个特征学一个 embedding $v_i$，交叉得分为 $\langle v_i, v_j \rangle x_i x_j$。参数共享使得即便稀疏数据也能泛化，但只能捕捉二阶交互，且所有特征对共用同一套向量空间，表达能力有限。

### DCN-V2（Deep & Cross Network V2，2021）

引入显式多项式交叉层，理论上可逼近任意有界阶数的特征交叉，同时保留 MLP 的隐式交叉能力。相比 FM 更灵活，但交叉层的参数矩阵随特征维度平方增长，大规模 scaling 时计算效率受限。

### Hiformer（2023）

将 Transformer self-attention 引入特征交叉，尝试用注意力机制捕捉任意特征对的协同信息。但 vanilla attention 所有 token 共享同一套 $W_Q/W_K/W_V$，不同语义的异构特征被压入同一表示空间，多层后 token 趋于相似，模型容量无法有效利用。

### Zenith（2025，ByteDance）

以"token 异质性"为核心原则——通过在 attention 投影和 FFN 两个环节均**不跨 token 共享参数**来主动维持异质性，分三阶段处理特征：

**1. Prime Tokenization（特征压缩）**：TikTok Live 有 4,552 个原始特征，其中大量特征语义相近（如多个刻画用户活跃度的统计量），直接送入 attention 既冗余又低效。Zenith 将特征按语义分组，每组通过独立 MLP 聚合为一个高维 Prime Token（共 T=32 个）：相似特征的信息被压缩提炼，信息密度更高，后续 attention 也只需处理 T 个 token 而非 K 个原始特征。ID 类特征（user_id、item_id）语义独立，直接作为单独 token，不经过 MLP 聚合。

**2. Token Fusion（跨 token 交叉）**：在 Prime Token 之间建模 pairwise 协同信息。Vanilla attention 所有 token 共享同一套 $W_Q/W_K/W_V$，多层后 token 趋于相似（相似度 0.5~0.68），容量浪费。Zenith 用 RSA（线性 attention + retokenization reshape），Zenith++ 用 TMHSA——每个 token 拥有独立的投影矩阵，不跨 token 共享，从根本上避免异构特征被压入同一空间。

**3. Token Boost（token 内增强）**：共享 FFN 会把所有 token 的表征"揉"向同一方向（representation collapse），Token Boost 改用每个 token 独立的变换参数，主动维持 token 异质性。Zenith 用 TSwiGLU（tokenwise 门控 FFN），Zenith++ 升级为 TSMoE（稀疏专家混合，不增加推理计算量的前提下扩展容量）。

最终将 token 相似度从 0.5~0.68 压至 0.06~0.47，实现有效 scaling。详见 [[2026][Bytedance] Zenith](<[2026][Bytedance] Zenith Scaling up Ranking Models for Billion-scale Livestreaming Recommendation.md>)。

---

## 相关论文

- **[19] Persia (KDD 2022)**：稀疏 embedding 扩展至 100T 参数
- **[22] Meta 分布式训练 (2021)**：大规模 embedding table 训练效率
- **[1] Understanding Scaling Laws for Rec (Meta 2022)**：推荐 scaling 规律
- **[32] Wukong (2023)**：因子化交叉层 + 推荐 scaling law
- **[7] Hiformer (2023)**：提出 vanilla attention not expressive enough
- **Zenith (2025)**：token 异质性原则，TikTok Live 部署
