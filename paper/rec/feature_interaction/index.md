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

### Wukong（Meta，2024）

将推荐模型的缩放轴从**稀疏扩展**（扩大 Embedding 表）转向**密集扩展**（堆深交叉层），核心论点是：Embedding 表增大不能增强特征交叉能力，且受限于内存带宽，无法利用 GPU 算力提升；真正的瓶颈是交叉层的表达能力。

**1. Factorization Machine Block（FMB）**：每层的核心交叉模块。计算 $XX^\top$ 捕捉所有特征对的两两点积，再经 flatten → LN → MLP → reshape，将 $n$ 个输入 embedding 编码为 $n_F$ 个新 embedding。MLP 的角色是把交叉结果重新编码为新的语义表示，而非仅做特征检测。实际用低秩近似 $X(X^\top Y)$（$Y \in \mathbb{R}^{n \times k}$）将复杂度从 $O(n^2 d)$ 降至 $O(nkd)$。

**2. Linear Compression Block（LCB）**：与 FMB 并行的轻量分支，$W_L X_i$ 线性重组输入 embedding，不引入新的交叉阶数，专门保留低阶信息。作用是保证第 $i$ 层的输出始终覆盖 $1$ 到 $2^i$ 阶，而非只有高阶项——单独去掉 LCB 损失有限，但与残差连接同时去掉时 LogLoss 退化 1.84%。

**3. 层结构与指数阶数增长**：每层并行运行 FMB 和 LCB，拼接后加残差、Post-LN 输出：$X_{i+1} = \text{LN}([\text{FMB}(X_i) \| \text{LCB}(X_i)] + W_{\text{res}} X_i)$。第 $i$ 层可捕捉 $1$ 到 $2^i$ 阶交叉，层数线性增加而交叉阶数指数增长，Post-LN 保证每层输出归一化，FM 点积始终在受控尺度内。

在 146B 样本的内部数据集上建立了推荐领域首个 scaling law：$y = -100 + 99.56x^{0.00071}$，算力每翻 4 倍 LogLoss 改善约 0.1%；DCNv2 等竞品在 30~40 GFLOP 后均饱和或训练崩溃，Wukong 跨两个数量级保持稳定提升。详见 [[2024][Wukong]](<[2024][Wukong] Wukong: Towards a Scaling Law for Large-Scale Recommendation.md>)。

### Hiformer（2023）

将 Transformer self-attention 引入特征交叉，核心贡献是两处异构化改造：

**1. Heterogeneous Attention（HeteroAtt）**：Vanilla attention 所有 token 共享同一套 $W_Q/W_K/W_V$，HeteroAtt 给每个特征 $i$ 分配独立的 $Q_i$、$K_i$、$V_i$，特征对 $(i, j)$ 的 attention score 变为 $e_i Q_i^h (e_j K_j^h)^\top / \sqrt{d_k}$，不同特征在各自专属的语义空间里完成投影。

**2. Hiformer = HeteroAtt + Composite Projection**：在 HeteroAtt 基础上进一步增强 Key 的表达力——Key 不再只是单特征投影，而是把所有特征拼接后整体投影（$\hat{K}^h \in \mathbb{R}^{Ld \times Ld_k}$），让每个 Key 感知全局特征上下文。代价是计算量升到 $O(L^2 d^2)$，需要 low-rank 近似 + 末层 pruning 才能部署。

**3. Per-feature FFN**：每个特征有独立的 $W_1^i / W_2^i$，用 GELU 激活，避免共享 FFN 把不同特征的表征"揉"向同一方向。

局限：多层叠加后 token 相似度仍达 0.5~0.68，representation collapse 问题未从根本上解决。

### Zenith（2025，ByteDance）

以"token 异质性"为核心原则——通过在 attention 投影和 FFN 两个环节均**不跨 token 共享参数**来主动维持异质性，分三阶段处理特征：

**1. Prime Tokenization（特征压缩）**：TikTok Live 有 4,552 个原始特征，其中大量特征语义相近（如多个刻画用户活跃度的统计量），直接送入 attention 既冗余又低效。Zenith 将特征按语义分组，每组通过独立 MLP 聚合为一个高维 Prime Token（共 T=32 个）：相似特征的信息被压缩提炼，信息密度更高，后续 attention 也只需处理 T 个 token 而非 K 个原始特征。ID 类特征（user_id、item_id）语义独立，直接作为单独 token，不经过 MLP 聚合。

**2. Token Fusion（跨 token 交叉）**：在 Prime Token 之间建模 pairwise 协同信息。Vanilla attention 所有 token 共享同一套 $W_Q/W_K/W_V$，多层后 token 趋于相似（相似度 0.5~0.68），容量浪费。Zenith 用 RSA（线性 attention + retokenization reshape），Zenith++ 用 TMHSA——每个 token 拥有独立的投影矩阵，不跨 token 共享，从根本上避免异构特征被压入同一空间。

**3. Token Boost（token 内增强）**：共享 FFN 会把所有 token 的表征"揉"向同一方向（representation collapse），Token Boost 改用每个 token 独立的变换参数，主动维持 token 异质性。Zenith 用 TSwiGLU（tokenwise 门控 FFN），Zenith++ 升级为 TSMoE（稀疏专家混合，不增加推理计算量的前提下扩展容量）。

最终将 token 相似度从 0.5~0.68 压至 0.06~0.47，实现有效 scaling。详见 [[2026][Bytedance] Zenith](<[2026][Bytedance] Zenith Scaling up Ranking Models for Billion-scale Livestreaming Recommendation.md>)。

