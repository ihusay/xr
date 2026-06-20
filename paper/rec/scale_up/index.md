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

## 核心方案

### Embedding → Token 的构造方法

各方法在 lookup 阶段高度趋同（类别特征查表、dense 经 MLP 聚合成少量向量、lookup 阶段保持简单把交叉留给后续），真正的分歧在于 **embedding 如何组织成 token**，可归为三种范式：

| 处理维度 | 特征级 token | 语义压缩 token | 位置切片 token |
|---|---|---|---|
| **代表** | Wukong、Hiformer | Zenith | MixFormer、UniMixer |
| **一个 token = ?** | 单个 embedding 向量 | 一组语义相近特征压成的高维向量 | concat 后等长切的一片 |
| **构造操作** | lookup 直接成 token（multi-hot 先 sum pooling，dense 经 MLP+split） | 分组 concat → per-group MLP 投影；ID 特征单独成 token | 全部或 domain-ordered embedding concat → chunk 切片 → 独立 $W_j$ 升维 |
| **分组依据** | 无（一特征一 token） | 人工语义分组（语义相近放一组） | 位置等分；UniMixer 先按 domain 组织 embedding，concat 后仍保留弱语义顺序 |
| **token 数** | 与特征数同阶（Wukong $n$；Hiformer $L=\lvert C\rvert+n^D+t$） | $T=32$（从 $K=4552$ 压来） | $N=8\sim16$ 固定超参 |
| **token 数 vs 特征数** | 同阶，随特征膨胀 | 语义解耦，分组决定（仍可能涨） | 完全固定，与特征数无关 |

**后两种方法的关键对立**（都压缩，分歧只在怎么分组）：

- **语义压缩**认为 token 必须语义纯净——因为它是 self-attention 单元，混入杂特征会引入 attention 噪声 → 人工语义分组
- **位置切片**认为 token 是 cross-attention 的 query 不是 self-attention 单元，纯净度无所谓，HeadMixing 能补语义模糊，几千特征人工分组是负担 → 位置等分即可

UniMixer 处在两者之间：原文先按 User/Profile/Behavior/Query 等语义 domain 做 embedding，再立即 concat 成长向量并按位置切 block。domain 分组不是后续建模的隔离边界，但 concat 顺序让 block token 带有弱语义布局。详见 [[2026][UniMixer][Kuaishou]](<[2026][UniMixer][Kuaishou] UniMixer A Unified Architecture for Scaling Laws in Recommendation Systems.md>)。

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

### OneTrans（2025，工业界）

将**序列建模**（用户行为历史）和**特征交叉**（user/item/context 属性）统一进同一个 Transformer stack，打破传统"先 encode 序列再做交叉"的两阶段流水线。

**1. 统一 Tokenization**：将所有输入统一为 token 序列——S-tokens（行为序列）和 NS-tokens（非序列特征，如 user/item profile、上下文）。两类 token 拼接后一起送入 Transformer，NS-token 在 attention 中自然聚合全部行为历史。

**2. 混合参数化（Mixed Parameterization）**：S-tokens 语义同质（都是行为），共享 Q/K/V 投影和 FFN；NS-tokens 语义异质（不同属性），每个 token 拥有独立投影矩阵和独立 FFN。这一设计与 Zenith 的"token 异质性"原则一致——不强迫异构特征共享参数空间，同时保持序列侧的计算效率。

**3. Pyramid Stack**：逐层对 S-tokens 做 progressive pruning（如 1190→12 个），信息向 NS-tokens 汇聚，将 attention 复杂度从 $O(L^2 d)$ 降至 $O(LL'd)$。

**4. 工程优化**：Cross-request KV caching 在多个候选之间复用 S 侧计算，将每请求序列计算量从 $O(L)$ 降至 $O(\Delta L)$；结合 FlashAttention-2 部署。

线上 A/B 结果：Click/user +5%～+8%，GMV/user +3.7%～+5.7%，同时延迟下降 3%～4%。在离线 scaling 曲线上斜率优于 RankMixer，参数效率更高。

### MixFormer（2026，ByteDance）

面向工业推荐里 dense 特征交叉和长行为序列建模分离的问题。传统架构通常先单独 encode 用户行为序列，再把序列表征作为特征交给 dense ranking tower，两个模块竞争同一计算预算，且只在固定接口处传递信息，无法协同 scaling。

MixFormer 的做法是在每个 block 内让非序列特征（NS-tokens）直接 attend 用户行为序列（S-tokens）：NS 特征先 concat 后按位置等长切成多个 head，经 Query Mixer 做无参数 HeadMixing，再作为 query 去 cross-attend 行为序列，最后通过 per-head SwiGLU 做 Output Fusion。这样 dense 特征交叉和序列建模不再是两个独立模块，而是在层内深度融合、梯度双向流动。

关键设计是 Query Mixer 中的 HeadMixing：把 $N$ 个 head 的表示 reshape 成 $N \times N \times D/N$，转置前两维后 flatten，用无参数转置实现跨 head 信息交换，避免 self-attention 的 $QK^T$ 计算。线上 A/B 在抖音场景带来 Finish Rate +0.3897%、Comments +0.7035%。详见 [[2026][MixFormer][ByteDance]](<[2026][MixFormer][ByteDance] MixFormer Co-Scaling Up Dense and Sequence in Industrial Recommenders.md>)。

### UniMixer（2026，Kuaishou）

Heterogeneous Attention 虽然可学习，但在异构推荐特征上用 $QK^T$ 内积决定交互强度，训练早期容易被输入 token 数值主导，产生不稳定的 attention pattern；TokenMixer 虽然高效稳定，但 mixing 规则固定、不可学习，难以适配不同业务场景。

UniMixer 的做法是把 TokenMixer 的固定置换矩阵参数化，拆成 block 内 local mixing $W_B^i$ 和 block 间 global mixing $W_G$，让原本规则化的 mixing pattern 变成可学习的 heterogeneous feature interaction。更一般地看，它把 attention-based、TokenMixer-based、FM-based 三类推荐 scaling block 统一为"Local Mixing Pattern + Global Mixing Pattern"。

Feature Tokenization 上，UniMixer 先按语义 domain 生成 embedding，再 concat 成 $E=[e_1,\ldots,e_N]$，随后等长切 block 并用 token-specific linear layer 得到 $X \in \mathbb{R}^{T \times D}$。这个设计不是 Zenith 式语义纯 token，也不是完全无语义的随机切片，而是"domain-ordered concat 后的位置切片"。

UniMixing-Lite 进一步用低秩 $A_GB_G$ 压缩 global mixing，并用 basis matrices 组合生成 local mixing，在快手广告留存任务上取得比 RankMixer 更高的参数/FLOPs scaling exponent。详见 [[2026][UniMixer][Kuaishou]](<[2026][UniMixer][Kuaishou] UniMixer A Unified Architecture for Scaling Laws in Recommendation Systems.md>)。
