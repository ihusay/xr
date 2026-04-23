# Hiformer: Heterogeneous Feature Interactions Learning with Transformers

Google DeepMind / Google Inc，2023，部署于 Google Play App 排序。

---

## 问题

Vanilla Transformer 的 $W_Q/W_K/W_V$ 所有特征共享（homogeneous attention）。NLP 中 token 语义上下文无关，共享合理；但推荐系统中特征语义高度异构且上下文相关（app_id 的含义依赖 hour_of_day、user_country），共享投影矩阵导致特征间的语义对齐信息丢失，表达能力受限。

AutoInt 已验证 attention 用于特征交叉有效，但仍是 homogeneous 设计。

---

## 核心方法

### 1. Preprocessing Layer

输入特征分两类处理，最终统一成长度 $L = |C| + n^D + t$ 的 embedding 序列送入 Feature Interaction Layer。

**Categorical Features**：embedding lookup，每个类别特征有独立的投影矩阵 $W_i^C \in \mathbb{R}^{V_i \times d}$：

$$\mathbf{e}_i = \mathbf{x}_i^C W_i^C$$

**Dense Scalar Features**：数值特征数量多（$|D|$ 个）且分布差异大。先 normalize 对齐分布，全部拼接后用 MLP $f_D$ 压缩，再 split 成 $n^D \ll |D|$ 个 embedding：

$$\mathbf{e}_i = \text{split}_i\!\left(f_D\!\left(\text{concat}(\text{normalize}(\{x_i^D\}))\right),\ \text{split\_size}=d\right)$$

本质是用 $f_D$ 把大量数值特征聚合为少量 token，减少后续 attention 的序列长度，降低推理复杂度。

**Task Token**：每个任务（点击、安装…）学一个 task embedding，类似 CLS token，共 $t$ 个。

### 2. Heterogeneous Attention Layer（HeteroAtt）

为每个特征 $i$ 学习独立的 Query 投影 $Q_i$、Key 投影 $K_j$，特征对 $(i, j)$ 的 attention score 变为：

$$\phi^h_{i,j}(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \mathbf{Q}_i^h (\mathbf{e}_j \mathbf{K}_j^h)^\top}{\sqrt{d_k}}$$

$$\text{Att}(i,j)^h = \frac{\exp(\phi^h_{i,j}(\mathbf{e}_i, \mathbf{e}_j))}{\sum_m \exp(\phi^h_{i,m}(\mathbf{e}_i, \mathbf{e}_m))}$$

输出：

$$\mathbf{o}_i = \text{concat}\!\left(\left\{\sum_j \text{Att}(i,j)^h \mathbf{e}_j \mathbf{V}_j^h\right\}_{h=1}^H\right) \mathbf{O}_j$$

与 vanilla attention 的本质区别：$\mathbf{Q}_i$ 是特征 $i$ 专属，$\mathbf{K}_j$ 是特征 $j$ 专属，参数量随特征数线性增长（而非固定一套）。

### 3. Hiformer

在 HeteroAtt 基础上进一步提升表达力：Key 不再只是单特征投影，而是把整个特征列表拼接后再投影（Composite Key）：

$$[\hat{\mathbf{k}}_1^h, \ldots, \hat{\mathbf{k}}_L^h] = \text{concat}([\mathbf{e}_1^h, \ldots, \mathbf{e}_L^h])\hat{\mathbf{K}}^h$$

其中 $\hat{\mathbf{K}}^h \in \mathbb{R}^{Ld \times Ld_k}$。这让每个 Key 都能感知到所有特征的全局上下文（cross projection），而不只是单个特征的局部信息。

**推理瓶颈**：Composite projection 的 QKV 计算为 $O(L^2d^2)$，比 vanilla $O(Ld^2)$ 高一个数量级。两个优化手段压回与 vanilla 同阶：
- **Low-rank approximation**：$\hat{\mathbf{K}}^h \approx \mathbf{L}_k^h (\mathbf{R}_k^h)^\top$，秩为 $r_k \ll Ld_k/2$，把 QKV 计算压到 $O(Lr_k(d+d_k))$
- **Model pruning**：最后一层只用 task embedding 做 Query，特征 embedding 做 Key/Value，复杂度从 $O(L^2)$ 降到 $O(L)$

### 4. Per-feature FFN

每个特征有独立的前馈网络，用 GELU 激活：

$$\text{FFN}^i_{\text{GELU}}(\mathbf{o}_i) = \text{GELU}(\mathbf{o}_i \mathbf{W}_1^i + \mathbf{b}_1^i)\mathbf{W}_2^i + \mathbf{b}_2^i$$

### 5. Task Token

类似 CLS token，每个任务（点击、安装、购买…）学一个 task embedding 加入特征序列，模型通过 attention 自动聚合对该任务最相关的特征信息，支持多任务共享主干网络。

---

## 与 AutoInt / Vanilla Transformer 的对比

| | Vanilla Transformer | AutoInt | HeteroAtt | Hiformer |
|---|---|---|---|---|
| Q/K/V 投影 | 所有特征共享 | 所有特征共享 | 每特征独立 | 每特征独立 + Composite |
| FFN | 共享 | 共享 | 共享 | 每特征独立 |
| 推理效率 | 高 | 高 | 高（同 vanilla FLOPs） | 低-rank + pruning 后可部署 |
| 任务感知 | 无 | 无 | 无 | Task Token |

---

## 局限（Zenith 指出的）

- 多层后 token 相似度仍达 0.5~0.68：Composite projection 提升了单层表达力，但没有从根本上阻止多层叠加后的 representation collapse
- 未解决 FFN 的 homogenization 问题（虽然有 per-feature FFN，但 Zenith 认为仍不够）

Zenith 的 TMHSA + TSMoE 在 Hiformer 的思路上更进一步，把 token 异质性作为 **scaling 的前提条件**显式维护。

---

## 效果

Google Play App 排序离线 + 在线 A/B 测试，关键 engagement 指标 **+2.66%**。

---

## 论文评论

### 核心价值点讨论：per-feature Q/K 真的解决了"上下文依赖语义"吗？

#### 论文的卖点（Intro 原文）

> Suppose we are recommending food delivery apps to users and we have the features: `app_id`, `hour_of_day`, and `user_country`. In this case, `app_id` could mean one thing for `hour_of_day` and another for `user_country`. To effectively detect different features, we would need semantic awareness and semantic space alignment. Transformer models unfortunately do not consider this. In the vanilla attention layer, the features are projected through the same projection matrices (i.e., $\mathbf{W}_Q$ and $\mathbf{W}_K$) that are shared across all features. This is a natural design for applications where the feature (text token) semantics are independent of the context; however for recommender systems where the feature semantics are often dependent of the context, this homogeneous design would lead to limited model expressiveness.

论文用这个例子论证：推荐特征**上下文相关**（`app_id` 对不同交互伙伴有不同含义），而 NLP 中 token 语义上下文无关，所以 vanilla attention 的共享 $W_Q/W_K$ 在推荐场景下表达力不足，需要 per-feature Q/K。

#### 评价：结论对，论据不严谨

**漏洞 1：例子与设计不匹配**

例子说的是 `app_id` 对 `hour_of_day` 有一种含义、对 `user_country` 有另一种含义——这暗示 `app_id` 本身应该在不同上下文里**变脸**。

但 Hiformer 的实际设计是：给 `app_id` 一个**固定的** $Q_{\text{app\_id}}$，和它交互的 `hour_of_day`、`user_country` 各有不同的 K。当 `app_id` 分别和两者交互时，用的是**同一个** $Q_{\text{app\_id}}$。

换句话说：论文说 "app_id 在不同上下文里含义不同"，设计上只是"app_id 有固定投影，只不过 pair 的另一方投影不同"——严格来说只是把一半的异构性做进 Key，没有让 `app_id` 真正"变脸"。

真正匹配这个例子的应该是某种 bilinear 或 FiLM-like 机制，让 $Q$ 本身是 $(i, j)$ 的函数。论文没走到这一步，但用例子暗示了更强的能力。

**漏洞 2：vanilla attention 并非无法区分特征**

Vanilla attention 里虽然 $W_Q/W_K$ 共享，但输入 embedding $\mathbf{e}_{\text{app\_id}}$、$\mathbf{e}_{\text{hour}}$、$\mathbf{e}_{\text{country}}$ 本就不同，经过同一个 $W_Q$ 投影后得到的 query 仍然不同。vanilla attention 不是"无法区分"，而是"区分能力受限于共享矩阵的表达力"——这是**量变**而非**质变**。

论文用 "limited expressiveness" 是对的，但例子暗示的是 qualitative 差异，两者对不上。

**漏洞 3：NLP 类比站不住**

论文说 "text token semantics are independent of the context"——这完全不对。`bank` 在 `river bank` 和 `investment bank` 下语义天差地别，整个 Transformer 的成功恰恰建立在 attention 捕捉**上下文相关词义**上，contextualized embeddings 是 BERT 以来的核心观念。

论文真正想说的应该是："NLP 里共享投影 + 足够深的网络可以涌现上下文语义，但推荐特征少、交互浅，来不及涌现，所以要硬编码进架构"——但它没这么说。

#### 实事求是的论证应该是

推荐特征的异构性有两层：
1. **词表异构**：`app_id`（高基数 ID）、`hour_of_day`（低基数类别）、`user_country`（地理类别）的取值空间、分布、统计性质完全不同，embedding 后的向量分布差异极大
2. **语义异构**：每对特征组合产生的 pairwise 信号需要不同的 projection 方向才能有效提取

Per-feature $Q/K$ 是**用参数换表达力**的简单有效手段，并非真正解决"上下文依赖语义"。Zenith 后来也承认：per-feature projection 只是 tokenwise 异质性的**一种初级形式**。

#### 一句话总结

论文用了一个直观但不精确的例子卖设计——直觉对，细节经不起抠。真正的理由是"推荐特征分布异构 + 交互深度浅 → 需要显式参数化容量"，而不是"NLP 上下文无关所以共享合理，推荐上下文相关所以必须独立"。

---

### 待讨论：Q/K/V 只能独享一个，谁应该独享？（感觉不太对，待澄清）

#### 初步结论：V 独享杠杆最大

三者在 attention $\text{softmax}(QK^\top/\sqrt{d}) V$ 中的角色：
- **Q**：查询意图（我想看什么）
- **K**：索引面（我对外怎么呈现自己）
- **V**：内容本身（我实际传递什么信息）

**Q 独享 / K 独享**：Q 和 K 是对偶的，只独享一边表达力差异不大。输入 embedding 本就不同，独享 Q/K 提升的是"查询/索引风格的多样性"，但影响的只是 attention 权重 $\alpha_{ij}$ 的分布。

**V 独享**：V 直接决定 attention 输出的表征空间。

#### 关键论据：representation collapse

$$\text{token}_i^{(l+1)} = \sum_j \alpha_{ij} \cdot (V \mathbf{e}_j)$$

token 新表征 = V 变换后邻居值的加权和。

- Q/K 只改变 $\alpha_{ij}$
- V 决定被加权的向量本身

V 共享 → 所有 token 最终都是"同一个 V 空间向量的线性组合" → 多层叠加必然塌缩到 V 列空间的低维子空间 → token homogenization。

V 独享 → 每个特征贡献自己专属子空间的向量 → 加权求和保留异质性。

#### 与 Zenith 的呼应

Zenith 强调 FFN 必须 tokenwise（TSwiGLU / TSMoE）——FFN 和 V 投影都作用在"被传递的内容"这条路径上，这条路径的异构性才是 scaling 能否有效的关键。Hiformer 把 budget 平均分给 Q/K/V，Zenith 的视角暗示应把重心放在 value 路径。

#### 存疑点

- 这个分析假设 V 共享是 representation collapse 的主因，但 Zenith 论文实际上归因于"所有 tokenwise 变换共享"，没有单独实验分离 Q/K/V 的贡献
- Hiformer 原文 $\mathbf{V}_j^h$ 其实也是 per-feature 的（公式 (3)），只是没有像 QK 那样在 Composite projection 里进一步强化——所以严格来说 Hiformer 已经 V 独享了，只是没强调
- "Q 和 K 对偶" 的说法略粗糙，softmax 之后 Q/K 的作用不完全对称
- **需要读 Zenith 的消融实验确认 Q/K/V 分别独享的效果**
