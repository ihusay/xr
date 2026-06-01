# LLM Rec 笔记

## GR（Generative Recommenders，Meta，2024）

**论文**：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations（[arxiv 2402.17152](https://arxiv.org/abs/2402.17152)，ICML 2024）

把推荐问题重新表述为**序列转导任务（sequential transduction）**，用 decoder-style Transformer 直接生成预测，而非传统的 pointwise 打分。核心架构是 HSTU（Hierarchical Sequential Transduction Units），专为推荐场景高基数、非平稳 streaming 数据设计。

**关键结论**：
- 模型质量随训练计算量呈幂律 scaling，跨三个数量级，与 GPT-3/LLaMA-2 规模相当
- 1.5 万亿参数版本在线 A/B 提升 12.4%，已部署于数十亿用户规模的平台
- 在 8192 长度序列上比 FlashAttention2-based Transformer 快 5.3x～15.2x

生成式推荐的标志性工作，验证了推荐领域同样存在 scaling law。

---

## MTGR（美团，2025）

**论文**：MTGR: Industrial-Scale Generative Recommendation Framework in Meituan（[arxiv 2505.18654](https://arxiv.org/abs/2505.18654)，CIKM 2025）

GR 的工业落地改进版。原始 GR 抛弃了传统 DLRM 精心设计的 cross feature，导致性能下降且 scaling 无法弥补。MTGR 在 HSTU 架构基础上**同时保留原始特征和 cross feature**，将生成式序列建模与传统特征交叉能力结合。

**关键设计**：
- 基于 HSTU 建模用户级序列数据，保留 DLRM 的 cross feature
- 定制 masking 策略：静态序列信息（user profile、历史行为）对所有 token 可见；动态序列信息（近期交互）遵循 causal visibility；candidate token 只能看到自身
- 通过用户级压缩实现训练和推理加速

多任务（CTR + CTCVR）联合训练，AUC/GAUC 均有提升。

---

## SIGMA (2602.22913)

### GR（生成式推荐）的三个核心局限

**1. 任务单一（Task Versatility）**

现有 GR 方法大多只针对 next-item prediction，无法覆盖多样化业务需求（节日主题推荐、商品特征推广等）。模型设计时没有考虑多任务泛化，任务形式固化。

**2. 过度依赖行为数据（Behavioral Dependency）**

主流 GR 方法严重依赖历史行为数据，导致：
- 对动态市场趋势响应滞后（新品、节日、突发热点无历史信号）
- 冷启动场景表现差（新用户/新商品行为稀疏）

模型推荐信号完全来自交互日志，缺乏外部语义补充。

**3. 未充分利用 LLM 语义能力（Knowledge Integration）**

LLM 本身具备丰富语义理解和参数化知识，但传统 GR 未有效利用：
- 没有 instruction-following 机制，无法灵活响应自然语言业务需求
- item 表征停留在 ID 层面，缺乏语义对齐

**SIGMA 对应设计：**

| 局限 | SIGMA 解法 |
|------|-----------|
| 任务单一 | 多任务指令驱动训练（instruction-driven multi-task） |
| 依赖行为数据 | 语义锚定的 item tokenization，引入语义信号 |
| LLM 能力未用足 | 统一语义空间 + 自适应概率融合生成 |