# LLM Rec 笔记

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