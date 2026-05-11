# Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction

WWW 2026 | 阿里巴巴 | [paper](https://arxiv.org/abs/2601.19965) | [code](https://github.com/alimama-tech/NetCVR)

---

## 问题定义

**NetCVR**：用户点击后完成购买且不退货的概率。

```
Click --[d1]--> Conversion --[d2]--> No Refund
NetCVR = P(conversion ∧ ¬refund | click)
```

与传统 CVR 的区别：两段级联延迟（d1、d2）方向相反——延迟短同时意味着高 CVR（好）和高退货率（坏），导致单段延迟方法不能直接套用。

---

## 核心贡献

### 1. CASCADE 数据集
首个面向在线持续 NetCVR 预估的大规模开源数据集，来自淘宝。记录完整用户链路（点击→转化→退货），每个事件带时间戳，支持流式实验环境复现。

**主要价值**：填补了 NetCVR 场景没有公开 benchmark 的空白。

### 2. 三个数据洞察
基于 CASCADE 的分析结论，指导方法设计：
1. NetCVR 存在明显时序模式，需要在线持续训练而非离线批训练
2. 分解建模（CVR × RFR）优于直接建模 NetCVR
3. 延迟时长与 CVR、RFR 均相关，是有效的预测信号

### 3. TESLA 框架
首个面向 NetCVR 的在线持续训练方法，三个设计点：

- **CVR-RFR 级联架构**：两个任务共享底层参数，分别预测转化率和退货率，最终相乘得到 NetCVR
- **分阶段去偏**：两段延迟分开处理，各自做重要性加权。以 CVR 塔为例：

  $$w_v^+ = 1 + p_v(x) \cdot P(h_v > W_v^{obs} \mid y=1, x)$$

  $$w_v^- = \frac{(1-p_v(x)) \cdot w_v^+}{1-p_v(x) + p_v(x) \cdot P(h_v > W_v^{obs} \mid y=1, x)}$$

  RFR 塔结构相同，变量换成 $p_r, h_r, W_r^{obs}$。

  **正样本权重的逻辑**：观察到的正例需要替同类被漏掉的假负例发声，权重 = 自身（1）+ 需要替代的假负例数量：

  $$P(\text{假负例}) = p_v(x) \cdot P(h_v > W_v^{obs} \mid y=1, x)$$

  - $p_v(x)$ 低：这类样本本来就少转化，漏掉的绝对数量少，补偿小
  - $P(h_v > W_v^{obs})$ 低：大多数转化在窗口内已被观察到，漏掉的比例小，补偿小

  负样本权重反向，对可能是假负例的样本降低置信度。本质是 FNC 的软加权版——FNC 对假负例补训一次（权重=1），TESLA 对所有样本按概率连续软加权，理论上更严格。
- **Delay-Aware Ranking Loss**：在 label 不完整时稳定正负例相对排序，三个子模块：

  **4.4.1 正例加权**：延迟越短的正例权重越高，视为更可靠的转化信号：

  $$w_i = w_{\min} + \alpha \cdot \sigma\left(\frac{m - h_v^{(i)}}{s}\right)$$

  **4.4.2 负例采样**：按模型预测 CVR 低的概率优先采样，过滤掉可能是假负例的高 CVR 负样本：

  $$\pi_j = \frac{(1 - \hat{p}_v(x_j))^{1/\tau}}{\sum_{k \in \mathcal{N}_v} (1 - \hat{p}_v(x_k))^{1/\tau}}$$

  **4.4.3 最终 Ranking Loss**：对每个正例采 K 个负例构成对，加权 pairwise loss：

  $$\mathcal{L}_v^{DAR} = \sum_{i \in \mathcal{P}_v} w_i \cdot \left( -\frac{1}{K} \sum_{j \sim \pi}^K \log \sigma(o_i^{CVR} - o_j^{CVR}) \right)$$

  RFR 塔结构对称。最终总 loss 四项之和：

  $$\mathcal{L} = \mathcal{L}_v + \mathcal{L}_r + \mathcal{L}_v^{DAR} + \mathcal{L}_r^{DAR}$$

  其中 $\mathcal{L}_v, \mathcal{L}_r$ 是 pointwise debiasing loss（4.3），$\mathcal{L}_v^{DAR}, \mathcal{L}_r^{DAR}$ 是 pairwise ranking loss。前者校准概率，后者保证排序，两者互补。

---

## 方法局限

- 延迟信号仅用于 loss 加权，未进入模型作为输入特征（inference 时 delay 不可得）
- 技术模块（多任务分解、delayed feedback 加权）均为已有方向的组合，核心贡献更多在问题定义与数据集
