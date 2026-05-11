# Delay Label

## 核心问题

1. **数据新鲜度 vs label 完整性不可兼得**：等 label 完整再训练则数据太旧；用新鲜数据训练则 label 未完成，存在大量假负例。

2. **delay 时间本身包含行为信息**：延迟不只是需要克服的噪声，延迟时长本身反映用户决策意图，可以被建模利用。

## 核心技术方案


### 1. Label Correction / Importance Weighting

核心范式：新样本到达立即作为临时负例，转化到达时以正例重新送入，同时用重要性权重修正假负例引入的偏差。

**数据流结构**

```text
click 到达  -> 立即以 negative 送入训练流
conversion 到达  -> 同一样本以 positive 重新送入训练流
```

两次送入同一样本会引入 double counting 偏差，需要 importance weighting 修正。

**权重设计**

- **FNW**：对重新送入的 positive 加权 `1 + P(delay > elapsed_time | x)`，elapsed_time 越短权重越大，抵消假负例阶段的梯度污染
- **ES-DFM**：改在采样阶段介入，未转化样本按 elapsed time 决定是否进入 negative 流（时间越长越可信为真负例），减少假负例数量，再配合 importance weight 校正残余偏差
- **DEFER**：在 label correction 上进一步稳定，降低权重估计误差带来的方差

优点：维护单一训练流，工程简洁，天然适合在线连续训练。

缺点：权重估计不准时偏差和方差都会放大；double counting 的修正依赖对 delay 分布的估计质量。

> **附注：延迟送入 negative**
>
> 上述范式默认 click 到达即送入 negative，但也可以为每个样本预估一个期望正例到达时间（如基于历史 delay 分布估计的 `E[delay | x]`），负例推迟到该时间点才进入训练流。这样在等待窗口内既不作为假负例污染梯度，也不需要事后用权重修正，代价是引入一定的训练延迟。

---

### 2. Multi-Window / Multi-Stream

把不同延迟阶段拆成多个数据流或多个窗口建模。

```text
短窗口：保证新鲜度
长窗口：保证 label 完整性
多流融合：平衡 bias / variance
```

代表方法：

- **DEFUSE**
- **DDFM**
- **DFSN**
    - 构建两类卫星模型：特征卫星（短窗口，无复制机制，捕捉最新特征分布）和无偏卫星（无窗口，重要性采样校正偏差）
    - Embedding Transfer：将卫星的 embedding 层与主模型 embedding 线性加权融合，刷新主模型的特征表示
    - Strategy Aggregation：主模型 logit 与无偏卫星 logit 直接相加后 sigmoid 归一化，特征卫星 logit 仅作预测下界（取 max），两者联合投票
    
- **MISS**
    - 多个等待窗口对应多个平等的 output head，共享底层参数，各 head 独立训练在各自的 D⁺_{τ,dᵢ} 上，保留不同窗口的"个性"
    - Assembled Pipeline Synthesizing：用轻量模型在 assembled pipeline（最新正样本 + 真实负样本）上学习 softmax 动态权重，加权聚合各 head 预测
    - Global Positive Weighting：全局放大正样本权重，无需额外模型，轻量消除 CVR 系统性低估

优点：能同时利用短期反馈和长期完整 label。

缺点：数据流设计复杂，需要处理不同窗口之间的样本重复、权重和一致性。

---

## Delay 作为信号

delay 不只是噪声，也可能是行为信号。

例如：

- 快速购买：可能表示购买意图强
- 长时间后购买：可能表示决策链路更弱
- 快速退款：可能表示明显不满意或冲动购买

因此有两种用法：

### 1. Debias signal

用 delay distribution 估计当前 label 是否完整：

```text
真实会转化，但还没观察到的概率
真实会退款，但还没观察到的概率
```

### 2. Training signal

在 loss 里使用 delay time：

```text
delay 越短的 positive sample 权重越高
```

TESLA 的 Delay-Aware Ranking Loss 就属于这种做法。

---

