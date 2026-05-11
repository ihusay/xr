# Label Modeling in Industrial Systems

用户行为 label 的设计直接决定优化目标与业务对齐程度。以下按场景梳理常见 label 建模方式。

---

## 1. 推荐系统场景

### 1.1 隐式反馈（Implicit Feedback）
最常见的 label 来源，用户没有显式打分，只有行为信号。

| Label | 定义 | 问题 |
|---|---|---|
| Click | 用户点击 = 1 | 误点、标题党污染严重 |
| Dwell Time | 停留时长 > 阈值 | 阈值敏感，跨内容类型不可比 |
| Finish Rate | 视频/文章读完比例 | 长内容天然吃亏 |
| Like / Share / Comment | 显式正向互动 | 稀疏，但质量高 |
| Negative Feedback | 不感兴趣、举报 | 稀疏，常被忽视 |


---

## 2. 电商场景

电商的核心矛盾：**转化链路长 + 反馈延迟 + 退货污染**。

### 2.1 标准转化漏斗 Label

```
Impression → Click → Add-to-Cart → Order → Pay → No-Refund
```

| Label | 建模目标 | 典型稀疏度 |
|---|---|---|
| CTR | P(click \| impression) | ~1–5% |
| CVR | P(order \| click) | ~1–10% |
| CTCVR | P(order \| impression) = CTR × CVR | 极稀疏 |
| NetCVR | P(order ∧ ¬refund \| click) | 更稀疏 |

### 2.2 Delayed Feedback 问题
转化 label 不能立即观测，存在延迟窗口：

- **Fake Negative**：样本截止时未转化，但之后会转化 → label=0 实为假负例
- **解法**：
  - Elapsed Time 建模（DFM，Chapelle 2014）：把延迟时间本身建模为指数分布
  - 重要性采样（FNC）：对假负例重新加权
  - 流式更新：样本到达时先标 negative，转化后发 delayed positive 事件更新

### 2.3 Cascaded Delayed Feedback → NetCVR

NetCVR 引入两段级联延迟，方向相反：

```
Click --[d1]--> Conversion --[d2]--> Refund
```

- d1 短 → CVR 高（正向）
- d2 短 → RFR（退货率）高（负向）

同一特征"延迟短"对两个阶段含义相反，不能用单段延迟方法直接套用。

**TESLA**（WWW 2026，阿里）：将 NetCVR 分解为 CVR × (1 - RFR) 两个任务分别建模去偏，再引入 delay-aware ranking loss 稳定排序；配套开源 CASCADE 数据集（来自淘宝，首个 NetCVR 公开 benchmark）。

### 2.4 GMV / Revenue Label → 延迟反馈回归

直接优化 GMV（连续回归量）而非转化率（二分类），挑战更大：

- Label = Σ purchase_price，一次点击可触发多次购买，label 在归因窗口内累积更新 N 次
- 53% 的样本是 repurchase 样本，其 GMV 分布与单次购买差异极大，统一建模效果差

**READER**（WWW 2026，阿里 + 厦门大学）：
- 双分支 + 独立路由器：router 预测是否 repurchase，将样本路由到对应专家塔，软融合处理不确定样本
- 三层去偏：Label Calibrator 在线预测 label 缺口并补偿；归因窗口关闭后用真实 label 做 GRA 对齐；PLU 逆向消除 calibrator 对最后一次购买的过度修正

---

## 3. LTV（Life-Time Value）场景

LTV = 用户在生命周期内带来的总收益，label 建模难度最高。

### 3.1 LTV 的时间结构

```
注册 → 首单 → 复购1 → 复购2 → ... → 流失
```

LTV label 本质上是一个**右截尾的时序累计值**，不可能等完整生命周期再训练。

### 3.2 常见建模策略

| 策略 | 做法 | 问题 |
|---|---|---|
| 代理 label | 用 N 日收入代替 LTV | 短期≠长期，低估重度用户 |
| 分解建模 | P(留存) × E(每日价值) | 误差传播 |
| 生存分析 | 建模用户流失时间分布（Weibull / Cox） | 难以捕捉收入异质性 |
| ZILN | 零膨胀对数正态，处理大量零值 + 长尾分布 | Google 推荐，适合电商 |

### 3.3 右截尾与 Delayed Label
- 新用户 LTV label 必然不完整，需要用早期行为预测长期价值
- 常见做法：用 7d/30d 行为特征预测 180d LTV，隐含假设早期行为可预测长期

### 3.4 LTV 分层
不直接回归 LTV 值，而是分层（高/中/低价值用户）做分类，降低长尾 label 方差影响。

---

## 4. 跨场景共性问题

| 问题 | 描述 | 典型解法 |
|---|---|---|
| 样本选择偏差 | CVR 只能在点击样本上训练 | ESMM、反事实估计 |
| Position Bias | 靠前位置点击率虚高 | PAL、Inverse Propensity Score |
| Label Sparsity | 正样本极稀疏 | 多任务、辅助 label |
| Delayed Feedback | label 延迟到达 | 重要性采样、流式更新、生存分析 |
| Label Noise | 误点、恶意刷单 | 鲁棒损失函数、置信度建模 |

---