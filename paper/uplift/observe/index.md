# Observational Data 与 RCT 融合

## 背景

在 Uplift/CATE 场景里，常见的数据来源有两类：Observational data 和 RCT data。

- **Observational data**：历史日志，规模大，包含用户特征、历史 treatment、结果反馈。
- **RCT data**：随机实验数据，因果信号更干净，但实验成本高、预算小，难以全量覆盖。

核心矛盾是：OBS 数据便宜但有偏，RCT 数据可信但稀缺。

## 论文

- [[2026][Didi][BAE] Budgeted Active Experimentation for Treatment Effect Estimation from Observational and Randomized Data](<[2026][Didi][BAE] Budgeted Active Experimentation for Treatment Effect Estimation from Observational and Randomized Data.md>)
