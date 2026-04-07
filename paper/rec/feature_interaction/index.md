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

| 模型 | 交叉方式 | 局限 |
|------|---------|------|
| FM | 二阶内积 $\langle v_i, v_j \rangle x_i x_j$ | 只有二阶，参数共享限制表达 |
| DCN-V2 | 有界阶数多项式交叉 | scaling 效率受限 |
| Hiformer | Transformer | vanilla attention not expressive enough |
| **Zenith** | tokenwise 参数化（TMHSA + TSMoE） | — |

---

## Vanilla Attention 为何不够

所有 token 共享同一套 W_Q/K/V，不同语义特征被压到同一空间：
- 多层后 token 趋于相似（相似度 0.5~0.68），容量浪费
- 参数量增加但性能不再提升

**Zenith 的解法**：每个 token 独立投影矩阵（TMHSA）+ tokenwise 稀疏 MoE，将 token 相似度压至 0.06~0.47，保持 token 异质性，实现有效 scaling。详见 [[2026][Bytedance] Zenith](<[2026][Bytedance] Zenith Scaling up Ranking Models for Billion-scale Livestreaming Recommendation.md>)。

---

## 相关论文

- **[19] Persia (KDD 2022)**：稀疏 embedding 扩展至 100T 参数
- **[22] Meta 分布式训练 (2021)**：大规模 embedding table 训练效率
- **[1] Understanding Scaling Laws for Rec (Meta 2022)**：推荐 scaling 规律
- **[32] Wukong (2023)**：因子化交叉层 + 推荐 scaling law
- **[7] Hiformer (2023)**：提出 vanilla attention not expressive enough
- **Zenith (2025)**：token 异质性原则，TikTok Live 部署
