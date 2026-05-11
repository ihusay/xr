# Online Conversion Rate Prediction via Neural Satellite Networks in Delayed Feedback Advertising


**作者**：Qiming Liu, Haoming Li, Xiang Ao, Yuyao Guo, Zhihong Dong, Ruobing Zhang, Qiong Chen, Jianfeng Tong, Qing He
**机构**：中科院计算所 + 腾讯
**会议**：SIGIR 2023


---

## 问题背景

在线广告采用按转化付费（CPA）策略，需要实时预测转化率（CVR）。核心挑战是**延迟反馈**：用户点击广告后，转化行为（购买等）可能在数小时乃至数天后才发生。

这导致训练数据中存在大量**假负例（Fake Negatives）**：样本点击时尚未转化，被标为负例，但之后会真正转化。在这种偏置数据上训练会严重损害模型性能。

---

## 核心矛盾

| | 假负例 | 数据时效性 |
|---|---|---|
| 等待窗口越长 | 越少 ✓ | 越差 ✗ |
| 等待窗口越短 | 越多 ✗ | 越好 ✓ |

单靠调整窗口无法同时解决两个问题。

---

## 3.1 Data Pipeline

**符号定义**：

- D_τ：时刻 τ 的完整真实数据集（含所有已点击样本）
- D_{τ,d}：等待窗口为 d 的观测训练集，只含 τ-d 之前的样本
- P_{τ,d}：已错误标为负例、但在 τ 前已确认转化的样本（重新标为正例）
- D⁺_{τ,d} = D_{τ,d} ∪ P_{τ,d}：加入复制正样本后的调整训练集

**样本复制机制（Sample Duplication）**：将假负例在收到转化反馈后复制一份标为正例重新注入训练流，缓解假负例问题。

---

## 方法：DFSN（Delayed Feedback modeling via neural Satellite Networks）

### 总体架构

一个主模型 + 两个卫星模型，三者使用不同的等待窗口，通过迁移学习将卫星知识注入主模型。

| 模型 | 等待窗口 | 训练数据 | 机制 |
|------|---------|---------|------|
| 主模型 M_m | d_m（长） | D⁺_{τ,d_m} | 复制机制 |
| 特征卫星 M_f | d_f（短） | D_{τ,d_f} | 无复制机制 |
| 无偏卫星 M_u | 无 | D⁺_{τ,0} | 重要性采样（FNW） |

其中 **d_m > d_f**。

### 4.1 Satellites with Diverse Perspectives

主模型使用长窗口 d_m 训练，仍有两个缺陷：
1. 少量残留假负例
2. 最近 d_m 时间内的新样本被排除，无法感知最新趋势

因此引入两类卫星：
- **特征卫星 M_f**：短窗口 d_f，无复制机制，捕捉最新特征分布；不设隐层，让知识集中在 embedding 层
- **无偏卫星 M_u**：无等待窗口，用重要性采样（FNW）校正偏差，提供最新数据上的无偏估计

### 4.2 Embedding Transfer

用卫星在最新数据上学到的 embedding 来"刷新"主模型的特征表示。

三个模型对每个特征 xⁱ 各有独立 embedding，通过可训练线性权重融合：

```
ê_m^i = w_m^i · e_m^i + w_f^i · φ_f^i(x^i) + w_u^i · φ_u^i(x^i)
```

融合后的 ê_m 代替原始 e_m 送入主模型全连接层。

**设计细节**：
- 权重逐特征独立学习，不共用
- 卫星不设隐层，知识集中在 embedding 层
- 训练主模型时卫星 embedding 层冻结，防止被旧数据带偏

### 4.3 Strategy Aggregation

进一步利用卫星的输出 logit，与 Embedding Transfer 互补。

最终预测（Eq. 15）：

```
y_m = max( σ(o_m + o_u),  σ(o_f) )
    = max( p_m,           σ(o_f) )
```

- **o_u 直接加到 o_m**：无偏卫星估计可靠，直接修正主模型预测
- **o_f 仅作下界**：特征卫星无复制机制，系统性低估转化率，直接融合会引入负偏差；但当 σ(o_f) 较高时说明近期确有大量即时转化，此时以其为准，提升鲁棒性

损失函数用 p_m 而非 y_m 计算，保证主模型在 max 选到 σ(o_f) 时仍能收到梯度。

### 4.4 Processing Efficiency

- 训练延迟：假设卫星处理延迟 1 小时，主模型拿到的是 1 小时前的卫星 embedding；实验表明性能几乎不受影响，因为卫星相比主模型的长窗口仍是"新鲜"的
- 推理效率：卫星结构简单，推理时间仅为主模型的 63%，不影响在线服务

---

## 实验

### 数据集

| 数据集 | 特征数 | 转化数 | 样本数 | 平均CVR | 日志周期 | 归因窗口 |
|--------|--------|--------|--------|---------|---------|---------|
| Criteo | 17 | 3.6M | 15.9M | 22.8% | 60天 | 30天 |
| Tencent | 19 | 0.6M | 22.6M | 2.76% | 9天 | 5天 |

### 评估指标

- **AUC**：排序能力
- **NLL**：预测绝对值准确性
- **Bias**：按广告Campaign分组的预测偏差，工业场景关键指标

使用相对提升（RI）衡量，以 Pretrain（下界）= 0%，Oracle（上界）= 100% 归一化。

### 主实验结果（RQ1）

DFSN-α（优化AUC）在 Criteo/Tencent 上分别比最强 baseline 提升 RI-AUC **6.1% / 5.4%**。

DFSN-β（优化Bias）在所有指标上均优于所有 baseline，Criteo 上 RI-Bias 超过 80%。

### 消融实验（RQ3）

- 去掉任一卫星均有明显下降，两者互补
- 去掉特征卫星后 Bias 反而降低，说明更多卫星不一定对 Bias 有利
- Embedding Transfer 中线性变换优于直接拼接（DFSN_C）
- Strategy Aggregation 的贡献相对 Embedding Transfer 较小，但有增益

### 在线模拟（工业场景）

真实流量，10000+ 特征，归因窗口 7 天，主模型窗口 5 天，卫星延迟 1 小时：**CVR 提升 5.7%**。

---

## 关键思路总结

> 单个模型无法同时做到标签准确（长窗口）和数据新鲜（短窗口），因此设计主模型专注准确性，卫星模型专注时效性，通过 Embedding Transfer 和 Strategy Aggregation 将卫星知识迁移给主模型，两全其美。

---

## 备注

- Figure 3 中 d_m 和 d_f 的大小关系在图中表达有误（视觉上 d_f 看起来大于 d_m，与 caption "d_m > d_f" 矛盾）
- Feature Satellite 使用 D_{τ,d_f}（无 + 号，无复制机制），Figure 3 中未标出该数据集标签
