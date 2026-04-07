# LLM 模块设计选型

## Tied vs Untied Embedding

Transformer 有两个形状互为转置的矩阵：
- **Input Embedding** `[vocab_size, hidden_size]`：token ID → 向量
- **Output Projection / LM Head** `[hidden_size, vocab_size]`：隐藏向量 → 词表 logits

**Tied（共享）**：两者共用同一份参数，省内存，但输入侧"查字典"与输出侧"打分卡"语义不同，强迫共享会互相干扰。

**Untied（不共享，Qwen 的选择）**：各自独立优化，性能更好，代价是多出约 `vocab_size × hidden_size` 参数（Qwen-7B 约 +1.2 GB）。

---

## RMSNorm vs LayerNorm

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 公式 | 减均值 + 除标准差 + γ/β | 只除 RMS + γ（无 β） |
| 计算量 | 需算 μ 和 σ² | 省约 7–8% |
| 中心化 | 有 | 无 |
| 下游性能 | 基准 | 与 LN 相当 |

**核心假设**：re-centering（均值归零）对训练稳定性无必要，re-scaling（尺度归一）才是关键。实验验证两者性能持平，但 RMSNorm 更快——LLaMA、Qwen、Mistral 等现代大模型均采用。

---

## 标准 Attention vs Flash Attention

先补充一个背景，标准Attention有一个很大的瓶颈，源于 **Softmax 数值稳定保证**：实现中必须减去最大值 `m = max(s)`，保证 `exp(xᵢ - m) ∈ (0,1]` 永不溢出。（减均值无此保证，离群值仍可使 exp 溢出）。

 `S = QKᵀ`的计算时，`QKᵀ`本身是可以分块进行的，但分块之后为了算`max`需要等所有的分块计算完成再求，这样中间结果需要落到显存再读取，影响速度

**Flash Attention 核心： Online Softmax**

分块处理，每块完全在 SRAM（片上缓存）内计算，不把完整 seq² 写回 HBM。Online Softmax 用增量修正因子实现单遍扫描：
核心利用点，`S` 计算完成后，会乘以V再累加，这个时候减不同的`max`，对`V`的结果影响是线性的，所以每个tile扫描的时候，发现了更大的`max`，直接用新老`max`的差去更新历史上累计`V`就可以了，这样一次计算就可以获得`V`的结果了，运算完全在片上SRAM中完成。

```
扫到新 tile 时：
  m_new = max(m_old, tile_max)
  O = O * exp(m_old - m_new)     ← 修正历史输出
  d = d * exp(m_old - m_new)     ← 修正历史归一化因子
  O += exp(s - m_new) @ V        ← 加入新 tile 贡献
  d += sum(exp(s - m_new))
最终：O = O / d
```

| 方案 | HBM 读 K/V 次数 | 存储 seq² |
|------|------------------------|-----------|
| 标准 Attention | 1次 | 是 |
| Flash Attention | **1次** | **否** |

效果：seq=1K 时约 **3× 加速**，显存从 O(seq²) 降至 O(seq)，数学结果完全等价。

---

## DCA + YARN：长上下文外推

**问题**：RoPE 训练时只见过有限长度 `L_train`，推理遇到更长序列时位置角度进入未训练区域，attention 分数混乱。

**YARN**（两步修正）
1. **NTK-aware 插值**：修改 RoPE base 频率，把长序列位置"压缩"回训练范围，高频维度少插值、低频维度多插值（保留高频位置信息）
2. **熵缩放**：序列变长后 softmax 输入维度增大，attention 分数整体变小。乘以修正因子 `√(log n / log L_train)` 稳定 attention entropy

**DCA（Dual Chunk Attention）**：解决跨块位置过大的结构性问题

YARN 之后仍有问题：块内位置差准确，但跨 chunk 的相对位置（如 token 0 与 token 65000）极大，模型从未见过。DCA 把超长序列分块，分三类处理：

```
① 块内 attention：     用真实相对位置（精确）
② 相邻块 attention：   跨一个 chunk 边界，位置差在训练范围内（精确）
③ 远距离跨块 attention：改用 chunk 级别的粗粒度相对位置（数值始终合理）
```

**两者配合**：YARN 解决位置编码数值外推，DCA 解决跨 chunk 位置过大的结构问题。Qwen2 组合使用，支持 131K tokens 上下文且 PPL 几乎无退化。

---

## MHA vs GQA

**标准 MHA 的问题**

Decoder-only的模型在解码的时候，第N个token的K/V只依赖前N-1个Token的QKV，这样Prefill阶段的KV可以缓存下来，后续Decoder的时候直接使用。KV Cache是在推理的时候会占用比较多的KV资源，把KV的数量缩减能够有效降低推理的成本。

推理时需要缓存所有历史 K、V（KV Cache），大小为：
`2 × seq_len × n_heads × head_dim × layers`
72B 模型 n_heads=64，seq=32K 时 KV cache 可达数十 GB——推理显存的主要瓶颈。

**三种方案对比**

```
MHA（标准）:  Q Q Q Q | K K K K | V V V V    每个 Q head 对应独立 K/V
MQA（极端）:  Q Q Q Q | K       | V          所有 Q head 共享同一个 K/V（质量损失大）
GQA（折中）:  Q Q | KV    Q Q | KV           Q head 分组，组内共享 K/V
```

Qwen2-72B：64 Q heads + **8 KV heads** → KV Cache 节省 8×，质量接近 MHA。

**为什么 GQA 有效**：KV heads 数量到 8 左右时质量已接近完整 MHA；MQA（1个KV head）速度最快但质量明显下降。GQA 是推理效率与模型能力之间的工程折中。

---

## MoE（Mixture of Experts）

### 核心动机

Dense 模型每个 token 都经过全部参数，参数量与计算量强耦合。MoE 将 FFN 层替换为多个 Expert，每个 token 只激活少数 Expert，实现**参数量与计算量解耦**：总参数大（容量强），实际激活参数小（推理快）。

### 典型结构（以 Qwen3.5-122B-A10B 为例）

```
SparseMoeBlock
├── TopKRouter（门控）        # weight [256, 3072]，softmax → topk(8)
├── Experts（256个，参数合并）  # gate_up_proj [256, 2048, 3072]
│                             # down_proj     [256, 3072, 1024]
└── SharedExpert（常驻）      # 普通 SwiGLU MLP，每个 token 都经过
    └── shared_expert_gate    # sigmoid 门控 shared expert 的贡献
```

所有 Expert 权重**合并为 3D tensor**（非独立 nn.Module），节省内存碎片，推理时只 index 激活的 Expert 切片。

### 参数量与激活量

| 模块 | 总参数 | 每 token 激活 |
|------|--------|---------------|
| Embedding + lm_head | 1,526M | 全激活 |
| Full Attention ×12 | 944M | 全激活 |
| Linear Attention ×36 | 3,187M | 全激活 |
| MoE FFN ×48（Expert 部分） | 115,962M | 8/256 ≈ 3.1% |
| MoE FFN ×48（Router+SharedExpert） | 493M | 全激活 |
| **合计** | **≈ 122B** | **≈ 10B（A10B）** |

> 推算：Active = 非MoE(~6.1B) + 激活Expert(48层 × 8/256 × 2,416M ≈ 3.6B) ≈ 9.7B ≈ 10B ✓

### 负载均衡损失

Router 若不加约束，所有 token 会集中路由到少数"强" Expert，退化为 Dense。
训练时加 **Auxiliary Loss**（Switch Transformer 提出）：

```
L_aux = num_experts × Σ_i (tokens_per_expert_i × router_prob_i)
```

惩罚分布不均匀，系数 `router_aux_loss_coef = 0.001`，与主 loss 加权相加。

### 与 Dense FFN 的切换

`mlp_only_layers` 控制哪些层强制用 Dense FFN。Qwen3.5 中该字段为空列表，**全部 48 层均使用 SparseMoeBlock**。
