# LLM 技术报告索引

## 模型系列

- **千问系列（Qwen）** → [model_qwen.md](model_qwen.md)

---

## 模型训练


### 模型结构

**架构**

- **Embedding**：Untied（input embedding 与 LM Head 不共享权重）
- **位置编码**：RoPE，inverse frequency matrix 用 FP32 精度
   - **Dynamic NTK-aware interpolation**：动态调整 RoPE base，按 chunk 缩放避免性能骤降
   -  **LogN-Scaling**：attention dot product 乘以 log(当前长度 / 训练长度)，稳定 attention entropy
   - **Layer-wise Window Attention**：低层短窗口、高层长窗口（低层对长度更敏感）

- **Bias**：大部分层去掉; QKV 层在Qwen3之前保留（增强长度外推）, Qwen3之后已移除
- **归一化**：Pre-Norm + RMSNorm，Qwen3增加QK-Norm，防止点积过大导致 softmax 饱和，稳定大模型训练
- **激活函数**：SwiGLU，FFN 维度为 hidden size 的 8/3 倍
- **上下文长度扩展**（推理阶段免训练，三技术叠加）

- **MoE 模型**
   - Shared Experts：Qwen之前存在，Qwen3之后  移除 
   - Load Balancing Loss：改为 global-batch 级别，鼓励 expert 专业化 


### 预训练

#### 数据处理

|  | Qwen1 | Qwen2.5 | Qwen3 |
|------|-------|---------|-------|
| **数据规模** | ~3T | 18T | 36T |
| **语言数** | 主要中英 | 29 种 | 119 种 |
| **PDF 提取** | — | — | Qwen2.5-VL OCR + Qwen2.5 精炼 |
| **质量过滤** | 规则 + 模型（LM 打分、质量模型、有害内容模型）+ 人工抽查 | Qwen2-Instruct 多维打分，多语言过滤增强 | 自研多语言标注系统（教育价值、领域、安全等多维度），覆盖 30T+ tokens |
| **合成数据** | — | Qwen2-72B + Math-72B 生成，RM-72B 过滤 | Qwen2.5 / Math / Coder 生成教材、问答、代码片段，量达万亿级 |
| **数据配比** | 上采样高质量来源 | 域级分类，下采样娱乐/电商，上采样技术/学术 | **样本级**消融实验，用小代理模型确定最优混合比 |
| **去重** | 精确（exact-match）+ 模糊（MinHash + LSH） | 未披露 | 未披露 |
| **污染过滤** | 删除与评测集 13-gram 重叠样本 | 未披露 | 未披露 |



#### 训练流程

- 任务：自回归语言建模（next-token prediction），上下文长度 2048（Qwen1） –> 4096（Qwen3）

- 多阶段预训练：
   - **S1 通用**：30T tokens，序列长度 4,096，建立语言和世界知识基础
   - **S2 推理**：5T tokens，序列长度 4,096，增加 STEM/代码/合成数据比例
   - **S3 长文本**：100B tokens，序列长度扩展至 32,768，引入 YARN + DCA 支持长上下文

- 参数选择
   - 优化器：AdamW，β₁=0.9，β₂=0.95，ε=10⁻⁸，cosine LR schedule（最低衰减至峰值 10%）
   - 精度：BFloat16 混合精度
   - 寻参（Qwen2.5 §3.2）:用 Scaling Law 确定 LR 和 batch size，将最优超参数拟合为模型规模 N 与数据量 D 的函数，避免对每个规模重复搜索。



### 后训练

#### 训练流程


- **Stage 1 — Long-CoT Cold Start**（SFT）：数学/代码/逻辑推理，每条带验证答案，用 QwQ-32B 生成后严格过滤；训练步数刻意少，为 RL 保留探索空间
- **Stage 2 — Reasoning RL**（GRPO）：3,995 query-verifier pairs，比 cold-start 更难；控制 entropy 保持训练稳定
- **Stage 3 — Thinking Mode Fusion**（continual SFT）：thinking 数据（Stage 2 模型 rejection sampling 生成）+ non-thinking 数据（代码/数学/指令/多语言/创意写作/角色扮演）；融合双模式，引入 /think 和 /no_think chat template；副产品：自然涌现 Thinking Budget 能力
- **Stage 4 — General RL**：reward 覆盖 20+ 任务，三类 reward（rule-based / model-based with ref / model-based without ref）；全面提升指令跟随、格式、偏好对齐、Agent、RAG 等能力

**蒸馏：**

- **Strong-to-Weak Distillation**：Off-policy（蒸馏旗舰 thinking + non-thinking 输出）→ On-policy（学生自采样后对齐 teacher logits，最小化 KL 散度）；仅需旗舰四阶段 1/10 GPU 时间

#### SFT 训练超参（Qwen2.5 §4.1）

- 数据量：100万+ 样本
- Epochs：2
- 序列长度：32,768 tokens
- LR：7×10⁻⁶ → 7×10⁻⁷（cosine decay）
- Weight decay：0.1
- 梯度裁剪：1.0


## Coder能力
### 代码数据 (Qwen Code 2.5)


### 预训练

#### 数据来源（5类）：

| 类型 | 来源 | 处理方式 |
|------|------|---------|
| Source Code | GitHub 公开仓库（92种语言，2024年2月前）+ PR/Commits/Jupyter/Kaggle | 参考 StarCoder2/DS-Coder 的规则过滤 |
| Text-Code Grounding | 文本与代码关联数据 | 自研分层过滤（见下） |
| Synthetic Data | 由 CodeQwen1.5 生成 | 代码执行器验证，只保留能跑通的代码 |
| Math Data | Qwen2.5-Math 预训练语料 | 直接复用 |
| Text Data | Qwen2.5 通用语料 | 去除代码段防止重叠 |

**分层过滤**（针对 Text-Code Grounding）：4 阶段递进，fastText 打分，自然形成质量分层；HumanEval+MBPP 从 41.6% → 46.8%


#### 数据流程
从零开始在纯代码上预训练，而是以 Qwen 基础模型为起点继续预训练——避免丢失通用语言能力（Code LLaMA 的教训），大概用了90B tokens 代码数据。代码训练的时候数据配比：Code 70% + Text 20% + Math 10%，共 **5.2T tokens** (Qwen-Coder 2.5)

### 后训练

#### 数据生成
**1. 编程语言识别与过滤**
微调 CodeBERT 对近 100 种语言分类，保留主流语言；删除不含代码的样本

**2. 从 GitHub 代码合成指令**（Self-Instruct 变体）
GitHub 代码片段（≤1024 tokens）→ 通用 LLM 反推 instruction → Code LLM 生成 response → LLM scorer 过滤

**3. 多语言多 Agent 协作框架**
解决低资源语言指令数据稀少问题：
- 每种语言一个 Agent + Memory Bank（避免重复生成）
- Agent 间跨语言知识蒸馏，高资源语言经验迁移到低资源语言
- 动态识别能力缺口，主动补充对应指令

**4. Checklist 质量评分**：9 个维度加权打分（正确性、可读性、注释、教育价值等）

**5. 多语言沙箱验证**：单元测试 + 语法检查 + 隔离执行，确保合成代码可运行

#### 训练策略（§4.2）

**1. 由粗到细 SFT**（标准指令格式）
- Stage 1：数千万条低质量但多样的指令数据，粗调覆盖广度
- Stage 2：数百万条高质量指令数据 + Rejection Sampling 精调

**2. 混合调优**（保长上下文能力）
- 大部分标准 SFT + 少量 FIM 格式数据
- FIM 数据用 tree-sitter 解析代码，提取不同层级逻辑块作为 infill 目标
- 具体混合比例及是否同步训练论文未披露

**3. 代码 DPO 对齐**
- 执行反馈：沙箱跑代码，以实际运行结果判断 chosen/rejected
- LLM-as-Judge：复杂代码片段用大模型打分
- 代码 DPO 数据 + 通用领域数据混合训练


## 技术选型

→ 详见 [block.md](block.md)（Embedding、RMSNorm、Flash Attention、YARN/DCA、GQA、MoE）

---

## 关键趋势

- **参数规模**：从 7B/72B 到 235B MoE，兼顾端侧（0.5B）与云端旗舰
- **数据规模**：Qwen1 约 3T -> Qwen2 约7T → Qwen2.5 扩至 18T → Qwen3 达 36T tokens
- **多语言**：Qwen2.5 支持 29 种语言，Qwen3 扩展至 119 种，Qwen3.5 达 201 种
- **推理能力**：Qwen3 将思考/非思考模式统一，QwQ 走强化学习专用推理路线
- **原生多模态**：Qwen3.5 起放弃"语言模型 + 视觉适配器"范式，改为预训练阶段即融合多模态 tokens
- **架构创新**：Qwen3.5 引入 Gated Delta Networks（线性注意力）替代标准 Attention，与稀疏 MoE 结合大幅提升推理效率
- **许可证**：Qwen3/3.5 系列全部采用 Apache 2.0
