# 推荐系统技术索引

## 模型研究方向

### 1. 信号引入
如何引入更丰富、更有效的输入信号。
- 稀疏 ID 特征 embedding 扩展
- 用户行为序列
- 多模态信号（图文、视频）
- 跨域/跨场景信号迁移

### 2. 特征交叉与序列建模
如何建模特征间的协同效应及用户行为序列。
- **特征交叉** → [feature_interaction.md](feature_interaction.md)
- 序列建模（Target Attention、SIM、长序列压缩）

### 3. 主网络结构
排序主干网络的架构设计与扩展。
- MLP / Cross Network（DCN）
- Transformer-based（Hiformer、Zenith）
- MoE 扩展（参数扩容不增加推断开销）
- Scaling Law in Recommendation

### 4. 目标建模
如何设计预测目标以更好对齐业务指标。
- 多任务学习（MTL）
- ESMM / PEPNet 等样本偏差修正
- 时长/质量目标建模
- 因果推断去偏
