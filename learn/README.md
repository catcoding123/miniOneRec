# MiniOneRec 学习指南

欢迎学习MiniOneRec！这个目录包含了帮助你理解和运行项目的所有资源。

## 📚 学习资源

### 1. 文档
- **`claude.md`**: 核心概念精简笔记（⭐推荐首先阅读）
- **`../MiniOneRec_Code_Analysis.md`**: 完整代码分析文档
- **`../Training_Sample_Organization.md`**: 训练样本详解

### 2. 工具脚本
- **`prepare_mini_data.py`**: 准备小规模demo数据集
- **`view_samples.py`**: 查看和理解训练样本

### 3. 分步运行脚本
按顺序执行，逐步理解整个流程：

```bash
bash learn/step1_prepare_data.sh      # 准备mini数据集
bash learn/step2_view_data.sh         # 查看数据样本
bash learn/step3_sft_train.sh         # SFT训练（需要GPU）
bash learn/step4_sft_evaluate.sh      # 评估SFT模型
bash learn/step5_rl_train.sh          # RL训练（需要GPU）
bash learn/step6_rl_evaluate.sh       # 评估RL模型
```

## 🚀 快速开始

### 步骤0：环境检查
```bash
# 确保在正确的conda环境中
conda activate MiniOneRec

# 检查依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 步骤1：准备数据（无需GPU，5分钟）
```bash
bash learn/step1_prepare_data.sh
```

这会从完整数据集中采样1000条数据，创建一个mini数据集用于快速学习。

**输出**：`./learn/mini_data/` 目录，包含：
- `train.csv`: 训练数据
- `test.csv`: 测试数据
- `valid.csv`: 验证数据
- `*.index.json`: SID映射表
- `*.item.json`: 物品特征

### 步骤2：理解数据（无需GPU，2分钟）
```bash
bash learn/step2_view_data.sh
```

这会展示：
- 训练数据的格式
- 三个训练任务如何构建
- Label masking机制
- SID的含义

**关键理解**：
- 任务1：SID序列 → 下一个SID（推荐能力）
- 任务2：SID ⇄ 标题（语义对齐）
- 任务3：SID序列 → 物品描述（融合生成）

### 步骤3：SFT训练（需要GPU，~10分钟）
```bash
# 使用Hugging Face模型（会自动下载）
bash learn/step3_sft_train.sh Qwen/Qwen2.5-1.5B-Instruct

# 或使用本地模型
bash learn/step3_sft_train.sh /path/to/your/model
```

**注意**：
- 需要GPU（建议至少8GB显存）
- 使用mini数据集训练很快（~10分钟）
- 可以通过修改脚本中的参数调整训练配置

**监控训练**：
```bash
# 查看训练日志
tail -f ./learn/output/sft/train.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 步骤4：评估SFT（需要GPU，~2分钟）
```bash
bash learn/step4_sft_evaluate.sh
```

**关键指标**：
- **HR@10**: 命中率（例如0.45表示45%的测试样本在Top-10中找到了正确物品）
- **NDCG@10**: 考虑排名位置的指标（越高越好）
- **CC**: 约束检查（应该为0，表示所有生成的SID都有效）

### 步骤5：RL训练（需要GPU，~20分钟）
```bash
bash learn/step5_rl_train.sh
```

**说明**：
- RL训练会生成多个候选，速度较慢
- 使用GRPO算法优化推荐指标
- 通常能在SFT基础上提升5-10%

### 步骤6：评估RL并对比（需要GPU，~2分钟）
```bash
bash learn/step6_rl_evaluate.sh
```

**对比SFT vs RL**：
- HR@10提升了多少？
- NDCG@10提升了多少？
- RL为什么更好？（直接优化推荐指标）

## 💡 理解核心概念

### 什么是SID？
**Semantic ID（语义ID）**：将物品压缩成一个特殊token

例如：`iPhone 14 Pro` → `<10_45_2>`

**格式**：`<layer1_layer2_layer3>`
- 3层RQ量化的结果
- 每层256个码本
- 总容量：256³ = 16,777,216个物品

### 为什么需要SID？
传统推荐系统使用物品ID（如`item_12345`），但这些ID没有语义信息。

SID的优势：
- ✅ 保留语义（相似物品有相似的SID）
- ✅ 可以用LLM做推荐（生成SID = 生成推荐）
- ✅ 冷启动友好（新物品可以直接编码）

### 三个训练任务的关系

```
任务1（基础推荐）     任务2（语义对齐）     任务3（融合生成）
      ↓                    ↓                    ↓
  学会推荐模式          理解SID含义          生成可解释推荐
      ↓                    ↓                    ↓
    协同过滤            建立词典              推理+生成
```

**类比**：
- 任务1 = 学会下围棋（模式匹配）
- 任务2 = 认识棋谱符号（理解语言）
- 任务3 = 解释为什么这样下（推理+表达）

### RL阶段做什么？

**GRPO算法**：
1. 生成多个候选（K=4或8）
2. 计算reward（命中=1，未命中=0，加排序惩罚）
3. 组内归一化（让相对优劣更明显）
4. 策略梯度更新（奖励好的，惩罚差的）

**为什么有效？**
- SFT只是模仿训练数据
- RL直接优化推荐指标（HR、NDCG）
- 通常能提升5-10%

## 🔧 常见问题

### Q1: 没有GPU怎么办？
**A**: 可以完成步骤1-2（准备和查看数据），理解核心概念。训练部分需要GPU。

### Q2: 显存不足怎么办？
**A**: 修改训练脚本中的参数：
```bash
# 在step3_sft_train.sh中
BATCH_SIZE=4           # 减小batch size
MICRO_BATCH_SIZE=2     # 减小micro batch size
MAX_LEN=256            # 减小序列长度
```

### Q3: 训练太慢怎么办？
**A**:
- 减少训练数据：`bash learn/step1_prepare_data.sh Industrial_and_Scientific 500`
- 减少训练轮数：修改`step3_sft_train.sh`中的`NUM_EPOCHS`
- 使用更小的模型：如`Qwen2.5-0.5B`

### Q4: 如何理解评估指标？
**A**:
- **HR@10 = 0.45**: 45%的测试用户在Top-10推荐中找到了他们实际购买的物品
- **NDCG@10 = 0.25**: 考虑排名位置的指标，越高越好
- **CC = 0**: 所有生成的SID都是有效的（100%有效率）

### Q5: mini数据集的效果能代表真实效果吗？
**A**: 不完全能。mini数据集主要用于：
- 快速理解流程
- 调试代码
- 验证想法

真实效果需要使用完整数据集训练。

## 📝 实验记录模板

在`claude.md`中记录你的实验：

```markdown
### 实验 #001: [日期] [描述]

**配置**：
- 数据：1000条
- 模型：Qwen2.5-1.5B
- Epochs：3

**结果**：
| 模型 | HR@10 | NDCG@10 |
|------|-------|---------|
| SFT  | 0.45  | 0.25    |
| RL   | 0.48  | 0.27    |

**发现**：
- RL比SFT提升了6.7%
- 训练收敛很快

**下一步**：
- 尝试更大的数据集
- 调整RL的reward权重
```

## 🎯 学习路径

### 新手路径（2-3天）
1. ✅ 阅读`claude.md`，理解核心概念
2. ✅ 运行步骤1-2，查看数据
3. ✅ 理解三个训练任务
4. ✅ （有GPU）运行步骤3-4，体验SFT
5. ✅ 分析结果，理解指标

### 进阶路径（1-2周）
1. ✅ 完成新手路径
2. ✅ 运行步骤5-6，体验RL
3. ✅ 对比SFT vs RL
4. ✅ 阅读完整代码分析文档
5. ✅ 修改训练参数做实验
6. ✅ 使用完整数据集训练

### 高级路径（持续）
1. ✅ 完成进阶路径
2. ✅ 实现自定义reward函数
3. ✅ 尝试不同的SID构建方法（RQ-Kmeans+）
4. ✅ 跨数据集测试
5. ✅ 贡献代码改进

## 🔗 相关资源

- **论文**: [MiniOneRec Paper](https://arxiv.org/abs/2510.24431)
- **GitHub**: [MiniOneRec](https://github.com/AkaliKong/MiniOneRec)
- **Hugging Face**: [模型下载](https://huggingface.co/kkknight/MiniOneRec)

---

**祝学习顺利！有问题可以在Issue中讨论。** 🚀
