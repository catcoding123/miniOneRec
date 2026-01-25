# MiniOneRec 学习笔记

## 📌 核心理解

### 什么是MiniOneRec？
- **目标**：将推荐问题转换为语言生成问题
- **核心创新**：物品 → 语义ID（SID）→ 用LLM生成推荐

### 三阶段流程
```
1. SID构建：物品文本 → Embedding → RQ量化 → SID token
   例：iPhone 14 → [768维向量] → RQ-VAE → <10_45_2>

2. SFT训练：多任务学习
   - 任务1：SID序列 → 下一个SID（推荐能力）
   - 任务2：SID ⇄ 标题（语义对齐）
   - 任务3：SID序列 → 物品描述（融合生成）

3. RL优化：GRPO算法
   - 生成多个候选
   - 计算reward（命中+排序惩罚）
   - 组内归一化
   - 策略梯度更新
```

## 🔑 关键技术点

### 1. SID是什么？
**Semantic ID（语义ID）**：将物品压缩成一个特殊token

**为什么需要SID？**
- ❌ 传统ID（item_12345）无语义信息，模型无法理解
- ✅ SID保留语义，相似物品有相似的SID
- ✅ 可以用LLM的next-token prediction来做推荐

**SID格式**：`<layer1_layer2_layer3>`
- 例如：`<10_45_2>` 表示3层RQ量化的结果
- 每层256个码本 → 总容量 256³ = 16,777,216 个物品

### 2. RQ-VAE原理
```
物品文本 → Text Encoder → 768维向量
         → MLP压缩 → 64维向量
         → RQ量化 → [idx1, idx2, idx3]
         → MLP解码 → 重建768维向量

Loss = 重建误差 + 量化损失
```

**3层残差量化**：
```python
# 第1层：找最接近的码本向量
z1 = codebook1[idx1]  # 粗粒度表示
residual1 = embedding - z1

# 第2层：对残差量化
z2 = codebook2[idx2]  # 细节修正
residual2 = residual1 - z2

# 第3层：对残差再量化
z3 = codebook3[idx3]  # 更细节修正

# 最终表示
reconstructed = z1 + z2 + z3
```

### 3. 三个训练任务详解

#### 任务1：序列推荐 (SidSFTDataset)
```
输入：用户历史 <10_45_2>, <23_67_8>, <45_12_90>
输出：<78_34_56>
目的：学习推荐模式（协同过滤）
```

#### 任务2：SID-标题对齐 (SidItemFeatDataset)
```
方向1 - Title → SID:
  输入：Which item has the title: iPhone 14 Pro?
  输出：<10_45_2>

方向2 - SID → Title:
  输入：What is the title of item "<10_45_2>"?
  输出：iPhone 14 Pro

目的：让模型理解SID的含义（建立"词典"）
```

#### 任务3：融合序列 (FusionSeqRecDataset)
```
输入：用户历史 <10_45_2>, <23_67_8>, <45_12_90>
输出：MacBook Pro - 专业笔记本电脑

目的：结合推荐+生成能力，产生可解释的推荐
```

**三个任务的关系**：
- 任务1：教模型**会推荐**（模式匹配）
- 任务2：教模型**懂语义**（认识SID）
- 任务3：教模型**能解释**（生成+推理）

### 4. RL阶段 - GRPO算法

**核心思想**：生成多个候选，基于reward优化

```python
# 步骤1：生成K个候选（K=4或8）
for prompt in batch:
    candidates = model.generate(num_beams=K)

# 步骤2：计算reward
for candidate in candidates:
    r = 1 if hit else 0  # 命中奖励
    if not hit:
        r -= alpha * log(prob)  # 惩罚高概率错误
    r += beta * cf_score  # 协同过滤加分

# 步骤3：组内归一化（GRPO关键）
normalized_r = (r - mean(r)) / std(r)

# 步骤4：策略梯度
loss = -log_prob * normalized_r + kl_penalty
```

**为什么需要组内归一化？**
- 稳定训练（避免reward尺度问题）
- 强调相对优劣（第1名vs第8名）
- 减少方差（梯度更稳定）

### 5. Constrained Decoding

**问题**：如果随意生成，可能产生无效SID（如 `<999_888_777>`）

**解决**：用哈希表限制每步只能生成有效token

```python
# 构建前缀 → 有效后续token的映射
hash_map = {
    "<": [0, 1, ..., 255],           # 第1层所有可能
    "<10_": [3, 45, 78, ...],        # 第2层在第1层=10时的可能
    "<10_45_": [2, 19, 234],         # 第3层的可能
}

# 生成时每步查表
def get_valid_tokens(prefix):
    return hash_map[prefix]
```

**效果**：保证100%生成有效SID（CC指标=0）

## 📊 数据格式

### train.csv
```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
U001,"['iPhone 14', 'AirPods']",MacBook Pro,"['I001', 'I002']",I003,"['<10_45_2>', '<23_67_8>']",<78_34_56>
```

### .index.json (SID映射表)
```json
{
  "I001": ["<10_45_2>"],
  "I002": ["<23_67_8>"],
  "I003": ["<78_34_56>"]
}
```

### .item.json (物品特征)
```json
{
  "I001": {
    "title": "iPhone 14 Pro Max",
    "description": "6.7-inch display...",
    "category": "Electronics"
  }
}
```

## 🚀 快速开始

### 环境准备
```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
pip install -r requirements.txt
```

### 使用现有数据快速运行

**数据已准备好**：
- Industrial_and_Scientific: 36,260条训练数据
- Office_Products: 38,925条训练数据

**1. SFT训练（监督微调）**
```bash
bash sft.sh
```

**2. RL优化（强化学习）**
```bash
bash rl.sh
```

**3. 评估**
```bash
bash evaluate.sh
```

## 💡 学习路线

### Day 1-2: 理解核心概念
- [x] 为什么要把物品变成SID？
- [ ] RQ-VAE如何量化？画一个流程图
- [ ] 生成式推荐 vs 传统推荐的区别

### Day 3-4: 数据处理
- [ ] 理解三个数据集的构建逻辑
- [ ] 查看真实样本：`learn/view_samples.py`
- [ ] 理解label masking机制（为什么用-100？）

### Day 5-7: SFT训练
- [ ] 运行demo_sft.sh
- [ ] 查看训练日志，理解loss变化
- [ ] 可视化SID embeddings

### Day 8-10: RL优化
- [ ] 理解GRPO vs PPO的区别
- [ ] 实现自定义reward函数
- [ ] 对比SFT和RL的效果

## 🔧 调试技巧

### 可视化训练样本
```python
from data import SidSFTDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
dataset = SidSFTDataset(...)

# 查看第一个样本
sample = dataset[0]
print("Input IDs:", sample["input_ids"])
print("Labels:", sample["labels"])
print("Decoded:", tokenizer.decode(sample["input_ids"]))
```

### 检查SID有效性
```python
import json

with open(".index.json") as f:
    sid_map = json.load(f)

print(f"Total items: {len(sid_map)}")
print(f"Sample SIDs: {list(sid_map.values())[:5]}")

# 检查碰撞
all_sids = [sid[0] for sid in sid_map.values()]
print(f"Unique SIDs: {len(set(all_sids))}")
print(f"Collision rate: {1 - len(set(all_sids)) / len(all_sids)}")
```

## 📈 实验记录

### 实验模板
```markdown
## 实验 #001: [简短描述]

**日期**：2026-01-23
**目标**：测试Industrial数据集的SFT训练

**配置**：
- 数据集：Industrial_and_Scientific
- 模型：Qwen2.5-1.5B-Instruct
- Batch size：128
- Learning rate：3e-4
- Epochs：10

**结果**：
| 指标 | 值 |
|------|-----|
| HR@10 | 0.452 |
| NDCG@10 | 0.245 |
| CC | 0 ✓ |

**发现**：
- 训练收敛快，loss在第3个epoch就稳定
- 任务2（对齐）的loss下降最快

**下一步**：
- 尝试RL优化
- 对比不同学习率
```

---

## 🎯 核心要点总结

### 技术创新点
1. **SID构建**：RQ-VAE将物品压缩成离散token
2. **多任务SFT**：推荐+对齐+生成，三位一体
3. **GRPO**：组相对策略优化，稳定RL训练
4. **Constrained Decoding**：保证生成有效SID

### 为什么有效？
1. **继承LLM能力**：利用预训练语言模型的世界知识
2. **语义对齐**：SID不是随机编码，而是有语义的
3. **可解释性**：可以生成自然语言推荐理由
4. **统一框架**：推荐、搜索、对话都是生成任务


2、note.md会帮我精简的记录笔记，方便我回顾，比如核心思路、形象比如（如果有）、逻辑
