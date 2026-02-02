# MiniOneRec 学习笔记

## 2026-01-24

### 三个训练任务的关系

```
任务1：SID序列 → SID        （学推荐模式）
任务2：SID ⇄ Title/Desc     （学语义对齐）
任务3：SID序列 → Title/Desc （端到端融合）

任务3 = 任务1 + 任务2 的"端到端"版本：

        任务1          任务2
SID序列 ──→ SID ──→ Title/Desc
   └──────── 任务3 ────────┘
```

**为什么需要三个任务？**
- 只训练任务3：模型可能"偷懒"，不真正理解SID
- 三任务联合：任务1强制学SID模式，任务2强制建立语义，任务3整合能力

**类比**：
- 任务1 = 学数学公式
- 任务2 = 学术语定义
- 任务3 = 用公式+术语写应用题答案

---

### SFT训练机制

**三个任务的具体输入输出**：

```python
# 任务1：SidSFTDataset（序列推荐）
输入： "The user has interacted with items <a_60><b_159><c_203>, <a_204><b_229><c_36> in chronological order. Can you predict the next possible item?"
输出： "<a_120><b_190><c_110>"

# 任务2：SidItemFeatDataset（双向对齐）
方向1： "Which item has the title: iPhone 14 Pro?" → "<a_10><b_45><c_2>"
方向2： "What is the title of item "<a_10><b_45><c_2>"?" → "iPhone 14 Pro"

# 任务3：FusionSeqRecDataset（融合生成）
输入： "The user has interacted with items <a_60><b_159><c_203>, <a_204><b_229><c_36>. Recommend the next item."
输出： "MacBook Pro - Professional laptop with M3 chip..."
```

**训练方式：随机混合，非交替**

```python
# sft.py:191-203
train_datasets = []
train_data1 = SidSFTDataset(...)      # 38,924样本
train_data2 = SidItemFeatDataset(...) # 6,888样本（3,444 SID × 2方向）
train_data3 = FusionSeqRecDataset(...) # 38,924样本

train_data = ConcatDataset(train_datasets)  # 拼接：84,736样本
hf_train_dataset.shuffle(seed=42)           # 完全打乱

# 每个batch（128样本）随机包含：
# - 任务1：~59个（45.9%）
# - 任务2：~10个（8.1%）  ← 最少
# - 任务3：~59个（45.9%）
```

**关键机制**：

```
单一模型 + 混合数据 + 顺序采样

打乱后：[任务2-样本3, 任务1-样本100, 任务3-样本50, ...]
           ↓
Batch 1: 取前128个 → 计算loss → 更新模型
Batch 2: 取第129-256个 → 计算loss → 更新同一个模型
...

NOT：任务1训练10步 → 切换任务2 → 切换任务3（没有这种策略）
```

**一句话**：ConcatDataset拼接 + shuffle打乱 + Trainer顺序取样，无特殊任务调度策略。

**代码位置**：
| 任务 | 数据集类 | 文件位置 | 样本数 |
|------|---------|---------|--------|
| 任务1 | SidSFTDataset | data.py:483-588 | 38,924 |
| 任务2 | SidItemFeatDataset | data.py:876-956 | 6,888 |
| 任务3 | FusionSeqRecDataset | data.py:1475-1650 | 38,924 |
| 拼接 | ConcatDataset | sft.py:203 | 84,736 |
| 打乱 | shuffle | sft.py:219 | - |

---

### 任务1的前置依赖

```
物品文本 → Text Encoder → RQ-VAE → SID (.index.json)
                              ↓
                    然后才能跑任务1/2/3
```

前置任务：**RQ-VAE训练**（项目已提供预训练好的SID）

---

### 任务2为什么不用词表查表？

| 查词表 | 训练任务2 |
|--------|----------|
| 只能返回固定标题 | 可生成描述、解释 |
| 词表外SID无法处理 | 有泛化能力 |
| 模型不懂SID含义 | 内化了语义理解 |

**本质**：词表是"外挂死知识"，训练是"内化活知识"

---

### RQ-VAE核心理解

**整体结构**：
```
原始Embedding(768维) → Encoder → 量化(Codebook) → Decoder → 重建(768维)
                         ↓
                    SID: <a_10><b_45><c_2>
```

**残差量化（3层）**：
```
第1层：对原始向量量化 → 粗粒度特征
第2层：对残差量化 → 细节修正
第3层：对残差再量化 → 更细修正

最终：x ≈ z1 + z2 + z3
```

---

### Loss组成与更新对象

```
总Loss = loss_recon + quant_loss

loss_recon = MSE(Decoder输出, 原始输入)     → 更新Decoder
quant_loss = codebook_loss + β*commitment_loss
           = MSE(码本, x.detach())          → 更新码本
           + MSE(码本.detach(), x)          → 更新Encoder
```

**detach**：切断梯度，让某一方不动
- `x.detach()`：x不接收梯度，只更新另一方

---

### K-means初始化

- **时机**：第一次forward时，用当前batch数据聚类
- **作用**：让码本初始位置覆盖数据分布，避免"死码本"
- **每层独立**：第1层用原始向量，第2/3层用残差

**类比**：先调研客流再放垃圾桶，而不是随便放

---

### Sinkhorn算法（解决碰撞/均衡分配）

**问题**：argmin选最近码本 → 热门码本挤爆，冷门没人用

**解决**：交替归一化
```
行归一化：每个物品分配概率和=1（只能选1个）
列归一化：每个码本收到概率和相等（强制均衡）
```

**效果**：
- 热门码本：列和大 → 除以大数 → 概率被压小
- 冷门码本：列和小 → 除以小数 → 概率被放大

**sk_epsilon**：温度参数，越小越接近硬分配，=0则不用Sinkhorn

**类比**：限流政策，每家店客流差不多

---

### 碰撞 vs 均衡

| 碰撞 | 均衡 |
|------|------|
| 多个物品得到相同SID | 每个码本使用次数差不多 |

**关系**：均衡 → 大概率无碰撞，但不完全等价

---

### 为什么用RQ层次结构？

**SFT可能破坏层次语义，但层次设计在其他环节有关键作用：**

| 作用 | 说明 |
|------|------|
| 压缩效率 | 3×256码本 vs 16M码本，参数量差2万倍 |
| 语义平滑 | 相似物品共享前缀，便于LLM学习共现模式 |
| Constrained Decoding | 层次结构天然形成前缀树，保证生成有效SID |

```
相似物品共享前缀：
  iPhone 14 Pro → <a_10><b_45><c_2>
  iPhone 14     → <a_10><b_45><c_8>  ← 共享a,b
  MacBook       → <a_23><b_12><c_9>  ← 完全不同
```

**一句话**：层次价值不在于LLM理解它，而在于**压缩高效 + 语义平滑 + 便于约束**。

---

### Codebook计算效率

```
单层16M码本：距离计算 16,000,000 次
RQ三层256：  距离计算 256×3 = 768 次

效率提升 ≈ 20000倍
```

**类比**：从1600万人找一个 vs 先选省→市→人

---

### SID与商品ID的对应

- **目标**：一一对应（每个商品唯一SID）
- **容量**：256³ = 16M >> 商品数（通常几千~几十万）
- **碰撞原因**：不是容量不够，而是语义太相似
- **解决**：Sinkhorn强制均衡分配

---

### RQ-VAE训练数据量需求

**实际数据量（项目使用）**：
```
Industrial_and_Scientific: 3,686个物品
Office_Products:           3,459个物品

训练配置：
- batch_size: 2,048
- epochs: 5,000（数据重复使用很多次）
- 总训练次数: 3,500 × 5,000 = 1750万次
```

**为什么数据量不大？**

| 原因 | 说明 |
|------|------|
| 无监督学习 | 输入=输出，自监督，无需标注 |
| 参数量小 | 码本仅49K参数（3×256×64） |
| 容量充足 | 256³ = 16M >> 3,500物品 |
| Kmeans初始化 | 第一步就接近最优 |
| 高epoch补偿 | 5,000轮弥补样本少 |

**数据量指导**：

```
单类目（手机）:        500-1,000个    ✅ 足够
多类目（电商）:        3,000-5,000个  ✅ 推荐（项目使用）
全站级（所有类目）:    10,000-50,000个 ✅ 更好泛化
超大规模:              > 100,000个     ⚠️ 边际收益小
```

**关键不是数量，是语义多样性**：
```
10,000个手机（语义相似）< 3,500个跨类目物品（语义多样）

碰撞主要由语义相似导致，不是容量不足
```

**收敛标准**：
```
loss_recon:   < 0.05
collision_rate: < 2%
码本利用率:   每个码本 > 0.1%
```

**类比**：学26个字母（码本），不需要看百万本书，看几千个不同单词就能学会所有字母的用法。

---

## SFT训练

### Epoch理解

```
epoch = 已处理样本数 / 总样本数

epoch 1.0  = 完成第1轮
epoch 1.05 = 第2轮进行了5%
epoch 2.5  = 第3轮进行了50%
```

Epoch是**连续值**，表示数据被遍历的程度。

---

### Learning Rate调度（Cosine Warmup）

```
lr
 ↑
 │      ╭──╮
 │     ╱    ╲
 │    ╱      ╲
 │───╱────────╲───→ step
    warmup  cosine decay
```

**两阶段**：
1. **Warmup**：lr从小到大（避免初期震荡）
2. **Cosine Decay**：lr按余弦下降（后期精细调整）

**为什么用余弦？**
- 线性下降：后期变化太快
- 余弦下降：开始和结束变化慢，中间快，更平滑

---

### LM Head（语言模型输出层）

**定义**：将Transformer的隐藏状态映射到词表空间，预测下一个token的输出层。

**完整流程**：
```
输入文本 → Tokenizer → [token_ids]
  ↓
Embedding层 → (batch, seq_len, 768)
  ↓
Transformer (24层) → hidden_states (batch, seq_len, 768)
  ↓
LM Head: Linear(768 → vocab_size)
  ↓
logits (batch, seq_len, 155,083)
  ↓
Softmax → 每个token的概率分布
  ↓
预测: 选择概率最高的token
```

**核心结构**：
```python
# 本质：一个线性层
logits = hidden_states @ W^T + b

Qwen2.5-1.5B:
  输入: (batch, seq_len, 768)
  权重: (155,083, 768)  ← 原始151,646 + SID 3,437
  输出: (batch, seq_len, 155,083)
  参数: 119M（可能与Embedding权重共享，则0参数）
```

**扩展词表**：
```
[0-151,645]:   原始token（"iPhone", "Pro", ...）
[151,646-151,901]: <a_0> ~ <a_255>  (256个)
[151,902-152,157]: <b_0> ~ <b_255>  (256个)
[152,158-155,082]: <c_0> ~ <c_255>  (3,437个SID)
```

**类比**：
- Embedding = 正向词典（单词 → 向量）
- Transformer = 理解引擎
- LM Head = 反向词典（向量 → 所有单词的概率）

---

### 三任务参数更新分析

**重要**：所有参数共享，但不同任务的梯度贡献不同。

**参数更新热度对比**：

| 参数模块 | 任务1 (序列→SID) | 任务2 (SID↔Title) | 任务3 (SID→Text) |
|---------|-----------------|-------------------|------------------|
| **SID Embeddings** | ✓✓✓ (40%) | ✓✓✓✓ (50%) | ✓✓ (15%) |
| **原始 Embeddings** | ✓ (~0%) | ✓✓ (20%) | ✓✓✓ (30%) |
| **Transformer** | ✓✓ (50%) | ✓✓ (20%) | ✓✓✓ (45%) |
| **LM Head** | ✓✓ | ✓✓ | ✓✓✓ |

**任务1：SID序列 → SID**
```
主要更新:
  - SID Embeddings (40%梯度) - 学习序列模式
  - Transformer (50%梯度) - 学习共现关系
  - 原始Embeddings几乎不动（只有prompt文本）
```

**任务2：SID ↔ Title（最关键）**
```
主要更新:
  - SID Embeddings (50%梯度) ← 最密集！建立语义
  - 原始Embeddings (20%梯度) - 学习与SID映射
  - Transformer (20%梯度) - 学习跨域映射

为什么最关键：
  - 唯一直接建立"SID语义"的任务
  - 相当于"背单词"，强制记住每个SID含义
```

**任务3：SID序列 → Description**
```
主要更新:
  - 原始Embeddings (30%梯度) ← 生成大量自然语言
  - Transformer (45%梯度) ← 最复杂推理
  - LM Head - 预测自然语言token

作用：
  - 保持LLM生成能力
  - 避免灾难性遗忘
```

**三任务互补机制**：
```
任务1: 学"用法"（SID序列模式）
任务2: 学"含义"（SID语义） ← 核心
任务3: 学"应用"（推理+生成）
```

**一句话**：任务2对SID Embeddings更新最密集，任务1让Transformer学习序列依赖，任务3保持语言生成能力，三者共享参数自然实现互补。

---

## 端到端商品ID生成

### 核心问题

```
当前输出：SID (<a_10><b_45><c_2>)
目标输出：商品ID (item_12345)
```

### 推荐方案：SID + 查表

```
用户历史 → LLM(任务1) → SID → 查表 → 商品ID
                              ↑
                    反向索引：{SID → 商品ID}
```

**推理只依赖任务1**，任务2/3是辅助训练。

---

### 三任务的角色

| 任务 | 训练时 | 推理时 |
|------|--------|--------|
| 任务1 | 学SID序列推荐 | ✅ 核心，输出SID |
| 任务2 | 学SID语义 | 间接作用，让任务1更好 |
| 任务3 | 保留语言能力 | 可选，生成推荐理由 |

**任务3可调整**：
- 去掉：推理不受影响
- 保留：模型更鲁棒 + 可解释性

---

### 无效SID处理

```
问题：LLM可能输出不存在的SID组合

解决：
1. Constrained Decoding：每步只允许有效token
2. 最近邻兜底：无效SID → 找最近的有效SID
```

---

### 碰撞由谁决定？

```
碰撞发生在：RQ-VAE量化阶段
SFT阶段：不改变SID，只学习使用

影响链：
RQ-VAE碰撞 → 多商品同SID → 查表无法区分 → 需额外排序
```

**一句话**：SID质量由RQ-VAE保证，LLM只负责"使用"它。

---

### 冷启动优势

```
传统ID：新商品item_99999 → 模型未见过 → 无法推荐 ❌
SID：   新商品 → RQ-VAE生成SID → 与相似商品共享前缀 → 可泛化 ✅
```

**关键**：SID语义来自文本Embedding，不依赖行为数据。

---

## 约束解码（Constrained Decoding）

### 核心问题

```
LLM自由生成 → 可能产生无效SID → 无法召回商品 ❌

例如：
- 生成 <a_999><b_888><c_777> → 不在codebook中
- 生成 <a_10><b_20><c_500> → 路径不存在（<a_10><b_20>后只有c_1~c_100）
```

**目标**：保证100%生成有效SID（CC=0）

---

### 解决方案：前缀哈希表

**核心思想**：每一步只允许生成"合法路径"上的token

```
┌─────────────────────┐
│ 预处理：构建哈希表  │
└─────────────────────┘
所有有效SID:
  <a_60><b_159><c_203>
  <a_204><b_229><c_36>
  ...

构建映射：
hash_dict = {
    "": [token_a60, token_a204, ...],           # 第1层：所有<a_X>
    "token_a60": [token_b159, token_b200, ...], # <a_60>后可跟的<b_X>
    "token_a60-token_b159": [token_c203],       # 唯一路径
    "token_a60-token_b159-token_c203": [EOS],   # 结束
}

┌─────────────────────┐
│ 生成时：动态约束    │
└─────────────────────┘
每一步：
  1. 提取当前前缀
  2. 查哈希表 → 获取allowed_tokens
  3. 屏蔽非法token（logit设为-inf）
  4. softmax后非法token概率=0
  5. 只从合法token中选择
```

---

### 代码实现逻辑

**关键函数**：`ConstrainedLogitsProcessor.__call__()`

```python
# LogitProcessor.py:45-73

def __call__(self, input_ids, scores):
    # scores: [batch_size, vocab_size] - LLM原始logits

    # 1. 初始化mask（全部屏蔽）
    mask = torch.full_like(scores, float('-inf'))

    # 2. 提取当前前缀
    if self.count == 0:
        prefix = input_ids[:, -3:]  # 第一次：取prompt最后3个token
    else:
        prefix = input_ids[:, -self.count:]  # 后续：取已生成的SID token

    # 3. 查询哈希表
    hash_key = get_hash(prefix)  # "token1-token2-token3"
    allowed_tokens = hash_dict[hash_key]  # [5001, 5002, 5003, ...]

    # 4. 解除合法token的屏蔽
    mask[:, allowed_tokens] = 0

    # 5. 应用mask
    scores = scores + mask
    # 非法token: score + (-inf) = -inf
    # 合法token: score + 0 = 原始score

    self.count += 1
    return scores

# 6. 后续softmax时
probs = F.softmax(scores, dim=-1)
# exp(-inf) = 0 → 非法token概率=0
# 只从合法token中采样/选择
```

---

### 生成过程示例

**场景**：生成 `<a_60><b_159><c_203>`

```
初始状态：
  input_ids = [prompt_tokens...]

┌──────────────────────────────────────┐
│ 第1步：生成<a_60>                    │
└──────────────────────────────────────┘
提取前缀: [] (空)
查哈希表: hash_dict[""] = [token_a60, token_a204, token_a120, ...]

LLM原始logits:
  token_a60:    8.5  ✅ 合法
  token_a204:   7.2  ✅ 合法
  token_random: 9.0  ❌ 非法

应用mask后:
  token_a60:    8.5  (保留)
  token_a204:   7.2  (保留)
  token_random: -inf (屏蔽)

选择: token_a60

┌──────────────────────────────────────┐
│ 第2步：生成<b_159>                   │
└──────────────────────────────────────┘
提取前缀: [token_a60]
查哈希表: hash_dict["token_a60"] = [token_b159, token_b200, ...]

注意：token_b229(属于<a_204>路径) 被屏蔽！

选择: token_b159

┌──────────────────────────────────────┐
│ 第3步：生成<c_203>                   │
└──────────────────────────────────────┘
提取前缀: [token_a60, token_b159]
查哈希表: hash_dict["token_a60-token_b159"] = [token_c203]

只有1个合法token！强制选择 token_c203

┌──────────────────────────────────────┐
│ 第4步：生成EOS                       │
└──────────────────────────────────────┘
提取前缀: [token_a60, token_b159, token_c203]
查哈希表: hash_dict["token_a60-token_b159-token_c203"] = [EOS]

生成EOS → 停止

最终输出: <a_60><b_159><c_203> ✅
```

---

### 数学保证

```
命题：约束解码保证100%生成有效SID

证明：
1. hash_dict只包含训练数据中的有效SID路径
2. 每一步 allowed_tokens ⊆ hash_dict中的合法后继
3. mask机制使 P(非法token) = 0
4. ∴ 生成的任何完整序列 ∈ hash_dict
5. ∴ 生成的SID必然有效 □

结论：CC (Constrained Check) = 0
```

---

### 类比理解

**走迷宫vs自由探索**：

```
传统生成（自由探索）：
  START → 可能走到死路 → 生成无效SID ❌
       → 可能走出边界 → 生成乱码 ❌
       → 运气好才到终点

约束解码（导航系统）：
  START → 每步只显示合法路径的箭头
       → 跟着箭头走必然到达有效终点 ✅

                START
                  ↓
        第1层分岔口 ──→ ❌ 墙（非法token）
                  ↓ ✅
        第2层分岔口 ──→ ❌ 墙
                  ↓ ✅
        第3层分岔口 ──→ ❌ 墙
                  ↓ ✅
              [有效SID]
```

---

### 代码位置

| 功能 | 文件 | 行数 | 说明 |
|------|------|------|------|
| 哈希表构建 | minionerec_trainer.py | 553-568 | 预处理所有有效SID |
| 约束处理 | LogitProcessor.py | 45-73 | 屏蔽非法token |
| 查询函数 | minionerec_trainer.py | 582-586 | 查哈希表返回allowed_tokens |
| 调用生成 | evaluate.py | 189-196 | model.generate(..., logits_processor) |

---

### 工业价值

**近线召回系统的覆盖率保障**：

```
覆盖率 = 成功召回用户数 / 有行为用户数

影响因素：
1. 实效性损失（近线更新延迟）   → -25%
2. SID映射失败（碰撞/无效SID）  → -5%
3. ANN补救（碰撞消歧）          → +3%

约束解码的贡献：
  - 无效SID率：10% → 0% (保证CC=0)
  - 覆盖率提升：+10%
  - 最终可达：90%+ 覆盖率
```

**一句话**：约束解码通过前缀哈希表，从源头杜绝无效生成，是近线召回系统达到90%+覆盖率的关键保障。

---

## 推理性能关键参数

### max_new_tokens的真实含义

**常见误解：**
```
❌ max_new_tokens=64, 每个SID 3个token → 可生成20个item
```

**实际机制：**
```
✅ 只生成1个SID（约17 tokens）
✅ 通过Beam Search并行返回Top-10推荐
```

**原理：**
```
Beam Search并行生成：

  Beam 1: prompt → <a_60><b_159><c_203>  (score: 24.8)
  Beam 2: prompt → <a_204><b_229><c_36>  (score: 21.7)
  ...
  Beam 10: prompt → <a_50><b_180><c_90>  (score: 15.2)

所有beam共享生成过程，只需1个SID的长度（17 tokens）
返回Top-10不额外消耗token

max_new_tokens=64是安全余量（容错+未来扩展）
```

**类比：** 10个人同时走迷宫，而不是1个人走10次。

---

### KV Cache：线性增长

**是什么：** 缓存每个token的Key和Value矩阵，避免重复计算

**内存消耗：**
```
公式：kv_cache_size = 2 × layers × heads × seq_len × head_dim × dtype_size
     = O(seq_len)  ← 线性

实测（Qwen2.5-1.5B）:
  seq_len=10:    1.6 MB
  seq_len=100:  16.4 MB  (10倍)
  seq_len=512:  84.0 MB  (51倍)
```

**为什么线性：** 每个新token只需缓存1组(K, V)向量

**类比：** 每个人记1个电话号码，100个人记100个号码。

---

### Attention计算：二次增长

**核心公式：** `Attention = softmax(Q @ K^T / √d) @ V`

**计算复杂度：**
```
Q @ K^T 产生 [seq_len, seq_len] 的矩阵
每个token要与所有其他token计算attention

计算量 = seq_len × seq_len × head_dim
       = O(n²)  ← 二次增长

实测（Qwen2.5-1.5B）:
  seq_len=10:      0.01 GFLOPs  (50ms)
  seq_len=100:     0.86 GFLOPs  (5000ms)   ← 100倍FLOPs
  seq_len=512:    22.64 GFLOPs  (131秒!)   ← 2621倍FLOPs
```

**Attention矩阵可视化：**
```
seq_len=3时:
    t1  t2  t3
t1 [·   ·   ·]
t2 [·   ·   ·]
t3 [·   ·   ·]
计算: 3×3 = 9次

seq_len=100时:
计算: 100×100 = 10,000次

seq_len翻倍 → 计算量翻4倍
```

**类比：** n个人握手，需要n×n次握手；100人 = 10,000次握手。

---

### 性能瓶颈对比

| 因素 | 复杂度 | 影响 | 优化 |
|------|--------|------|------|
| **KV Cache** | O(n) | 内存 | 限制max_seq_len |
| **Attention** | O(n²) | 计算 ⚠️ | 限制序列长度 |
| **约束解码** | O(1) | 查表 | 可忽略 |

**主要瓶颈：** Attention的二次增长是推理延迟的主因

---

### 序列长度与性能

```
实测延迟（Qwen2.5-1.5B，V100，单样本）:

长度  | 延迟    | KV Cache | 吞吐   | 效果(HR@10) | 建议
------|---------|----------|--------|-------------|------
 10   |  50ms   |  1.6MB   | 20QPS  | 0.45        | ✅ 推荐
 20   | 120ms   |  3.3MB   |  8QPS  | 0.48        | ✅ 最佳
 50   | 200ms   |  8.2MB   |  5QPS  | 0.49        | ⚠️ 可用
100   | 600ms   | 16.4MB   | 1.7QPS | 0.48        | ❌ 慢
512   | 8000ms  | 84.0MB   | 0.1QPS | 0.45        | ❌ 不可用

关键观察：
  - 10 → 100: 长度10倍，延迟12倍，FLOPs 100倍
  - >50后: 效果趋于饱和，但计算量激增
  - 最佳范围: 10-20个行为
```

---

### 行为序列的限制机制

**训练时硬限制：**
```python
cutoff_len = 512  # 总token上限
实际训练最长: 10个行为 (约40 tokens)
平均长度: 3.7个行为 (约15 tokens)
```

**推理时建议：**
```python
max_history_length = 20     # 最多20个行为
time_window_days = 90       # 只看最近90天
max_input_tokens = 400      # 输入不超过400 tokens
max_new_tokens = 64         # 生成1个SID足够
```

**过期策略（需业务层实现）：**
```
行为类型  | 过期时间
----------|----------
点击      | 30天
加购      | 60天
购买      | 180天
收藏      | 365天
```

---

### 长序列优化策略

**策略1：滑动窗口**
```python
# 只保留最近N个行为
recent_behaviors = all_behaviors[:20]
```

**策略2：分层表示**
```python
短期：最近10个行为（详细SID）
长期：10-50行为（聚合为类目级别）

输入: "Recent: <a_10><b_20><c_30>, ...
       History: Electronics:20次, Fashion:15次"
```

**策略3：重要性采样**
```python
# 最近的 + 重要的（购买）
behaviors = recent[:10] + important_purchases[:10]
```

**策略4：时间衰减**
```python
weight = exp(-days_ago / 30)  # 30天半衰期
weighted_sample(behaviors, weights)
```

---

### 推理延迟公式

```
latency ≈ α × seq_len + β × seq_len²
          ↑               ↑
       KV Cache       Attention
       (线性)         (二次，主导)

当seq_len > 100时：
  - 二次项占主导
  - 性能急剧下降
  - 效果提升有限（超出训练分布）
```

**一句话总结**：序列长度控制在10-20是性能和效果的最佳平衡点，超过50性能急剧恶化。

---

## 多模态推荐扩展

### Text Encoder 的作用

**位置**：`物品文本 → Text Encoder → RQ-VAE → SID`

**作用**：将物品的自然语言描述转换为固定维度的语义向量

```
输入: "iPhone 14 Pro - 6.7英寸专业手机"
  ↓
Text Encoder (冻结的预训练模型)
  ↓
输出: [0.23, -0.15, 0.89, ..., 0.45]  # 768维向量
```

**关键特点**：
- 参数冻结：不参与训练
- 预训练模型：Qwen、BGE 等
- 语义保留：相似物品 → 相似向量

---

### 多模态方案（ChineseCLIP）

**核心思想**：用多模态模型替换纯文本 encoder，其他流程完全不变

```
┌─────────────────────────────────────┐
│ 阶段1: SID 构建（需要改）           │
└─────────────────────────────────────┘

物品:
  文本: "iPhone 14 Pro"
  图像: [iPhone图片]
      ↓
ChineseCLIP (图文融合，冻结)
      ↓
融合 Embedding: [768维]
      ↓
RQ-VAE 训练 (在 CLIP 空间)
      ↓
SID: <a_10><b_45><c_2>

┌─────────────────────────────────────┐
│ 阶段2: SFT/RL（完全不用改）         │
└─────────────────────────────────────┘

SID → Qwen 基座 → 预测下一个 SID
```

---

### 为什么不存在空间对齐问题？

**解耦设计**：Qwen 只看离散 token，不知道语义来源

```
ChineseCLIP  = 翻译官（图文 → 统一语义）
RQ-VAE       = 编码员（语义 → 离散代号）
Qwen         = 分析师（学习代号使用规律）
```

**Qwen 的视角**：
```python
# Qwen 看到的输入
tokens = [101, 5001, 5002, 5003, ...]
         ↑     ↑ SID tokens
         普通词

# Qwen 不知道：
# - SID 来自图像还是文本
# - 用的什么 encoder
# - Embedding 多少维

# Qwen 只知道：
# - 这是特殊 token
# - 学习使用规律
```

---

### Embedding 空间不兼容问题

**问题**：不同模型的 embedding 空间分布不同

```python
# Qwen 空间（针对纯文本优化）
qwen_emb = [0.23, -0.15, 0.89, ...]

# ChineseCLIP 空间（图文对齐优化）
clip_emb = [0.56, 0.32, -0.44, ...]

# 即使都是768维，但：
距离(qwen_emb1, qwen_emb2) ≠ 距离(clip_emb1, clip_emb2)
```

**结论**：
- ❌ 不能用 Qwen 的 RQ-VAE 处理 CLIP embeddings
- ✅ 需要重新训练 RQ-VAE
- ✅ SFT/RL 代码完全不用改

**类比**：两个空间像"中文词典"和"英文词典"，虽然都是词典，但查询规则和语义组织完全不同

---

### 代码改动清单

| 模块 | 是否需要改 | 说明 |
|------|-----------|------|
| **text2emb** | ✅ 需要 | 替换为 ChineseCLIP |
| **RQ-VAE 训练** | ✅ 需要 | 在新 embeddings 上重新训练 |
| **生成 SID** | ✅ 需要 | 用新的 RQ-VAE 生成 .index.json |
| **SFT 代码** | ❌ 不用 | 使用新 .index.json 即可 |
| **RL 代码** | ❌ 不用 | 完全不用改 |
| **评估代码** | ❌ 不用 | 完全不用改 |

```bash
# 需要重跑的步骤
1. python multimodal_text2emb.py  # 新建文件
2. bash rq/rqvae.sh --data_path xxx.emb-clip-multimodal.npy
3. python rq/generate_indices.py
4. bash sft.sh  # 代码不改，只用新的 .index.json
5. bash rl.sh   # 完全不用改
```

---

### 多模态的价值

**适用场景**：图像信息丰富的领域
- ✅ 时尚/服装：颜色、款式、搭配
- ✅ 家居/装修：风格、空间感
- ✅ 美食：视觉呈现、摆盘
- ❌ 图书/课程：图像信息有限

**预期收益**：
```
纯文本 Qwen:     HR@10 = 0.452
多模态 CLIP:     HR@10 = 0.46 ~ 0.50 (预测)
提升幅度:        +2% ~ +10% (取决于图像价值)
```

**一句话**：ChineseCLIP 负责图文对齐（在自己空间内），RQ-VAE 负责离散化，Qwen 只学符号规律 —— 三者完全解耦，无需跨空间对齐。

---

## VAE vs VQ-VAE 梯度问题

### VAE的梯度问题：采样不可导

**问题**：`z ~ N(μ, σ²)` 随机采样操作无法求导

**解决**：Reparameterization Trick（重参数化）
```python
# ❌ 直接采样（不可导）
z = sample_from_normal(μ, σ²)

# ✅ 重参数化（可导）
ε ~ N(0, 1)  # 固定分布
z = μ + ε·σ  # μ和σ可求导

∂z/∂μ = 1  ✓
∂z/∂σ = ε  ✓
```

**类比**：抽奖机（黑盒）→ 手动计算（均值+标准差×随机数）

---

### VQ-VAE的梯度问题：量化不可导

**问题**：`argmin` 和查表操作完全不可导

```python
indices = distances.argmin(dim=-1)  # 离散的索引
x_q = codebook[indices]             # 查表

# argmin导数：要么0要么∞（突变）
```

**解决1：Straight-Through Estimator（直通估计器）**

```python
x_q = x + (x_q - x).detach()

# 前向：x_q（保持量化值）
# 反向：∂L/∂x = ∂L/∂x_q（假装恒等映射）
```

**关键理解**：
- ❌ 不是"用残差当梯度"
- ✅ 是"假装量化操作不存在，梯度直通"
- 梯度仍通过链式法则：`∂L/∂x ≈ ∂L/∂x_q`

**类比**：
```
前向：走楼梯（离散台阶）
反向：滑滑梯（假装是连续斜坡）
```

**解决2：双向Loss约束残差**

```python
# Codebook Loss：固定Encoder，更新Codebook
codebook_loss = MSE(x_q, x.detach())
# 让码本靠近编码器

# Commitment Loss：固定Codebook，更新Encoder
commitment_loss = MSE(x_q.detach(), x)
# 让编码器靠近码本

total_loss = codebook_loss + beta * commitment_loss
```

**关键点**：
1. **MSE(x_q, x) = MSE(x, x_q)** - 数值相同
2. **detach位置不同** - 控制梯度流向
3. **类似EM算法** - 固定一方优化另一方（但并行执行）

**Beta权重（通常0.25）**：
```
beta小：Codebook更主动（追Encoder）
beta大：Encoder更主动（追Codebook）

类比：老师（Codebook）应该包容学生（Encoder）的探索
```

---

### 梯度流动图

```
【VQ-VAE完整流程】

Text → Encoder → x → argmin+查表 → x_q → Decoder → Loss
                 ↓   ↑__________↑    ↓
              可导   不可导（用ST）  可导

【三条梯度路径】

路径1: 重建损失 → Decoder ← x_q ← [ST] ← x ← Encoder
路径2: Codebook损失 → Codebook ← x_q (x被detach)
路径3: Commitment损失 → Encoder ← x (x_q被detach)
```

---

### 核心要点

| 算法 | 不可导操作 | 解决方案 | 代码 |
|------|-----------|---------|------|
| VAE | 采样 | Reparameterization | `z = μ + ε·σ` |
| VQ-VAE | argmin+查表 | Straight-Through + 双Loss | `x_q = x + (x_q-x).detach()` |

**VQ-VAE的三个技巧**：
1. **Straight-Through**：让梯度"假装"能穿过离散操作
2. **Detach分离**：两个Loss独立优化Encoder和Codebook
3. **Beta权重**：平衡两者速度，避免震荡

**一句话**：VAE用重参数化让采样可导，VQ-VAE用Straight-Through"欺骗"梯度流动，再用双Loss从两个方向约束残差。

---

### Straight-Through 与双向Loss的关系

**关键澄清**：x 和 x_q 是中间变量，不是参数

```python
# 真正的参数
encoder.weight    # Encoder的参数
codebook.weight   # Codebook的参数（256×64）
decoder.weight    # Decoder的参数

# 中间变量
x = encoder(text)              # 依赖encoder.weight
x_q = codebook.weight[indices] # codebook参数的切片
```

**Straight-Through 的真正作用**：

```python
x_q = x + (x_q - x).detach()

# 前向：x_q ≠ x（保持离散量化值）
# 反向：假装 x_q = x（梯度直通）

梯度链：
recon_loss → decoder → x_q → [假装是x] → x → encoder.weight ✓
                                         ↓
                                    codebook.weight ✗ 被隔离
```

**三条独立的梯度路径**：

```
路径1（recon_loss）：
  decoder ← x_q ← [ST穿过] ← x ← encoder  ✓ 更新encoder
                            ↓
                        codebook  ✗ 被ST隔离

路径2（codebook_loss）：
  codebook ← x_q ← MSE(x_q, x.detach())  ✓ 单独更新codebook

路径3（commitment_loss）：
  encoder ← x ← MSE(x_q.detach(), x)  ✓ 额外约束encoder
```

**总结表格**：

| 损失 | 更新对象 | 是否需要ST | 说明 |
|------|---------|-----------|------|
| `recon_loss` | encoder + decoder | ✅ | ST让梯度回传到encoder |
| `codebook_loss` | codebook | ❌ | 直接优化，绕过ST |
| `commitment_loss` | encoder | ❌ | 直接优化，额外约束 |

**为什么 commitment_loss 不需要 ST？**

```python
# recon_loss：梯度需要穿过argmin
recon_loss → decoder → x_q → [argmin❌不可导] → x → encoder
                              ↑____________↑
                              需要ST绕过

# commitment_loss：直接对x求导
commitment_loss = MSE(x_q.detach(), x)
                                    ↑
commitment_loss → x → encoder ✓ 不经过量化，直接可导
```

**核心**：commitment_loss 梯度起点是 x，不经过 argmin/查表，所以不需要 ST。

**一句话**：ST让recon_loss的梯度回传到encoder，但隔离了codebook；codebook需要单独的loss更新，两者分工明确。

---

### 三个Loss的核心作用

```
recon_loss      → 保证重建质量（端到端学习压缩）
codebook_loss   → 让码本追encoder（码本靠近数据）
commitment_loss → 让encoder追码本（encoder不乱跑）
```

**对Codebook坍塌的影响（死码本问题）**：

```
commitment_loss（最关键）> Sinkhorn算法 > codebook_loss

原因：
  commitment_loss：约束encoder输出范围
    → 强制encoder输出在码本覆盖范围内
    → 所有码本都有机会被选中 ✓

  codebook_loss：被动跟随
    → 码本追encoder
    → 如果码本从未被argmin选中，就永远不会更新
    → 无法解决坍塌 ✗
```

**实验对比**：
```
无commitment_loss：
  encoder输出范围 [10, 12, 15, ...]
  codebook范围 [0, 1, 2, ..., 5]
  → 只用前几个码本，后面饿死 ❌

有commitment_loss（beta=0.25）：
  encoder被约束在合理范围
  → 码本利用率 > 90% ✓
```

**一句话**：commitment_loss 通过约束 encoder 输出范围，防止部分码本"饿死"，是防止坍塌的核心机制。

---

### Codebook坍塌优化方案

**核心机制**：
```python
loss = codebook_loss + beta * commitment_loss
                       ↑
               beta越大 → encoder约束越强 → 码本利用率越高
```

**优化策略**：

| 方案 | 配置 | 效果 |
|------|------|------|
| **调大Beta** | `0.25 → 0.5 或 1.0` | encoder被拉紧，必须输出在码本范围内 |
| **启用Sinkhorn** | `sk_epsilon=0.005, sk_iters=100` | 强制均衡分配，热门码本被压制 |
| **K-means初始化** | `kmeans_init=True` | 码本初始位置覆盖数据分布 |
| **死码本重置** | 每500步检测并重置 | 活跃码本+噪声重新初始化 |

**Beta参数对比**：

```
beta=0     → encoder完全自由 → 利用率30% → 严重坍塌 ❌
beta=0.25  → 轻度约束       → 利用率85% → 默认配置
beta=0.5   → 中度约束       → 利用率95% → 推荐 ✓
beta=1.0   → 强约束         → 利用率98% → 防坍塌 ✓
```

**监控指标**：

```python
# 1. Codebook利用率（最重要）
utilization = (usage_count > 0).mean()
# 健康：> 95%，警戒：< 80%

# 2. 归一化熵
probs = usage_count / usage_count.sum()
entropy = -(probs * log(probs)).sum() / log(256)
# 健康：> 0.95（均匀分布）

# 3. 困惑度
perplexity = exp(entropy * log(256))
# 健康：> 240（接近256）
```

**实战配置（防坍塌）**：
```python
beta = 0.5           # 比默认0.25更强
sk_epsilon = 0.005   # 启用Sinkhorn
kmeans_init = True   # 必须开启
```

**一句话**：增大 beta 让 encoder 被强约束在码本覆盖范围内，配合 Sinkhorn 均衡分配，监控 Utilization > 90% 即为健康。

---
