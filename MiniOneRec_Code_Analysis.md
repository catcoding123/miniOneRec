# MiniOneRec 代码深度解析

## 📋 目录

- [项目概述](#项目概述)
- [整体流程图](#整体流程图)
- [核心技术详解](#核心技术详解)
  - [1. SID构建](#1-sid构建-semantic-id-construction)
  - [2. SFT阶段](#2-sft阶段-supervised-fine-tuning)
  - [3. RL阶段](#3-rl阶段-reinforcement-learning)
  - [4. 数据处理流程](#4-数据处理流程)
  - [5. 评估系统](#5-评估系统)
- [学习路线建议](#学习路线建议)
- [关键技术点总结](#关键技术点总结)

---

## 项目概述

**MiniOneRec** 是一个生成式推荐系统框架，核心创新是将推荐问题转换为**语言生成问题**：

1. **物品→语义ID（SID）**：将每个商品压缩成一个特殊token
2. **推荐→生成**：基于用户历史序列，生成下一个物品的SID
3. **优化→RL**：用强化学习进一步优化推荐效果

**三大核心技术**：

1. **SID构建**：将物品embedding通过RQ-VAE/RQ-Kmeans量化为离散token
2. **SFT阶段**：监督微调，让模型学会预测下一个物品SID
3. **RL阶段**：基于GRPO的强化学习，用推荐指标作为reward优化

---

## 整体流程图

```
原始数据 → 数据预处理 → SID构建 → SFT训练 → RL优化 → 评估
   ↓           ↓            ↓         ↓         ↓        ↓
Amazon    数据过滤      RQ-VAE   监督微调   GRPO    HR@K
Reviews   构建序列      量化编码  多任务学习  强化学习  NDCG@K
```

---

## 核心技术详解

### 1. SID构建 (Semantic ID Construction)

**文件位置**: `rq/` 目录

**核心思想**: 将物品embedding压缩为离散token序列

#### 实现方法 (4种可选)

##### 方法A: RQ-VAE

**文件**: `rq/rqvae.py`, `rq/models/rqvae.py`

**架构**：
```
物品文本 → Text Encoder (Qwen/LLaMA) → 768维向量
        → MLP Encoder (压缩) → 64维向量
        → RQ量化器 (3层,每层256码本) → [idx1, idx2, idx3]
        → MLP Decoder (重建) → 768维向量
```

**关键代码** (`rq/models/rqvae.py:61-66`):
```python
def forward(self, x, use_sk=True):
    x = self.encoder(x)  # 压缩
    x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)  # 量化
    out = self.decoder(x_q)  # 重建
    return out, rq_loss, indices
```

**工作原理**:
- **Encoder**: 将768维embedding压缩到64维
- **RQ量化器**: 3层残差量化，每层256个码本
  - 第1层: 找最近的码本向量 → idx1
  - 第2层: 对残差再量化 → idx2
  - 第3层: 对残差的残差量化 → idx3
- **Decoder**: 重建原始embedding
- **Loss**: MSE重建损失 + VQ量化损失

**参数配置** (`rq/rqvae.py:40-44`):
```python
num_emb_list=[256, 256, 256]  # 3层，每层256个码本
e_dim=32                       # 码本向量维度
layers=[2048,1024,512,256,128,64]  # MLP层尺寸
```

##### 方法B: RQ-Kmeans

**文件**: `rq/rqkmeans_faiss.py`

- 使用faiss库进行高效K-means聚类
- 逐层残差量化
- **缺点**: 碰撞率较高，需要约束版本

##### 方法C: Constrained RQ-Kmeans

**文件**: `rq/rqkmeans_constrained.py`

- 添加平衡约束，确保SID均匀分布
- 使用额外层处理碰撞
- 需要安装: `k_means_constrained`

##### 方法D: RQ-Kmeans+

**文件**: `rq/rqkmeans_plus.py`

- GPR论文方法的首个开源实现
- 更高的语义保留能力
- 推荐使用此方法

---

### 2. SFT阶段 (Supervised Fine-Tuning)

**文件位置**: `sft.py`, `data.py`

**核心思想**: 多任务学习，将SID和自然语言对齐

#### 三大训练任务

**任务配置** (`sft.py:191-198`):
```python
train_datasets = []

# 任务1: 序列推荐
train_data1 = SidSFTDataset(train_file=train_file, ...)
train_datasets.append(train_data1)

# 任务2: SID-物品对齐
train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, ...)
train_datasets.append(train_data2)

# 任务3: 融合序列推荐
train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, ...)
train_datasets.append(train_data3)
```

##### 任务1: 序列推荐 (`SidSFTDataset`)

**目标**: 学习用户历史序列 → 下一个SID

```
输入: "用户历史: <10> <23> <45>"
输出: "<78>"  # 预测下一个物品的SID
```

##### 任务2: SID-物品对齐 (`SidItemFeatDataset`)

**目标**: 学习物品文本 → SID的映射

```
输入: "物品标题: iPhone 15 Pro Max"
输出: "<10>"  # 学习物品文本→SID的映射
```

##### 任务3: 融合序列 (`FusionSeqRecDataset`)

**目标**: 学习SID → 物品文本的映射

```
输入: "用户历史标题: iPhone 14, AirPods, iPad → 下一个物品: <78>"
输出: "iPhone 15 Pro Max"  # SID→物品标题的映射
```

#### Token扩展机制

**关键创新** (`sft.py:30-55`):
```python
class TokenExtender:
    """扩展tokenizer词表，添加SID tokens"""
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file

    def get_new_tokens(self):
        """从index.json中提取所有SID tokens"""
        # 例如: ["<0>", "<1>", ..., "<255>"]
        with open(self.index_file, 'r') as f:
            indices = json.load(f)

        new_tokens = set()
        for index in indices.values():
            for token in index:
                new_tokens.add(token)
        return sorted(list(new_tokens))
```

**使用方式** (`sft.py:149-159`):
```python
# 1. 加载SID tokens
token_extender = TokenExtender(data_path=..., dataset=...)
new_tokens = token_extender.get_new_tokens()

# 2. 扩展词表
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# 3. 初始化新token的embeddings (自动完成)
```

#### 可选: 冻结LLM参数

**配置** (`sft.py:162-183`):
```python
if freeze_LLM:
    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 只训练新增的SID token embeddings
    embedding_layer = model.get_input_embeddings()
    embedding_layer.weight.requires_grad = True

    # 3. 使用梯度mask，只更新新token部分
    def mask_grad(grad):
        grad[:original_vocab_size].zero_()  # 原始词表的梯度清零
        return grad

    embedding_layer.weight.register_hook(mask_grad)
```

**优势**:
- 大幅减少训练参数量
- 保留LLM的语言能力
- 只学习推荐相关的SID embeddings

#### 数据格式

**Tokenizer封装** (`data.py:13-35`):
```python
class Tokenizer:
    """封装tokenizer，统一处理BOS/EOS token"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.tokenizer.encode(s)
        # 移除默认的BOS/EOS
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]
        # 按需添加
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t
```

**训练样本构建** (`data.py:102-147`):
```python
def pre(self, idx):
    # 1. 构建instruction
    instruction = f"""Below is an instruction that describes a task...
    ### Instruction:
    {random.choice(self.instructs)}
    """
    tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

    # 2. 添加用户历史
    history = self.get_history(self.data.iloc[idx])
    prompt = f"""### User Input:
    {history["input"]}
    ### Response:\n"""
    tokens += self.tokenizer.encode(prompt, bos=False, eos=False)

    # 3. 添加标签 (只计算output部分的loss)
    golden_tokens = self.tokenizer.encode(history["output"], bos=False, eos=True)
    labels = [-100] * len(tokens) + golden_tokens  # -100表示不计算loss

    return {
        "input_ids": tokens + golden_tokens,
        "attention_mask": [1] * len(tokens + golden_tokens),
        "labels": labels
    }
```

---

### 3. RL阶段 (Reinforcement Learning)

**文件位置**: `rl.py`, `minionerec_trainer.py`, `LogitProcessor.py`

**核心思想**: GRPO (Group Relative Policy Optimization)

#### GRPO算法流程

**完整流程** (`minionerec_trainer.py`):

```python
# 1. 生成多个候选 (每个prompt生成K个候选)
for prompt in batch:
    candidates = model.generate(
        prompt,
        num_return_sequences=K,  # 通常K=4或8
        num_beams=K,
        do_sample=False
    )

# 2. 计算reward
rewards = []
for candidate in candidates:
    # 2.1 二元奖励 (命中=1, 未命中=0)
    r = hit_reward(candidate, ground_truth)

    # 2.2 排序惩罚 (惩罚高概率但错误的答案)
    if r == 0:  # 未命中
        prob = get_probability(candidate)
        rank_penalty = -log(prob + eps)
        r += alpha * rank_penalty

    # 2.3 协同过滤分数 (可选)
    cf_score = collaborative_filtering_score(candidate, user_history)
    r += beta * cf_score

    rewards.append(r)

# 3. 组内归一化 (GRPO的关键)
normalized_rewards = (rewards - mean(rewards)) / (std(rewards) + eps)

# 4. 计算策略梯度损失
loss = 0
for i, candidate in enumerate(candidates):
    log_prob = model.get_log_prob(prompt, candidate)
    advantage = normalized_rewards[i] - baseline

    # Policy gradient loss
    pg_loss = -log_prob * advantage

    # KL散度惩罚 (保持策略接近参考模型)
    kl_penalty = KL_divergence(model, ref_model, prompt, candidate)

    loss += pg_loss + kl_weight * kl_penalty

# 5. 反向传播更新
loss.backward()
optimizer.step()
```

#### Reward函数设计

**三部分组成**:

1. **Hit Reward** (二元奖励):
```python
def hit_reward(predicted_sid, ground_truth_sid):
    return 1.0 if predicted_sid == ground_truth_sid else 0.0
```

2. **Rank-aware Penalty** (惩罚高概率错误):
```python
def rank_penalty(predicted_sid, probability, ground_truth_sid):
    if predicted_sid != ground_truth_sid:
        # 概率越高的错误答案，惩罚越重
        return -alpha * log(probability + 1e-8)
    return 0.0
```

3. **Collaborative Filtering Score** (可选):
```python
def cf_score(predicted_item, user_history):
    # 基于协同过滤计算相似度
    # 例如：使用SASRec模型的预测分数
    return sasrec_model.score(user_history, predicted_item)
```

#### Constrained Decoding

**核心创新**: 确保生成的token序列是有效的SID

**实现** (`LogitProcessor.py:24-73`):

```python
class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    约束解码：确保每一步只生成有效的SID token
    """
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0  # 当前生成步数
        self.eos_token_id = eos_token_id

        # 根据模型类型确定prefix长度
        if base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 1. 转换为log概率
        scores = torch.nn.functional.log_softmax(scores, dim=-1)

        # 2. 创建mask，初始化为-inf (屏蔽所有token)
        mask = torch.full_like(scores, float('-inf'))

        # 3. 对每个beam，找到允许的tokens
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, -1)):
            for beam_id, sent in enumerate(beam_sent):
                # 3.1 提取当前前缀
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]  # 初始前缀
                else:
                    hash_key = sent[-self.count:]  # 已生成的SID部分

                # 3.2 查询哈希表，获取允许的下一个tokens
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(
                    batch_id,
                    hash_key.tolist()
                )

                # 3.3 如果没有有效token，强制生成EOS
                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(f"Invalid prefix {hash_key} at step {self.count}")
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue

                # 3.4 解除有效tokens的mask
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        # 4. 应用mask
        self.count += 1
        scores = scores + mask
        return scores
```

**工作原理**:

假设SID为3层，每层256个码本，token格式为 `<layer1_layer2_layer3>`：

```
步骤1: 生成 "<"
  - 允许的token: ["<"]

步骤2: 生成第1层索引
  - 允许的token: ["0", "1", ..., "255"] (256个)

步骤3: 生成 "_"
  - 允许的token: ["_"]

步骤4: 生成第2层索引
  - 允许的token: 取决于第1层的值
  - 例如: 如果第1层=10，则查表 hash_map["<10_"] -> ["3", "45", "78", ...]

步骤5: 生成 "_"
  - 允许的token: ["_"]

步骤6: 生成第3层索引
  - 允许的token: 取决于前两层
  - 例如: hash_map["<10_45_"] -> ["2", "19", "234"]

步骤7: 生成 ">"
  - 允许的token: [">"]
```

**哈希表构建** (在数据预处理阶段完成):

```python
# 构建前缀→有效下一个token的映射
hash_map = defaultdict(set)

for sid in all_sids:  # 例如: "<10_45_2>"
    tokens = sid.split("_")

    # 记录每个前缀的有效后续
    hash_map["<"].add(tokens[0])
    hash_map[f"<{tokens[0]}_"].add(tokens[1])
    hash_map[f"<{tokens[0]}_{tokens[1]}_"].add(tokens[2])
    hash_map[f"<{tokens[0]}_{tokens[1]}_{tokens[2]}"].add(">")

# 转换为列表形式供LogitsProcessor使用
def prefix_allowed_tokens_fn(batch_id, prefix):
    prefix_str = "".join(prefix)
    return list(hash_map[prefix_str])
```

#### RepeatRandomSampler

**作用**: 在RL阶段，每个样本需要生成多个候选

**实现** (`minionerec_trainer.py:80-118`):
```python
class RepeatRandomSampler(Sampler):
    """
    将数据集重复N次，用于GRPO
    例如: ["a", "b", "c"] → ["b", "b", "c", "c", "a", "a"]
    """
    def __init__(self, data_source, repeat_count, seed=None):
        self.data_source = data_source
        self.repeat_count = repeat_count  # 通常 = num_return_sequences
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(len(self.data_source), generator=self.generator)
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)
```

---

### 4. 数据处理流程

**文件位置**: `data/amazon18_data_process.py`

#### 完整流程

```python
# 1. 加载Amazon Reviews数据
reviews = load_reviews(category, start_date, end_date)
metadata = load_metadata(category)

# 2. 清洗文本
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)      # 移除HTML标签
    text = html.unescape(text)                # 解码HTML实体
    text = text.replace("&quot;", "\"")       # 替换特殊字符
    text = re.sub(r'\s+', ' ', text)          # 规范空格
    return text.strip()

# 3. 过滤数据 (user_k=5, item_k=5)
filtered_users = [u for u in users if len(u.interactions) >= 5]
filtered_items = [i for i in items if len(i.users) >= 5]

# 4. 构建用户历史序列
for user in filtered_users:
    history = sorted(user.interactions, key=lambda x: x.timestamp)

    # 创建训练样本: history[:-1] → history[-1]
    train_sample = {
        "user_id": user.id,
        "history_item_id": [h.item_id for h in history[:-1]],
        "history_item_title": [items[h.item_id].title for h in history[:-1]],
        "item_id": history[-1].item_id,
        "item_title": items[history[-1].item_id].title
    }
    train_data.append(train_sample)

# 5. 划分训练/测试集
train_set, test_set = split_by_time(train_data, split_ratio=0.8)

# 6. 保存为CSV
pd.DataFrame(train_data).to_csv("train.csv")
pd.DataFrame(test_data).to_csv("test.csv")
```

#### 文本编码 (Text2Embedding)

**文件**: `rq/text2emb/amazon_text2emb.py`

```python
# 1. 加载文本编码模型
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")

# 2. 对每个物品编码
item_embeddings = []
for item in items:
    # 拼接标题和描述
    text = item.title + " " + item.description
    text = clean_text(text)

    # 编码
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

    item_embeddings.append(embedding.cpu().numpy())

# 3. 保存
np.save("Industrial_and_Scientific.emb-qwen-td.npy", np.array(item_embeddings))
```

**多GPU加速** (新版本):
```python
# 使用Accelerate进行多GPU并行
from accelerate import Accelerator

accelerator = Accelerator()
model = accelerator.prepare(model)
dataloader = accelerator.prepare(dataloader)

for batch in dataloader:
    embeddings = model(**batch)
    all_embeddings = accelerator.gather(embeddings)
```

---

### 5. 评估系统

**文件位置**: `evaluate.py`, `calc.py`

#### 评估指标

##### 1. Hit Rate @ K

```python
def hit_rate_at_k(predictions, ground_truths, k=10):
    """
    计算Top-K命中率

    Args:
        predictions: 每个用户的Top-K预测列表
        ground_truths: 每个用户的真实物品
        k: 考虑前K个预测

    Returns:
        HR@K: 命中率
    """
    hits = 0
    for pred, gt in zip(predictions, ground_truths):
        if gt in pred[:k]:
            hits += 1

    return hits / len(predictions)
```

**示例**:
```
用户1: 预测=[10, 23, 45, 78, ...], 真实=23  → Hit!
用户2: 预测=[5, 8, 12, 19, ...], 真实=99   → Miss
用户3: 预测=[1, 2, 99, 4, ...], 真实=99   → Hit!

HR@10 = 2/3 = 0.667
```

##### 2. NDCG @ K

```python
def ndcg_at_k(predictions, ground_truths, k=10):
    """
    计算归一化折损累积增益

    NDCG考虑排序位置：排名越靠前的命中，得分越高
    """
    ndcgs = []
    for pred, gt in zip(predictions, ground_truths):
        # DCG: Discounted Cumulative Gain
        dcg = 0
        for i, item in enumerate(pred[:k]):
            if item == gt:
                dcg = 1.0 / math.log2(i + 2)  # i+2 因为位置从0开始
                break

        # IDCG: Ideal DCG (最好的情况：真实物品在第1位)
        idcg = 1.0 / math.log2(2)

        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return sum(ndcgs) / len(ndcgs)
```

**示例**:
```
用户1: 真实物品在位置2 → NDCG = 1/log2(3) ≈ 0.631
用户2: 真实物品在位置1 → NDCG = 1/log2(2) = 1.000
用户3: 真实物品在位置5 → NDCG = 1/log2(6) ≈ 0.387

NDCG@10 = (0.631 + 1.000 + 0.387) / 3 ≈ 0.673
```

##### 3. CC (Constrained Check)

```python
def constrained_check(predictions, valid_sids):
    """
    检查是否生成了无效的SID

    Args:
        predictions: 模型生成的SID列表
        valid_sids: 训练时的有效SID集合

    Returns:
        无效SID的数量 (应该为0)
    """
    invalid_count = 0
    for pred in predictions:
        if pred not in valid_sids:
            invalid_count += 1
            print(f"Invalid SID generated: {pred}")

    return invalid_count
```

**重要提示** (README.md:22):
> 如果CC指标非零，说明constrained decoding失败，模型生成了大量无效物品。
> 这可能与transformers库版本有关，建议切换到base模型（如Qwen2.5-base）。

#### 评估脚本

**使用方式** (`evaluate.sh`):
```bash
bash evaluate.sh --exp_name your_model_path

# 内部调用 evaluate.py
python evaluate.py \
    --model_path your_model_path \
    --test_file test.csv \
    --sid_index_path .index.json \
    --item_meta_path .item.json \
    --k_values 5 10 20
```

**输出示例**:
```
Evaluating model: ./output/Industrial_and_Scientific/checkpoint-1000

HR@5:  0.3245
HR@10: 0.4521
HR@20: 0.5678

NDCG@5:  0.2134
NDCG@10: 0.2456
NDCG@20: 0.2789

CC: 0  ✓ (All predictions are valid SIDs)
```

---

## 学习路线建议

### 阶段1: 理解核心概念 (1-2天)

- [x] 阅读 `README.md`
- [ ] 阅读论文: [arXiv:2510.24431](https://arxiv.org/abs/2510.24431)
- [ ] 查看 `rq/models/rqvae.py` - 理解RQ-VAE原理
- [ ] 绘制系统架构图

**学习重点**:
- 为什么要将物品转换为SID？
- RQ-VAE如何压缩embedding？
- 生成式推荐 vs 传统推荐的区别？

---

### 阶段2: 数据处理 (2-3天)

**实践步骤**:

1. **下载并处理Amazon数据**
```bash
# 下载数据 (约10GB)
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Industrial_and_Scientific.json.gz

# 解压
gunzip Industrial_and_Scientific.json.gz

# 处理数据
bash data/amazon18_data_process.sh \
    --dataset Industrial_and_Scientific \
    --user_k 5 \
    --item_k 5 \
    --st_year 2017 --st_month 10 \
    --ed_year 2018 --ed_month 11 \
    --output_path ./data/Amazon18
```

2. **生成物品embeddings**
```bash
bash rq/text2emb/amazon_text2emb.sh \
    --dataset Industrial_and_Scientific \
    --root ./data/Amazon18 \
    --plm_name qwen \
    --plm_checkpoint /path/to/Qwen-7B
```

3. **训练RQ-VAE**
```bash
bash rq/rqvae.sh \
    --data_path ./data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_dir ./output/rqvae \
    --lr 1e-3 \
    --epochs 10000 \
    --batch_size 20480
```

4. **生成SID映射**
```bash
python rq/generate_indices.py \
    --data_path ./data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_path ./output/rqvae/best_model.pth \
    --output_path ./output/Industrial_and_Scientific.index.json
```

5. **转换数据集格式**
```bash
python convert_dataset.py \
    --dataset_name Industrial_and_Scientific \
    --data_dir ./data/Amazon18/Industrial_and_Scientific \
    --output_dir ./data/processed
```

**学习重点**:
- 数据过滤的作用 (user_k=5, item_k=5)
- Text Encoder的选择 (Qwen vs LLaMA)
- RQ-VAE的碰撞率 (collision rate)
- SID的长度和码本大小的权衡

---

### 阶段3: SFT训练 (3-5天)

**实践步骤**:

1. **准备配置**
```bash
# 检查GPU
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=MiniOneRec
```

2. **启动SFT训练**
```bash
bash sft.sh \
    --base_model /path/to/Qwen2.5-1.5B-Instruct \
    --output_dir ./output/sft \
    --sid_index_path ./output/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/processed/Industrial_and_Scientific.item.json \
    --train_file ./data/processed/train.csv \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 10 \
    --learning_rate 3e-4
```

3. **监控训练过程**
```bash
# 查看wandb日志
wandb login
# 访问 https://wandb.ai/your-project

# 查看本地日志
tail -f ./output/sft/train.log
```

4. **评估SFT模型**
```bash
bash evaluate.sh --exp_name ./output/sft/checkpoint-best
```

**学习重点**:
- 三个训练任务如何协同工作？
- Token扩展后的embedding初始化策略
- freeze_LLM的作用和适用场景
- 如何平衡三个任务的loss权重？

**调试技巧**:
```python
# 可视化训练数据
from data import SidSFTDataset
dataset = SidSFTDataset(train_file="train.csv", ...)
sample = dataset[0]
print(tokenizer.decode(sample["input_ids"]))
print(sample["labels"])
```

---

### 阶段4: RL优化 (3-5天)

**实践步骤**:

1. **启动RL训练**
```bash
bash rl.sh \
    --model_path ./output/sft/checkpoint-best \
    --output_dir ./output/rl \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --num_return_sequences 8
```

2. **对比SFT和RL效果**
```bash
# 评估SFT模型
bash evaluate.sh --exp_name ./output/sft/checkpoint-best

# 评估RL模型
bash evaluate.sh --exp_name ./output/rl/checkpoint-best
```

3. **分析生成的样本**
```python
# 查看RL生成的candidates
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/rl/checkpoint-best")
tokenizer = AutoTokenizer.from_pretrained("./output/rl/checkpoint-best")

prompt = "User history: <10> <23> <45>"
candidates = model.generate(
    tokenizer.encode(prompt, return_tensors="pt"),
    num_beams=8,
    num_return_sequences=8,
    max_length=50
)

for i, c in enumerate(candidates):
    print(f"Candidate {i}: {tokenizer.decode(c)}")
```

**学习重点**:
- GRPO vs PPO的区别
- Constrained Decoding如何保证有效性
- Reward函数的设计哲学
- 组内归一化的作用
- KL散度惩罚的必要性

**实验想法**:
- 调整 `num_return_sequences` (4, 8, 16)
- 尝试不同的reward权重
- 添加diversity bonus
- 对比有/无constrained decoding的效果

---

### 阶段5: 实验与改进 (持续)

**改进方向**:

1. **SID构建优化**
   - [ ] 尝试RQ-Kmeans+ (更好的语义保留)
   - [ ] 调整码本层数和大小
   - [ ] 测试不同的Text Encoder (BERT, RoBERTa, etc.)

2. **训练策略优化**
   - [ ] 实现curriculum learning (从简单到困难)
   - [ ] 尝试多阶段训练 (先易后难)
   - [ ] 添加数据增强 (序列截断、打乱等)

3. **Reward函数改进**
   - [ ] 添加diversity reward
   - [ ] 引入时间衰减 (最近的历史权重更高)
   - [ ] 使用多个CF模型的ensemble

4. **跨数据集测试**
   - [ ] Amazon Books
   - [ ] Amazon Toys
   - [ ] Yelp
   - [ ] MovieLens

5. **性能优化**
   - [ ] 使用FlashAttention-2
   - [ ] 实现8-bit量化
   - [ ] 优化constrained decoding的哈希表查询

**实验记录模板**:
```markdown
## 实验 #001: 测试RQ-Kmeans+

### 配置
- SID方法: RQ-Kmeans+
- 码本: [512, 512, 512]
- Text Encoder: Qwen-7B
- Base Model: Qwen2.5-1.5B-Instruct

### 结果
| 模型 | HR@10 | NDCG@10 | 训练时间 |
|------|-------|---------|----------|
| SFT  | 0.452 | 0.245   | 3h       |
| RL   | 0.478 | 0.267   | 1.5h     |

### 分析
- RL相比SFT提升了5.8%的HR@10
- Constrained decoding有效，CC=0
- 发现：更大的码本提升了语义多样性

### 下一步
- 尝试4层码本 [256,256,256,256]
- 测试frozen LLM的效果
```

---

## 关键技术点总结

| 技术 | 文件位置 | 核心思想 | 关键参数 |
|------|---------|---------|----------|
| **RQ-VAE** | `rq/models/rqvae.py` | 多层残差量化，压缩embedding | `num_emb_list=[256,256,256]`<br>`e_dim=32` |
| **Token扩展** | `sft.py:30-55` | 向词表添加SID tokens | `new_tokens=["<0>", ...]` |
| **多任务SFT** | `sft.py:191-198` | 序列推荐+SID对齐+融合生成 | 3个Dataset的组合 |
| **GRPO** | `minionerec_trainer.py` | 组相对策略优化 | `num_return_sequences=8`<br>`kl_weight=0.1` |
| **Constrained Decoding** | `LogitProcessor.py` | 保证生成有效SID | `prefix_allowed_tokens_fn` |
| **Reward设计** | `rl.py` | Hit + Rank Penalty + CF | `alpha=0.1, beta=0.05` |
| **数据过滤** | `data/amazon18_data_process.py` | 保证数据质量 | `user_k=5, item_k=5` |
| **Text Encoder** | `rq/text2emb/amazon_text2emb.py` | 物品文本→embedding | `Qwen-7B / LLaMA-7B` |

---

## 常见问题 (FAQ)

### Q1: 为什么需要SID？直接用物品ID不行吗？

**A**: 传统ID (如 `item_12345`) 的问题：
- **无语义信息**: ID只是一个编号，模型无法理解物品含义
- **泛化能力差**: 新物品需要重新训练
- **冷启动困难**: 无法处理没见过的物品

**SID的优势**:
- **语义压缩**: 将物品的文本描述压缩为token
- **可解释性**: SID保留了部分语义信息
- **泛化能力**: 相似物品有相似的SID

---

### Q2: RQ-VAE的层数和码本大小如何选择？

**经验规则**:
- **层数**: 通常3-4层
  - 2层: 表达能力不足
  - 5层+: 碰撞率低，但token序列太长

- **码本大小**: 每层128-512
  - 128: 覆盖率可能不足
  - 1024: 计算成本高，增益有限

**推荐配置**:
```python
# 小数据集 (10K-50K items)
num_emb_list = [256, 256, 256]

# 中数据集 (50K-200K items)
num_emb_list = [512, 512, 512]

# 大数据集 (200K+ items)
num_emb_list = [512, 512, 512, 512]
```

**碰撞率检查**:
```python
# 计算理论容量
capacity = 256 * 256 * 256 = 16,777,216

# 实际物品数
num_items = 50,000

# 理论碰撞率
collision_rate = 1 - (capacity - num_items) / capacity ≈ 0.003
```

---

### Q3: freeze_LLM什么时候用？

**使用场景**:
- ✅ **预算有限**: 减少训练参数，降低显存需求
- ✅ **数据量小**: 避免过拟合LLM
- ✅ **保留语言能力**: 只学习推荐知识
- ✅ **快速实验**: 加速训练迭代

**不适用场景**:
- ❌ **从头训练**: 需要完整学习语言模型
- ❌ **充足资源**: 可以fine-tune整个模型
- ❌ **特定领域**: LLM预训练数据与目标域差异大

---

### Q4: RL阶段为什么有时效果不明显？

**可能原因**:

1. **SFT已经足够好**: RL的提升空间有限
```bash
# 解决: 降低SFT训练时长，留更多空间给RL
--num_epochs 5  # 原来10
```

2. **Reward设计不当**: 信号太弱或太noisy
```python
# 解决: 添加多样性奖励
def diversity_reward(candidates):
    unique_items = len(set(candidates))
    return unique_items / len(candidates)
```

3. **KL惩罚太强**: 策略更新太保守
```python
# 解决: 降低kl_weight
kl_weight = 0.05  # 原来0.1
```

4. **生成候选数太少**: GRPO需要足够的对比
```bash
# 解决: 增加候选数
--num_return_sequences 16  # 原来8
```

---

### Q5: Constrained Decoding失败 (CC > 0) 怎么办？

**诊断步骤**:

1. **检查transformers版本**
```bash
pip list | grep transformers
# 推荐: transformers==4.57.1
```

2. **验证哈希表完整性**
```python
# 检查是否所有SID都在哈希表中
with open(".index.json") as f:
    sid_map = json.load(f)

print(f"Total SIDs: {len(sid_map)}")
# 应该等于数据集中的物品数
```

3. **测试LogitsProcessor**
```python
from LogitProcessor import ConstrainedLogitsProcessor

processor = ConstrainedLogitsProcessor(
    prefix_allowed_tokens_fn=your_fn,
    num_beams=4,
    base_model="Qwen2.5-1.5B",
    eos_token_id=tokenizer.eos_token_id
)

# 模拟生成
input_ids = torch.tensor([[1, 2, 3, 4]])
scores = torch.randn(1, vocab_size)
processed_scores = processor(input_ids, scores)

# 检查是否有有效token
valid_tokens = (processed_scores > float('-inf')).sum()
print(f"Valid tokens: {valid_tokens}")
```

4. **切换到base模型**
```bash
# Instruct模型可能生成特殊token
--base_model Qwen2.5-1.5B  # 而不是 Qwen2.5-1.5B-Instruct
```

---

### Q6: 如何处理新物品 (冷启动)？

**策略**:

1. **重新编码SID** (推荐)
```python
# 为新物品生成embedding
new_item_emb = text_encoder.encode(new_item.title + " " + new_item.description)

# 使用训练好的RQ-VAE编码
new_sid = rqvae.get_indices(new_item_emb)

# 添加到SID映射表
sid_map[new_item.id] = new_sid
```

2. **基于相似度匹配**
```python
# 找到最相似的已有物品
similarities = cosine_similarity(new_item_emb, all_item_embs)
most_similar_idx = similarities.argmax()

# 使用相似物品的SID
new_sid = sid_map[most_similar_idx]
```

3. **在线学习** (高级)
```python
# 定期重新训练RQ-VAE，纳入新物品
# 注意: SID可能会变化，需要版本控制
```

---

### Q7: 多语言/跨领域支持？

**适配步骤**:

1. **选择多语言Text Encoder**
```python
# 例如: XLM-RoBERTa, mBERT
text_encoder = AutoModel.from_pretrained("xlm-roberta-large")
```

2. **多领域数据混合**
```python
# 在SFT阶段混合多个领域
train_data = concatenate([
    load_dataset("amazon_books"),
    load_dataset("amazon_electronics"),
    load_dataset("yelp")
])
```

3. **领域标记** (可选)
```python
# 添加领域前缀
prompt = f"[Domain: Books] User history: <10> <23> <45>"
```

---

## 参考资源

### 论文
- [MiniOneRec Paper](https://arxiv.org/abs/2510.24431)
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [GPR (RQ-Kmeans+)](https://arxiv.org/abs/...)

### 代码库
- [MiniOneRec GitHub](https://github.com/AkaliKong/MiniOneRec)
- [Hugging Face Models](https://huggingface.co/kkknight/MiniOneRec)
- [ModelScope](https://modelscope.cn/models/k925238839/MiniOneRec)

### 数据集
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

### 工具
- [Weights & Biases](https://wandb.ai/) - 训练监控
- [nvitop](https://github.com/XuehaiPan/nvitop) - GPU监控
- [tmux](https://github.com/tmux/tmux) - 多窗口终端

---

## 贡献者

文档整理：Claude Code
项目作者：Xiaoyu Kong et al. (LDS, AlphaLab, NExT)
最后更新：2026-01-22

---

**祝学习顺利！如有问题，欢迎提Issue讨论。** 🚀

