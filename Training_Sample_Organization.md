# MiniOneRec 训练样本组织详解

## 📋 目录

- [概述](#概述)
- [数据文件格式](#数据文件格式)
- [三大训练任务](#三大训练任务)
  - [任务1: 序列推荐 (SidSFTDataset)](#任务1-序列推荐-sidsftdataset)
  - [任务2: SID-物品对齐 (SidItemFeatDataset)](#任务2-sid-物品对齐-siditemfeatdataset)
  - [任务3: 融合序列推荐 (FusionSeqRecDataset)](#任务3-融合序列推荐-fusionseqrecdataset)
- [Token化处理](#token化处理)
- [完整示例](#完整示例)

---

## 概述

MiniOneRec的SFT阶段使用**多任务学习**，同时训练3个不同但相关的任务：

1. **SidSFTDataset**: SID序列 → 下一个SID (纯推荐任务)
2. **SidItemFeatDataset**: SID ⇄ 物品标题 (双向对齐任务)
3. **FusionSeqRecDataset**: SID序列 → 物品标题/描述 (融合任务)

这种设计让模型同时学习：
- 推荐能力 (预测下一个物品)
- 语义理解 (SID与自然语言的映射)
- 世界知识 (继承LLM的语言能力)

---

## 数据文件格式

### 输入文件

#### 1. 训练数据文件 (train.csv)

```csv
user_id,history_item_id,history_item_title,history_item_sid,item_id,item_title,item_sid
U001,"['I001', 'I002', 'I003']","['iPhone 14', 'AirPods Pro', 'iPad Air']","['<10_45_2>', '<23_67_8>', '<45_12_90>']",I004,MacBook Pro,<78_34_56>
U002,"['I005', 'I006']","['Samsung Phone', 'Galaxy Watch']","['<5_89_12>', '<67_23_45>']",I007,Galaxy Buds,<34_78_9>
```

**字段说明**:
- `user_id`: 用户ID
- `history_item_id`: 历史物品ID列表 (字符串格式的Python list)
- `history_item_title`: 历史物品标题列表
- `history_item_sid`: 历史物品的SID列表
- `item_id`: 目标物品ID
- `item_title`: 目标物品标题
- `item_sid`: 目标物品的SID

#### 2. 物品特征文件 (.item.json)

```json
{
  "I001": {
    "title": "iPhone 14 Pro Max",
    "description": "6.7-inch Super Retina XDR display with ProMotion...",
    "category": "Electronics",
    "price": 1099.99
  },
  "I002": {
    "title": "AirPods Pro 2nd Generation",
    "description": ["Active Noise Cancellation", "Adaptive Transparency", "Personalized Spatial Audio"],
    "category": "Audio",
    "price": 249.99
  }
}
```

**注意**:
- `description` 可以是字符串或列表
- 如果是列表，会选择最长的一个
- 如果为空，会用 `title` 代替

#### 3. SID索引文件 (.index.json)

```json
{
  "I001": ["<10_45_2>"],
  "I002": ["<23_67_8>"],
  "I003": ["<45_12_90>"],
  "I004": ["<78_34_56>"]
}
```

每个物品ID映射到一个SID。SID格式：`<layer1_layer2_layer3>`

---

## 三大训练任务

### 任务1: 序列推荐 (SidSFTDataset)

**目标**: 学习从用户历史SID序列预测下一个SID

**文件位置**: `data.py:483-590`

#### 数据构建流程

```python
class SidSFTDataset(Dataset):
    def get_history(self, row):
        # 1. 解析历史SID列表
        history_item_sid = eval(row['history_item_sid'])
        # 例如: ['<10_45_2>', '<23_67_8>', '<45_12_90>']

        # 2. 拼接为字符串
        history_str = ", ".join(history_item_sid)
        # 结果: "<10_45_2>, <23_67_8>, <45_12_90>"

        # 3. 获取目标SID
        target_sid = row['item_sid']
        # 例如: "<78_34_56>"

        return {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?",
            "output": target_sid + "\n"
        }
```

#### 样本示例

**原始数据**:
```csv
user_id: U001
history_item_sid: ['<10_45_2>', '<23_67_8>', '<45_12_90>']
item_sid: <78_34_56>
```

**转换后的训练样本**:

```
### Instruction:
Can you predict the next possible item that the user may expect?

### User Input:
The user has interacted with items <10_45_2>, <23_67_8>, <45_12_90> in chronological order. Can you predict the next possible item that the user may expect?

### Response:
<78_34_56>
```

#### Token化处理

```python
def pre(self, idx):
    # 1. Instruction部分 (固定)
    instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you predict the next possible item that the user may expect?

"""
    tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
    # tokens: [<BOS>, 39407, 374, 459, 7754, ...]

    # 2. User Input部分
    history = self.get_history(self.data.iloc[idx])
    prompt = self.generate_prompt(history)
    tokens += self.tokenizer.encode(prompt, bos=False, eos=False)
    # tokens: [<BOS>, ..., 791, 1217, 706, 16681, ..., 220, 605, ...]

    # 3. Response部分 (只有这部分参与loss计算)
    input_prompt_len = len(tokens)
    target_item = history['output']  # "<78_34_56>\n"
    golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
    # golden_tokens: [27, 2495, 62, 1958, 62, 3487, 29, <EOS>]

    tokens = tokens + golden_tokens

    # 4. 构建labels (-100表示不计算loss)
    labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
    # labels: [-100, -100, ..., -100, 27, 2495, 62, 1958, 62, 3487, 29, <EOS>]

    return {
        "input_ids": tokens[-self.max_len:],
        "attention_mask": [1] * len(tokens[-self.max_len:]),
        "labels": labels[-self.max_len:]
    }
```

**关键点**:
- ✅ 只有 `### Response:` 后的内容参与loss计算
- ✅ Instruction和User Input部分的labels全部设为-100
- ✅ 模型学习预测SID token序列

---

### 任务2: SID-物品对齐 (SidItemFeatDataset)

**目标**: 建立SID与物品标题之间的双向映射

**文件位置**: `data.py:876-1014`

#### 特点

这个数据集会创建**两种类型**的样本：

1. **Title → SID**: 给定物品标题，预测SID
2. **SID → Title**: 给定SID，预测物品标题

#### 数据构建流程

```python
class SidItemFeatDataset(Dataset):
    def __init__(self, item_file, index_file, tokenizer, ...):
        # 1. 加载物品特征和SID索引
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        # 2. 构建双向映射
        self.sid2title = {}
        self.title2sid = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # 拼接3层SID: ['<10_45_2>'] -> '<10_45_2>'
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    # 实际上sids[0]已经是完整的'<10_45_2>'格式
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid

        # 3. 创建样本 (每个物品产生2个样本)
        self.data = []

        # Type 1: SID → Title
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })

        # Type 2: Title → SID
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })
```

#### 样本示例

**Type 1: SID → Title**

```
### Instruction:
Answer the question about item identification.

### User Input:
What is the title of item "<10_45_2>"?

### Response:
iPhone 14 Pro Max
```

**Type 2: Title → SID**

```
### Instruction:
Answer the question about item identification.

### User Input:
Which item has the title: iPhone 14 Pro Max?

### Response:
<10_45_2>
```

#### Token化处理

```python
def pre(self, idx):
    # 1. Instruction
    instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question about item identification.

"""
    tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

    # 2. 根据任务类型生成prompt
    data_point = self.data[idx]

    if data_point['task'] == 'title2sid':
        prompt = f"Which item has the title: {data_point['input']}?"
        response = data_point['output']  # SID
    else:  # sid2title
        prompt = f'What is the title of item "{data_point["input"]}"?'
        response = data_point['output']  # Title

    full_prompt = f"""### User Input:
{prompt}

### Response:
"""
    tokens += self.tokenizer.encode(full_prompt, bos=False, eos=False)

    # 3. 添加response
    input_prompt_len = len(tokens)
    golden_tokens = self.tokenizer.encode(response + '\n', bos=False, eos=True)
    tokens = tokens + golden_tokens

    # 4. 构建labels
    labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

    return {
        "input_ids": tokens[-self.max_len:],
        "attention_mask": [1] * len(tokens[-self.max_len:]),
        "labels": labels[-self.max_len:]
    }
```

**关键点**:
- ✅ 每个物品产生2个训练样本
- ✅ 学习SID ⇄ Title的双向映射
- ✅ 帮助模型理解SID的语义含义

---

### 任务3: 融合序列推荐 (FusionSeqRecDataset)

**目标**: 从SID历史序列预测物品的自然语言描述

**文件位置**: `data.py:1475-1706`

#### 特点

这是最复杂的任务，结合了：
- SID序列（输入）
- 自然语言描述（输出）

有两种变体：
1. **预测标题**: 输出目标物品的title
2. **预测描述**: 输出目标物品的description

#### 数据构建流程

```python
class FusionSeqRecDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, ...):
        # 1. 加载数据
        self.data = pd.read_csv(train_file)
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        # 2. 构建SID到标题/描述的映射
        self.sid2title = {}
        self.sid2description = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']

                # 处理description
                processed_description = self._process_description(description, title)

                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description

    def _process_description(self, description, title):
        """
        处理描述字段的规则:
        1. 如果description为空 → 使用title
        2. 如果description是列表 → 选择最长的一个
        3. 如果列表中最长的也是空 → 使用title
        """
        if not description or description == '':
            return title

        if isinstance(description, list):
            non_empty = [d for d in description if d and d.strip()]
            if non_empty:
                return max(non_empty, key=len)  # 返回最长的描述
            else:
                return title

        return description if description.strip() else title
```

#### 样本生成

```python
def get_history(self, row):
    # 1. 获取历史SID
    history_item_sid = eval(row['history_item_sid'])
    history_str = ", ".join(history_item_sid)
    # 例如: "<10_45_2>, <23_67_8>, <45_12_90>"

    # 2. 获取目标物品的标题和描述
    target_sid = row['item_sid']

    if target_sid in self.sid2title:
        target_title = self.sid2title[target_sid]
    else:
        target_title = target_sid

    if target_sid in self.sid2description:
        target_description = self.sid2description[target_sid]
    else:
        target_description = f"An item with semantic ID {target_sid}"

    return {
        "history_str": history_str,
        "target_title": target_title,
        "target_description": target_description,
        "target_sid": target_sid
    }

def pre(self, idx):
    history = self.get_history(self.data.iloc[idx])

    # 随机选择任务类型 (50% title, 50% description)
    task_type = random.choice(['title', 'description'])

    if task_type == 'title':
        # 任务: 预测标题
        prompt = f"The user has sequentially interacted with items {history['history_str']}. Can you recommend the next item for him? Tell me the title of the item"
        target = history['target_title']
    else:
        # 任务: 预测描述
        prompt = f"Please review the user's historical interactions: {history['history_str']}, and describe what kind of item he still needs."
        target = history['target_description']

    # Token化处理 (与前两个任务类似)
    # ...
```

#### 样本示例

**Type 1: 预测标题**

```
### Instruction:
Can you recommend an item based on the user's history?

### User Input:
The user has sequentially interacted with items <10_45_2>, <23_67_8>, <45_12_90>. Can you recommend the next item for him? Tell me the title of the item

### Response:
MacBook Pro 16-inch
```

**Type 2: 预测描述**

```
### Instruction:
Can you recommend an item based on the user's history?

### User Input:
Please review the user's historical interactions: <10_45_2>, <23_67_8>, <45_12_90>, and describe what kind of item he still needs.

### Response:
A high-performance laptop with M2 Pro chip, 16-inch Liquid Retina XDR display, perfect for professional content creation and development.
```

#### Token化处理

```python
def pre(self, idx):
    # 1. Instruction
    instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you recommend an item based on the user's history?

"""
    tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

    # 2. 获取历史和目标
    history = self.get_history(self.data.iloc[idx])

    # 3. 随机选择任务类型
    task_type = random.choice(['title', 'description'])

    if task_type == 'title':
        prompt_text = self.generate_prompt_title(history['history_str'])
        target = history['target_title']
    else:
        prompt_text = self.generate_prompt_description(history['history_str'])
        target = history['target_description']

    # 4. 构建完整prompt
    full_prompt = f"""### User Input:
{prompt_text}

### Response:
"""
    tokens += self.tokenizer.encode(full_prompt, bos=False, eos=False)

    # 5. 添加target
    input_prompt_len = len(tokens)
    golden_tokens = self.tokenizer.encode(target + '\n', bos=False, eos=True)
    tokens = tokens + golden_tokens

    # 6. 构建labels
    labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

    return {
        "input_ids": tokens[-self.max_len:],
        "attention_mask": [1] * len(tokens[-self.max_len:]),
        "labels": labels[-self.max_len:]
    }
```

**关键点**:
- ✅ 输入是SID序列，输出是自然语言
- ✅ 随机选择预测title或description
- ✅ 学习SID → 语义的映射
- ✅ 帮助模型理解物品的内容

---

## Token化处理

### Tokenizer封装

```python
class Tokenizer:
    """统一的Tokenizer封装，处理BOS/EOS token"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        # 1. 基础编码
        t = self.tokenizer.encode(s)

        # 2. 移除默认的BOS/EOS
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        # 3. 按需添加BOS/EOS
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]

        return t
```

### Label Masking规则

在所有三个任务中，都遵循相同的label masking规则：

```python
# 1. Instruction + User Input → labels = -100 (不计算loss)
input_prompt_len = len(tokens_before_response)
labels = [-100] * input_prompt_len

# 2. Response部分 → labels = token_ids (计算loss)
labels = labels + tokens[input_prompt_len:]

# 完整示例:
# Tokens:  [<BOS>, 791, 1217, ..., 220, 605, ..., 27, 2495, <EOS>]
#          |-- Instruction --|-- Input --|-- Response --|
# Labels:  [-100, -100, ..., -100, -100, ..., 27, 2495, <EOS>]
```

**为什么这样设计？**
- Instruction和Input是**上下文**，不需要模型生成
- Response是**目标输出**，模型需要学习生成
- 只对Response部分计算loss，提高训练效率

---

## 完整示例

### 输入数据

**train.csv**:
```csv
user_id,history_item_id,history_item_title,history_item_sid,item_id,item_title,item_sid
U001,"['I001', 'I002', 'I003']","['iPhone 14', 'AirPods Pro', 'iPad Air']","['<10_45_2>', '<23_67_8>', '<45_12_90>']",I004,MacBook Pro,<78_34_56>
```

**.item.json**:
```json
{
  "I001": {"title": "iPhone 14", "description": "Latest iPhone model"},
  "I002": {"title": "AirPods Pro", "description": "Active noise cancellation"},
  "I003": {"title": "iPad Air", "description": "Powerful tablet"},
  "I004": {"title": "MacBook Pro", "description": "Professional laptop"}
}
```

**.index.json**:
```json
{
  "I001": ["<10_45_2>"],
  "I002": ["<23_67_8>"],
  "I003": ["<45_12_90>"],
  "I004": ["<78_34_56>"]
}
```

---

### 任务1样本: SID序列推荐

```
[Token IDs: 1, 791, 374, ..., 220]  <- Instruction (labels = -100)
[Token IDs: 791, 1217, 706, ...]     <- User Input (labels = -100)
[Token IDs: 27, 2495, 62, 1958, 2]  <- Response: <78_34_56> (labels = same as tokens)

完整文本:
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you predict the next possible item that the user may expect?

### User Input:
The user has interacted with items <10_45_2>, <23_67_8>, <45_12_90> in chronological order. Can you predict the next possible item that the user may expect?

### Response:
<78_34_56>
"
```

---

### 任务2样本: SID-标题对齐

**Sample 1: SID → Title**
```
完整文本:
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question about item identification.

### User Input:
What is the title of item "<78_34_56>"?

### Response:
MacBook Pro
"
```

**Sample 2: Title → SID**
```
完整文本:
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question about item identification.

### User Input:
Which item has the title: MacBook Pro?

### Response:
<78_34_56>
"
```

---

### 任务3样本: 融合推荐

**Sample 1: 预测标题**
```
完整文本:
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you recommend an item based on the user's history?

### User Input:
The user has sequentially interacted with items <10_45_2>, <23_67_8>, <45_12_90>. Can you recommend the next item for him? Tell me the title of the item

### Response:
MacBook Pro
"
```

**Sample 2: 预测描述**
```
完整文本:
"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Can you recommend an item based on the user's history?

### User Input:
Please review the user's historical interactions: <10_45_2>, <23_67_8>, <45_12_90>, and describe what kind of item he still needs.

### Response:
Professional laptop
"
```

---

## 训练流程总结

### 数据加载

```python
# 在 sft.py 中
from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset

# 1. 创建三个数据集
train_data1 = SidSFTDataset(
    train_file=train_file,
    tokenizer=tokenizer,
    max_len=cutoff_len,
    category=category
)

train_data2 = SidItemFeatDataset(
    item_file=item_meta_path,
    index_file=sid_index_path,
    tokenizer=tokenizer,
    max_len=cutoff_len,
    category=category
)

train_data3 = FusionSeqRecDataset(
    train_file=train_file,
    item_file=item_meta_path,
    index_file=sid_index_path,
    tokenizer=tokenizer,
    max_len=cutoff_len,
    category=category
)

# 2. 合并数据集
from torch.utils.data import ConcatDataset
train_dataset = ConcatDataset([train_data1, train_data2, train_data3])

# 3. 创建DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn
)
```

### 训练循环

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # batch包含来自三个任务的混合样本
        input_ids = batch['input_ids']        # [B, L]
        attention_mask = batch['attention_mask']  # [B, L]
        labels = batch['labels']              # [B, L]

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss  # 自动只计算labels != -100的位置

        # 反向传播
        loss.backward()
        optimizer.step()
```

---

## 设计理念

### 为什么需要三个任务？

1. **任务1 (SID → SID)**:
   - 专注于推荐能力
   - 学习物品之间的协同过滤模式
   - 类似传统推荐模型

2. **任务2 (SID ⇄ Title)**:
   - 建立语义对齐
   - 让模型理解SID的含义
   - 双向学习增强记忆

3. **任务3 (SID → Text)**:
   - 融合推荐和生成
   - 保留LLM的语言能力
   - 生成可解释的推荐

### 多任务学习的优势

```
单任务训练:
  SID序列 → 下一个SID
  ↓
  模型只学会模式匹配，不理解语义

多任务训练:
  SID序列 → 下一个SID  (任务1)
  SID ⇄ 标题           (任务2)
  SID序列 → 描述       (任务3)
  ↓
  模型同时学习:
  ✓ 推荐模式 (协同过滤)
  ✓ 语义理解 (SID含义)
  ✓ 语言生成 (世界知识)
```

---

## 任务2与任务3的联系与区别

### 快速对比

| 维度 | 任务2: 语义对齐 | 任务3: 融合生成 |
|------|----------------|----------------|
| **输入** | 单个SID 或 单个标题 | SID序列（用户历史） |
| **输出** | 标题 或 SID | 物品标题/描述 |
| **目标** | 建立1对1映射 | 序列推荐+生成 |
| **本质** | 字典查询 | 推理预测 |

### 任务2: 语义对齐（建立"翻译字典"）

**作用**: 让模型学会SID与标题的一一对应关系

```python
# 样本示例
输入: What is the title of item "<10_45_2>"?
输出: iPhone 14 Pro Max

# 模型学到的是"词汇表"
vocabulary = {
    "<10_45_2>": "iPhone 14 Pro Max",
    "<23_67_8>": "AirPods Pro",
    ...
}
```

**特点**:
- ✅ 静态映射（不涉及用户历史）
- ✅ 双向学习（SID ⇄ 标题）
- ✅ 像背单词

### 任务3: 融合生成（推理+生成）

**作用**: 从用户SID历史序列推理下一个物品的描述

```python
# 样本示例
输入: The user has sequentially interacted with items
      <10_45_2>, <23_67_8>, <45_12_90>.
      Can you recommend the next item?
输出: MacBook Pro - A powerful laptop for professionals

# 模型需要推理
if history == [iPhone, AirPods, iPad]:
    → 用户喜欢苹果生态
    → 预测: MacBook Pro
    → 生成: 自然语言描述
```

**特点**:
- ✅ 动态推理（分析用户历史）
- ✅ 单向生成（SID序列 → 文本）
- ✅ 像阅读理解+作文

### 协同关系

```
任务2（基础）        任务3（进阶）
   ↓                    ↓
学会SID含义    →    基于语义推理预测
   ↓                    ↓
提供"词汇"          进行"造句"
```

**信息流动**:
```
Step 1 (任务2):
<10_45_2> → 学习 → "iPhone 14"
<23_67_8> → 学习 → "AirPods Pro"
<45_12_90> → 学习 → "iPad Air"

Step 2 (任务3):
输入: [<10_45_2>, <23_67_8>, <45_12_90>]
  ↓ (利用任务2的知识理解)
理解: [iPhone, AirPods, iPad]
  ↓ (推理)
预测: "用户需要MacBook来完善生态"
```

**为什么两者都需要？**

| 配置 | 效果 |
|------|------|
| 只有任务2 | ✅ 知道SID含义<br>❌ 不会序列推理 |
| 只有任务3 | ✅ 能处理序列<br>❌ 不理解SID语义，只能模式匹配 |
| 任务2+3 | ✅ 理解语义<br>✅ 能推理<br>✅ 生成可解释的推荐 |

**形象比喻**:
- **任务2** = 教模型认字（学习词汇）
- **任务3** = 教模型写作文（理解+推理+生成）

---

## Label Masking机制详解

### 为什么需要-100？

在训练样本中：
```
### Instruction: Can you predict the next item?
### User Input: The user interacted with <10>, <23>, <45>
### Response: <78>
```

**问题**: 我们只希望模型学习生成"Response"部分，不希望它学习生成"Instruction"和"User Input"。

### -100的作用

**PyTorch约定**: `label=-100` 的位置在计算loss时会被自动忽略

```python
# 构建labels
labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

# 可视化
tokens: [BOS, tok1, tok2, ..., tok50, <27, 2495, 23, EOS>]
        |----Instruction----|--Input--|  |--Response---|
labels: [-100, -100, ...,     -100,     27, 2495, 23, EOS]
        |-------不计算loss-------|       |---计算loss---|
```

### 训练时发生什么？

```python
# Step 1: 模型看到完整输入
input_ids = [BOS, tok1, ..., tok50, 27, 2495, 23, EOS]
# 通过attention理解整个上下文

# Step 2: 只对Response部分计算loss
for position in range(len(tokens)):
    if labels[position] == -100:
        continue  # 跳过，不产生梯度
    else:
        loss += CrossEntropy(pred[position], labels[position])
        # 计算loss，产生梯度，更新权重

# 结果: 只有Response部分的参数得到更新
```

### 效果对比

**❌ 不使用-100 (所有位置都计算loss)**:
```
输入: "Below is an instruction..."
模型输出: "Below is an instruction that describes a task..."
        ↑ 在背诵模板！
```

**✅ 使用-100 (只对Response计算loss)**:
```
输入: "Below is an instruction... ### Response:\n"
模型输出: "<78>"
        ↑ 直接给出答案！
```

### 核心要点

1. **-100 = "不要学习这个位置"**
2. **模型仍能看到完整上下文**（attention_mask全是1）
3. **只有Response部分产生梯度**
4. **让模型专注于生成答案，而非复述问题**

---

## 常见问题

### Q1: 为什么labels中大部分是-100？

**A**: 这是PyTorch的CrossEntropyLoss的特殊设计：
```python
# 在计算loss时，会自动忽略label=-100的位置
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100  # 默认值
)
```

这样做的好处：
- 只对目标输出计算loss
- 避免浪费计算在上下文部分
- 提高训练效率

---

### Q2: 三个任务的比例如何控制？

**A**: 在代码中，三个数据集直接合并：
```python
train_dataset = ConcatDataset([train_data1, train_data2, train_data3])
```

比例取决于每个数据集的大小：
- `train_data1`: N个样本 (序列推荐)
- `train_data2`: 2M个样本 (M个物品 × 2个方向)
- `train_data3`: N个样本 (融合推荐)

**调整策略**:
```python
# 方法1: 采样控制
train_data2 = SidItemFeatDataset(..., sample=1000)  # 只用1000个样本

# 方法2: 重复数据集
from torch.utils.data import ConcatDataset
train_dataset = ConcatDataset([
    train_data1, train_data1,  # 重复2次
    train_data2,
    train_data3
])

# 方法3: 加权采样
from torch.utils.data import WeightedRandomSampler
weights = [1.0] * len(train_data1) + [0.5] * len(train_data2) + [1.0] * len(train_data3)
sampler = WeightedRandomSampler(weights, len(weights))
```

---

### Q3: 如何处理过长的序列？

**A**: 使用 `max_len` 参数截断：
```python
return {
    "input_ids": tokens[-self.max_len:],      # 保留最后max_len个token
    "attention_mask": attention_mask[-self.max_len:],
    "labels": labels[-self.max_len:]
}
```

**注意**:
- 从后往前截断，保留最近的历史
- 如果序列太长，可能会丢失一些历史信息
- 可以考虑增大 `max_len` (默认512，可设为1024或2048)

---

### Q4: 为什么要处理description为列表的情况？

**A**: Amazon数据集中，description字段可能有多种格式：

```json
// 格式1: 字符串
{"description": "This is a great product"}

// 格式2: 列表
{"description": ["Feature 1", "Feature 2", "Feature 3"]}

// 格式3: 空值
{"description": ""}
```

处理策略：
```python
def _process_description(self, description, title):
    if not description:
        return title  # 空值 → 用标题代替

    if isinstance(description, list):
        non_empty = [d for d in description if d.strip()]
        return max(non_empty, key=len) if non_empty else title  # 选最长的

    return description  # 字符串 → 直接返回
```

---

### Q5: SID的格式是如何确定的？

**A**: SID格式在RQ-VAE训练时确定：

```python
# 假设RQ-VAE配置: 3层，每层256个码本
num_emb_list = [256, 256, 256]

# 生成过程:
embedding = text_encoder.encode("iPhone 14 Pro Max")  # [768维]
indices = rqvae.get_indices(embedding)  # [layer1, layer2, layer3]
# 例如: [10, 45, 2]

# 转换为SID token:
sid = f"<{indices[0]}_{indices[1]}_{indices[2]}>"
# 结果: "<10_45_2>"
```

**格式规则**:
- 用 `<>` 包裹
- 层之间用 `_` 分隔
- 每层的值范围: 0 ~ (码本大小-1)

---

## 总结

MiniOneRec的训练样本组织非常精巧：

1. **三个互补任务**:
   - 推荐能力 (SID → SID)
   - 语义对齐 (SID ⇄ Text)
   - 融合生成 (SID → Description)

2. **统一的格式**:
   - Instruction-Input-Response结构
   - Label masking机制
   - 灵活的token化处理

3. **可扩展设计**:
   - 易于添加新任务
   - 支持多数据集混合
   - 方便调整任务权重

这种设计让模型既能做推荐，又能生成自然语言解释，实现了真正的生成式推荐系统！🚀

