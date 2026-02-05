# LLM 深度学习笔记

> 本笔记记录深入学习LLM过程中确认的知识点和共识

## 目录

### 1. Transformer基础
- [ ] Attention机制
- [ ] 归一化（LN/BN/RMS Norm）
- [ ] 位置编码（Position Embedding/ALiBi/RoPE）
- [ ] 优化器
- [ ] 激活函数
- [ ] MHA与线性Attention

### 2. LLM核心机制
- [ ] Tokenization
- [ ] LoRA
- [ ] RAG
- [ ] 多模态对齐
- [ ] 推理Pipeline

### 3. 后训练（Post-training）
- [ ] SFT/DPO/PPO/GRPO/DAPO
- [ ] Reward设计
- [ ] GRPO深度分析
- [ ] 重要性采样与On-policy

---

## 1. Transformer基础

### Attention机制与位置编码

#### Self-Attention的排列不变性

**核心问题**：Self-Attention只看"内容相似度"，不看"位置关系"

```python
输入：["I", "love", "AI"]  vs  ["AI", "love", "I"]

Self-Attention计算（无位置编码）：
  score = Q·K^T  # 只基于内容的点积相似度
  output = softmax(score)·V

结果：相同token得到相同的输出表示，无论位置在哪
```

**数学定义**：
```
∀排列π, Attention(X) 经过π后 = Attention(π(X))
```

---

#### 位置编码的作用

**目的1：区分同token在不同位置的embedding**

```python
"I love I"
- 没有PE：两个"I"的embedding完全相同
- 有PE：第1个"I" = emb_I + PE_pos0
       第3个"I" = emb_I + PE_pos2
       → 向量不同，可区分
```

**目的2：让模型理解token的顺序关系**

```python
"I love AI"  vs  "AI love I"
- 内容token相同，但顺序不同 → 语义完全不同
- 位置编码让模型知道"I在前AI在后"还是"AI在前I在后"
```

**一句话**：位置编码给每个位置的token一个唯一的"位置ID"，打破Self-Attention的排列不变性。

---

#### Multi-Head Attention (MHA)

**设计目的**：并行化 + 子空间专门化 + 隐式正则化

```python
单头512维：在高维空间学习一个attention模式
8头×64维：强制每个head在低维子空间工作，成为"某方面的专家"

计算量：几乎相同（并行不是为了加速，是为了多样性）
```

---

**Head差异化机制：隐式实现，非显式约束**

核心机制：
```python
output = W^O · Concat(head1, head2, ..., head8)

W^O学习"如何组合8个head"：
  - 如果head1和head2学到相似的pattern
    → W^O学会"只用其中一个"（另一个权重接近0）
    → 被忽略的head梯度变小
    → 该head被迫学习其他模式（否则对loss无贡献）

  - 类似Self-Attention内部：
    每个token的Q查询所有K → 产生不同的attention分布
    如果所有token的分布都一样 → 信息冗余 → loss无法最优
    → 梯度驱动差异化

结论：不是显式正交化约束，而是loss优化自然驱动的差异化
```

---

**关键问题：为什么训练不会坍缩？**

理论风险：8个head随机初始化 → 梯度下降 → 会不会收敛到同一个解？

实际不会坍缩的三个原因：

**1. 随机初始化的对称性破缺**
```python
head1初始化：W^Q_1 ~ N(0, 0.02)
head2初始化：W^Q_2 ~ N(0, 0.02)

分布相同，但数值不同
→ 初始attention pattern不同
→ 梯度更新路径分叉
→ 收敛到不同局部最优
```

**2. W^O的选择性强化机制**
```python
W^O动态调整每个head的贡献：
  - head_i学到独特pattern → W^O权重大 → 梯度大 → 该pattern被强化
  - head_j冗余 → W^O权重小 → 梯度小 → 更新慢

head_j的分化过程（随机探索 + 选择性强化）：
  1. 初期：head_j和head_i学到类似pattern
     → W^O抑制head_j（权重小）
     → 梯度小，更新慢

  2. 探索：SGD随机性 + 不同mini-batch数据
     → head_j在参数空间缓慢"游走"

  3. 发现：某次碰巧学到新pattern（如主谓关系）
     → W^O发现head_j有独特贡献
     → W^O权重增大 → 梯度增大
     → 新pattern被强化保留

  4. 如果一直冗余：
     → W^O权重持续小 → 成为"死head"
     （这也是后来MQA/GQA简化的原因）

关键：不是主动"被迫"，而是随机探索 + 选择性放大
依赖：初始化差异 + SGD随机性 + 数据多样性

最终形成"分工"（部分head）：
  head1: 句法依赖
  head2: 语义相似
  head3: 位置邻近
  head4-8: 可能冗余或学到其他细微pattern
```

**3. 非凸优化的多样性**
```
Attention参数空间是非凸的 → 存在多个局部最优
不同初始化 + 不同随机梯度 → 到达不同局部最优 → 自然多样化
```

**类比**：
```
8个专家解决同一个问题：
  - 如果都用同一种方法 → 信息冗余 → 团队效率低
  - 评委（W^O）会发现："你们贡献都一样，我只需要1个人"
  → 其他7个人被迫找新方法（否则被淘汰）
  → 最终形成8种互补的解决方案
```

**一句话**：MHA通过隐式机制（W^O的组合学习 + 随机初始化 + 非凸优化）驱动不同head学习互补的attention pattern，避免坍缩到单一模式。

---

#### LayerNorm：为什么NLP必须用LN？

**核心作用：让相似度计算变成纯"方向匹配"，消除"幅度干扰"**

**Attention中的关键问题**：

```python
Self-Attention: Q·K^T 计算相似度

不归一化的问题：
  Token1 "apple":  [1, 2, 3]      (模长 ≈ 3.74)
  Token2 "fruit":  [10, 20, 30]   (模长 ≈ 37.4, 方向与apple完全相同)
  Token3 "banana": [1, 2, 4]      (模长 ≈ 4.58, 方向与apple略不同)

  点积：
    apple · fruit  = 140  ← 被模长放大10倍
    apple · banana = 17

  Softmax([140, 17]) ≈ [1.0, 0.0]

  问题：虽然banana语义也相似，但被fruit的大模长掩盖了

LayerNorm后：
  所有向量归一化到单位球面，模长≈1

  点积 ≈ cosine相似度（纯方向）：
    apple_norm · fruit_norm  = 1.0
    apple_norm · banana_norm = 0.996

  Softmax([1.0, 0.996]) ≈ [0.51, 0.49]

  结果：权重真正反映语义相似度，不被幅度误导
```

**BatchNorm vs LayerNorm**：

```python
输入：(Batch, Seq_Len, Hidden_Dim) = (32, 128, 768)

BatchNorm：
  在batch维度归一化（对每个特征维度）
  → 假设不同样本的同一特征应该有相似统计
  → CV适用（同一像素位置跨图片有意义）
  → NLP不适用（不同句子的"位置1"语义完全不同）

LayerNorm：
  在特征维度归一化（对每个样本独立）
  → 假设同一个token的768维特征有统计关系
  → NLP适用（每个token的embedding归一化）
  → 序列长度可变，不需要padding
```

**学习空间的简化**：

```python
不归一化：
  模型需要同时学习：
    1. 方向（语义）
    2. 模长（重要性/置信度）
  → 优化空间：三维空间的任意点（角度 + 距离）

LayerNorm后：
  模型只需学习方向，模长被固定为1
  → 优化空间：单位球面上的点（只需角度）

  类比：
    在球面上找最优点（2个自由度）
    vs
    在三维空间找最优点（3个自由度）

  结果：优化更简单，收敛更快更稳定
```

**为什么NLP必须用LN**：
1. Attention依赖相似度计算（方向匹配）
2. 优化空间简化（球面约束）
3. 梯度稳定（统一尺度）
4. 推理鲁棒（不依赖训练统计量）

---

#### RMSNorm：简化版LayerNorm

**简化内容**：

```python
LayerNorm（完整）：
  1. mean = (x1 + ... + x_n) / n
  2. var = ((x1-mean)² + ... + (x_n-mean)²) / n
  3. x_norm = (x - mean) / sqrt(var + ε)
  4. output = γ * x_norm + β

RMSNorm（简化）：
  1. rms = sqrt((x1² + ... + x_n²) / n)
  2. x_norm = x / rms
  3. output = γ * x_norm

去掉：❌ mean计算  ❌ 偏置β参数
保留：✓ 缩放γ参数
```

**为什么去掉mean不影响效果？数值验证**：

```python
正常场景（偏移较小）：
  token1 "apple": [10, 20, 30, 40]
  token2 "fruit": [15, 25, 35, 45]
  token3 "car":   [5, 10, 50, 60]

  Attention权重差异：
    LayerNorm:  [fruit: 0.5479, car: 0.4521]
    RMSNorm:    [fruit: 0.5376, car: 0.4624]
    差异：~1%，相对排序一致 ✓

极端场景（偏移很大）：
  token1: [100, 101, 102, 103]  mean=101.5
  token3: [0, 1, 2, 3]          mean=1.5
  （方向完全相同，但偏移差100）

  相似度：
    LayerNorm:  4.0000（完全消除偏移）
    RMSNorm:    3.2333（保留了偏移信息）

  Attention权重：
    LayerNorm:  [0.50, 0.50]
    RMSNorm:    [0.68, 0.32]
    差异：明显，但实际训练中很少遇到
```

**RMSNorm能work的原因**：

**1. 真实embedding偏移小**
```python
预训练LLM的embedding：
  - 初始化：N(0, 0.02)，天然centered
  - 训练后：优化过程倾向mean≈0
  → 不会出现[100,101,102,103]这样的极端偏移
  → LayerNorm和RMSNorm差异<1-2%
```

**2. Softmax的平滑性**
```python
相似度差异1-2% → Softmax后权重差异更小
→ 对最终的Attention输出影响微弱
```

**3. 计算效率的巨大提升**
```python
LayerNorm：需要两次遍历（mean + var）
RMSNorm：只需一次遍历（平方和）

计算量：RMSNorm省30-40%
内存访问：减少一半
```

**工程权衡**：
```
精度损失：~1-2%（正常场景）
速度提升：~30-40%
现代LLM：LLaMA/Qwen/Mistral全部采用

结论：精度损失可接受，效率提升显著
```

**一句话**：RMSNorm去掉mean和β，在真实场景中偏移较小导致精度损失<2%，而计算速度提升30-40%，是工程上的优秀权衡。

---

#### 残差连接（Residual Connection）

**数学定义**：
```python
传统网络：y = F(x)         # 直接学习映射
残差网络：y = x + F(x)     # 学习残差（差值）
```

**为什么叫"残差"？**

来源于统计学：残差 = 目标值 - 当前值 = 需要修正的量

```python
回归问题：
  真实值 y_true = 100
  预测值 y_pred = 98
  残差 = 2（还差多少）

残差网络：
  输入 x = 98（当前估计）
  F(x) 学习的是"还差多少" = 2
  输出 y = x + F(x) = 98 + 2 = 100

不是"从头学整个目标"，而是"在原有基础上修正"
```

**Transformer中的残差**：
```python
y = x + Attention(x)

x：token的当前表示（来自embedding或上一层）
Attention(x)：学习的是"需要添加的上下文信息"（修正量）
y：融合后的表示

含义：不是重建整个表示，而是学习"需要加上什么"
```

**梯度保护机制（核心）**：
```python
y = x + F(x)

求梯度：
  ∂y/∂x = ∂(x + F(x))/∂x
        = 1 + ∂F(x)/∂x
          ↑
     恒等项，保证梯度畅通

即使 ∂F(x)/∂x → 0（子层梯度消失）
总梯度至少还有"1" → ∂L/∂x ≥ ∂L/∂y

类比：高速公路（残差，梯度=1）+ 地面道路（子层）
      即使地面堵车，高速保证通行
```

---

#### Pre-Norm vs Post-Norm

**两种架构**：

```python
Post-Norm（原始Transformer, 2017）：
  x1 = x + Attention(x)          # 先残差
  x2 = LayerNorm(x1)             # 后归一化

Pre-Norm（现代主流：GPT-3/LLaMA/Qwen）：
  x1 = LayerNorm(x)              # 先归一化
  x2 = x + Attention(x1)         # 后残差
```

**关键差异：残差的"恒等项1"在LayerNorm的内外**

**Post-Norm的梯度**：
```python
y = LayerNorm(x + Attention(x))
              ↑
          残差在LN里面

∂y/∂x = ∂LN/∂(x+attn) · [1 + ∂attn/∂x]
        ↑
    必须经过LN的导数

问题：
  即使有恒等项"1"，但外面包了LayerNorm
  → 梯度被LN的导数缩放
  → 深层累积：∂LN_24 · ∂LN_23 · ... · ∂LN_1
  → 梯度消失

为什么原始Transformer（6层）能work？
  6层：梯度衰减^6，还可忍受
  24层+：梯度衰减^24，基本消失
```

**Pre-Norm的梯度**：
```python
y = x + Attention(LayerNorm(x))
    ↑           ↑
  残差      LN在里面

∂y/∂x = 1 + ∂Attention/∂LN · ∂LN/∂x
        ↑
   纯净的恒等项，不经过LN

优势：
  即使 ∂Attention/∂LN · ∂LN/∂x → 0（经过LN衰减）
  梯度至少还有：∂L/∂y · 1 = ∂L/∂y

  深层网络（24层+）：
    最底层梯度 ≈ 最顶层梯度（高速路畅通）
    → 训练稳定
```

**为什么现代LLM都用Pre-Norm？**

| 指标 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| **浅层网络（6层）** | ✓ 能训练 | ✓ 能训练 |
| **深层网络（24层+）** | ✗ 梯度消失 | ✓ 稳定 |
| **初始化敏感性** | 高（需要warmup） | 低（鲁棒） |
| **学习率** | 需要小心调整 | 更宽容 |

**一句话**：Pre-Norm的残差"恒等项1"在LayerNorm外面，保证梯度至少以∂L/∂y的幅度直达浅层，避免深层网络的梯度消失。

---

