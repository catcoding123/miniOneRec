#!/usr/bin/env python3
"""
查看和理解训练样本的工具
帮助理解三个数据集是如何构建的
"""

import pandas as pd
import json
import sys
from pathlib import Path

def view_dataset_samples(
    data_dir="./learn/mini_data",
    dataset_name="Industrial_and_Scientific",
    n_samples=3
):
    """查看各个数据集的样本"""

    data_path = Path(data_dir)

    # 加载文件
    train_file = data_path / "train.csv"
    index_file = data_path / f"{dataset_name}.index.json"
    item_file = data_path / f"{dataset_name}.item.json"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               MiniOneRec 数据样本查看工具                      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 1. 查看训练数据
    print(f"\n{'='*70}")
    print(f"📊 训练数据文件：{train_file}")
    print(f"{'='*70}")

    if not train_file.exists():
        print(f"❌ 文件不存在！请先运行：python learn/prepare_mini_data.py")
        return

    train_df = pd.read_csv(train_file)
    print(f"\n总样本数：{len(train_df)}")
    print(f"列名：{list(train_df.columns)}")

    print(f"\n展示前 {n_samples} 个样本：")
    print("-" * 70)

    for idx, row in train_df.head(n_samples).iterrows():
        print(f"\n【样本 #{idx}】")
        print(f"  用户ID: {row['user_id']}")
        print(f"  历史物品ID: {row['history_item_id']}")
        print(f"  历史物品标题: {row['history_item_title']}")
        print(f"  历史物品SID: {row['history_item_sid']}")
        print(f"  目标物品ID: {row['item_id']}")
        print(f"  目标物品标题: {row['item_title']}")
        print(f"  目标物品SID: {row['item_sid']}")
        print("-" * 70)

    # 2. 查看SID索引
    print(f"\n{'='*70}")
    print(f"🔑 SID索引文件：{index_file}")
    print(f"{'='*70}")

    with open(index_file, 'r') as f:
        index_data = json.load(f)

    print(f"\n总物品数：{len(index_data)}")
    print(f"\n前 {n_samples} 个物品的SID映射：")

    for i, (item_id, sid) in enumerate(list(index_data.items())[:n_samples]):
        print(f"  {item_id} → {sid}")

    # 3. 查看物品特征
    print(f"\n{'='*70}")
    print(f"📦 物品特征文件：{item_file}")
    print(f"{'='*70}")

    with open(item_file, 'r') as f:
        item_data = json.load(f)

    print(f"\n总物品数：{len(item_data)}")
    print(f"\n前 {n_samples} 个物品的详细信息：")

    for i, (item_id, item_info) in enumerate(list(item_data.items())[:n_samples]):
        print(f"\n  【物品 {i+1}: {item_id}】")
        print(f"    标题: {item_info.get('title', 'N/A')}")

        desc = item_info.get('description', 'N/A')
        if isinstance(desc, list):
            desc = desc[0] if desc else 'N/A'
        print(f"    描述: {desc[:100]}...")

        if 'category' in item_info:
            print(f"    类别: {item_info['category']}")
        if 'price' in item_info:
            print(f"    价格: ${item_info['price']}")

    # 4. 模拟三个训练任务的样本
    print(f"\n{'='*70}")
    print(f"🎯 三个训练任务的样本构建")
    print(f"{'='*70}")

    # 取第一个样本
    sample = train_df.iloc[0]
    history_sids = eval(sample['history_item_sid'])
    target_sid = sample['item_sid']
    target_title = sample['item_title']

    # 任务1：序列推荐
    print(f"\n【任务1：序列推荐 (SidSFTDataset)】")
    print(f"目标：从SID序列预测下一个SID")
    print(f"-" * 70)
    print(f"### Instruction:")
    print(f"Can you predict the next possible item that the user may expect?")
    print(f"\n### User Input:")
    history_str = ", ".join(history_sids)
    print(f"The user has interacted with items {history_str} in chronological order.")
    print(f"Can you predict the next possible item that the user may expect?")
    print(f"\n### Response:")
    print(f"{target_sid}")

    # 任务2：SID-标题对齐
    print(f"\n{'='*70}")
    print(f"【任务2：SID-标题对齐 (SidItemFeatDataset)】")
    print(f"目标：建立SID与标题的双向映射")
    print(f"-" * 70)

    print(f"\n▸ 方向1: Title → SID")
    print(f"### Instruction:")
    print(f"Answer the question about item identification.")
    print(f"\n### User Input:")
    print(f"Which item has the title: {target_title}?")
    print(f"\n### Response:")
    print(f"{target_sid}")

    print(f"\n▸ 方向2: SID → Title")
    print(f"### Instruction:")
    print(f"Answer the question about item identification.")
    print(f"\n### User Input:")
    print(f'What is the title of item "{target_sid}"?')
    print(f"\n### Response:")
    print(f"{target_title}")

    # 任务3：融合推荐
    print(f"\n{'='*70}")
    print(f"【任务3：融合推荐 (FusionSeqRecDataset)】")
    print(f"目标：从SID序列预测物品描述")
    print(f"-" * 70)

    print(f"\n▸ 变体1: 预测标题")
    print(f"### Instruction:")
    print(f"Can you recommend an item based on the user's history?")
    print(f"\n### User Input:")
    print(f"The user has sequentially interacted with items {history_str}.")
    print(f"Can you recommend the next item for him? Tell me the title of the item")
    print(f"\n### Response:")
    print(f"{target_title}")

    print(f"\n▸ 变体2: 预测描述")
    print(f"### Instruction:")
    print(f"Can you recommend an item based on the user's history?")
    print(f"\n### User Input:")
    print(f"Please review the user's historical interactions: {history_str},")
    print(f"and describe what kind of item he still needs.")
    print(f"\n### Response:")

    # 获取描述
    if sample['item_id'] in item_data:
        desc = item_data[sample['item_id']].get('description', target_title)
        if isinstance(desc, list) and desc:
            desc = max([d for d in desc if d], key=len) if any(desc) else target_title
        print(f"{desc[:200]}...")
    else:
        print(f"{target_title}")

    # 5. Label Masking示例
    print(f"\n{'='*70}")
    print(f"🔒 Label Masking机制说明")
    print(f"{'='*70}")
    print(f"""
在训练时，我们只希望模型学习生成 Response 部分，
不希望它学习生成 Instruction 和 User Input。

因此使用 -100 来标记不需要计算loss的位置：

Tokens:  [BOS, tok1, tok2, ..., tok_n, <response_tok1>, <response_tok2>, EOS]
         |--- Instruction ---|-- Input --||----- Response -----|
Labels:  [-100, -100, -100, ..., -100,   <response_tok1>, <response_tok2>, EOS]
         |-------- 不计算loss ---------|  |------- 计算loss --------|

PyTorch的CrossEntropyLoss会自动忽略label=-100的位置。
这样模型只学习生成答案，而不是背诵问题。
    """)

    print(f"\n{'='*70}")
    print(f"✅ 数据查看完成！")
    print(f"{'='*70}")
    print(f"""
💡 关键理解：
1. 三个任务使用同一份训练数据，但构建方式不同
2. 任务1专注推荐能力，任务2建立语义对齐，任务3融合生成
3. Label Masking确保只对目标输出计算loss
4. 多任务学习让模型同时具备推荐、理解、生成能力
    """)

if __name__ == "__main__":
    # 命令行参数
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./learn/mini_data"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "Industrial_and_Scientific"
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    view_dataset_samples(data_dir, dataset_name, n_samples)
