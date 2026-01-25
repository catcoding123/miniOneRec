#!/usr/bin/env python3
"""
准备一个小规模的demo数据集，用于快速学习和测试
从完整数据集中采样1000条数据
"""

import pandas as pd
import json
import shutil
from pathlib import Path

def prepare_mini_dataset(
    source_dataset="Industrial_and_Scientific",
    n_samples=1000,
    output_dir="./learn/mini_data"
):
    """
    从完整数据集中采样，创建mini版本

    Args:
        source_dataset: 源数据集名称
        n_samples: 采样数量
        output_dir: 输出目录
    """
    print(f"🎯 准备Mini数据集：{source_dataset}")
    print(f"📊 采样数量：{n_samples}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 源数据路径
    data_root = Path("./data/Amazon")
    train_file = data_root / "train" / f"{source_dataset}_5_2016-10-2018-11.csv"
    test_file = data_root / "test" / f"{source_dataset}_5_2016-10-2018-11.csv"
    valid_file = data_root / "valid" / f"{source_dataset}_5_2016-10-2018-11.csv"
    index_file = data_root / "index" / f"{source_dataset}.index.json"
    item_file = data_root / "index" / f"{source_dataset}.item.json"

    # 1. 采样训练数据
    print(f"\n📖 读取训练数据：{train_file}")
    train_df = pd.read_csv(train_file)
    print(f"   原始数据量：{len(train_df)}")

    # 随机采样
    mini_train_df = train_df.sample(n=min(n_samples, len(train_df)), random_state=42)
    print(f"   采样后数据量：{len(mini_train_df)}")

    # 保存mini训练集
    mini_train_file = output_path / "train.csv"
    mini_train_df.to_csv(mini_train_file, index=False)
    print(f"   ✅ 保存到：{mini_train_file}")

    # 2. 采样测试数据（200条）
    print(f"\n📖 读取测试数据：{test_file}")
    test_df = pd.read_csv(test_file)
    mini_test_df = test_df.sample(n=min(200, len(test_df)), random_state=42)
    mini_test_file = output_path / "test.csv"
    mini_test_df.to_csv(mini_test_file, index=False)
    print(f"   原始：{len(test_df)} → 采样：{len(mini_test_df)}")
    print(f"   ✅ 保存到：{mini_test_file}")

    # 3. 采样验证数据（100条）
    print(f"\n📖 读取验证数据：{valid_file}")
    valid_df = pd.read_csv(valid_file)
    mini_valid_df = valid_df.sample(n=min(100, len(valid_df)), random_state=42)
    mini_valid_file = output_path / "valid.csv"
    mini_valid_df.to_csv(mini_valid_file, index=False)
    print(f"   原始：{len(valid_df)} → 采样：{len(mini_valid_df)}")
    print(f"   ✅ 保存到：{mini_valid_file}")

    # 4. 收集涉及到的物品ID（转成字符串以匹配index.json的key）
    print(f"\n🔍 收集涉及的物品ID...")
    all_item_ids = set()

    for df in [mini_train_df, mini_test_df, mini_valid_df]:
        # 目标物品（转成字符串）
        all_item_ids.update([str(x) for x in df['item_id'].tolist()])

        # 历史物品（转成字符串）
        for history in df['history_item_id']:
            history_list = eval(history)
            all_item_ids.update([str(x) for x in history_list])

    print(f"   涉及物品数：{len(all_item_ids)}")

    # 5. 过滤index和item文件
    print(f"\n📦 处理SID索引文件...")
    with open(index_file, 'r') as f:
        full_index = json.load(f)

    mini_index = {k: v for k, v in full_index.items() if k in all_item_ids}
    mini_index_file = output_path / f"{source_dataset}.index.json"

    with open(mini_index_file, 'w') as f:
        json.dump(mini_index, f, indent=2)
    print(f"   原始物品：{len(full_index)} → 过滤后：{len(mini_index)}")
    print(f"   ✅ 保存到：{mini_index_file}")

    # 6. 过滤item特征文件
    print(f"\n📦 处理物品特征文件...")
    with open(item_file, 'r') as f:
        full_items = json.load(f)

    mini_items = {k: v for k, v in full_items.items() if k in all_item_ids}
    mini_item_file = output_path / f"{source_dataset}.item.json"

    with open(mini_item_file, 'w') as f:
        json.dump(mini_items, f, indent=2)
    print(f"   原始物品：{len(full_items)} → 过滤后：{len(mini_items)}")
    print(f"   ✅ 保存到：{mini_item_file}")

    # 7. 输出统计信息
    print(f"\n" + "="*60)
    print(f"✅ Mini数据集创建成功！")
    print(f"="*60)
    print(f"📂 输出目录：{output_path}")
    print(f"\n📊 数据统计：")
    print(f"   训练集：{len(mini_train_df)} 条")
    print(f"   测试集：{len(mini_test_df)} 条")
    print(f"   验证集：{len(mini_valid_df)} 条")
    print(f"   物品数：{len(mini_items)} 个")
    print(f"   唯一SID数：{len(mini_index)} 个")

    # 8. 展示几个样本
    print(f"\n📋 训练样本示例（前3条）：")
    print("-" * 60)
    for idx, row in mini_train_df.head(3).iterrows():
        print(f"\n样本 {idx}:")
        print(f"  用户ID: {row['user_id']}")
        print(f"  历史标题: {row['history_item_title'][:100]}...")
        print(f"  目标标题: {row['item_title']}")
        print(f"  历史SID: {row['history_item_sid'][:50]}...")
        print(f"  目标SID: {row['item_sid']}")

    return output_path

if __name__ == "__main__":
    import sys

    # 可以通过命令行参数指定数据集
    dataset = sys.argv[1] if len(sys.argv) > 1 else "Industrial_and_Scientific"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          MiniOneRec - Mini数据集准备工具                      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    prepare_mini_dataset(
        source_dataset=dataset,
        n_samples=n_samples
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  下一步：使用这个mini数据集进行训练                            ║
║  python learn/demo_sft.py                                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
