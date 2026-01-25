#!/bin/bash

# ============================================================================
# 步骤4：评估SFT模型
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  步骤4：评估SFT模型                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

DATASET="Industrial_and_Scientific"
MODEL_PATH=${1:-"/root/autodl-tmp/minionerec/sft"}
DATA_DIR="./learn/mini_data"
TEST_FILE="$DATA_DIR/test.csv"

echo "📊 评估配置："
echo "   模型路径：$MODEL_PATH"
echo "   测试数据：$TEST_FILE"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在：$MODEL_PATH"
    echo "   请先完成SFT训练：bash learn/step3_sft_train.sh"
    exit 1
fi

echo "🚀 开始评估..."
echo ""

python evaluate.py \
    --base_model_path "$MODEL_PATH" \
    --test_data "$TEST_FILE" \
    --sid_index_path "$DATA_DIR/${DATASET}.index.json" \
    --item_meta_path "$DATA_DIR/${DATASET}.item.json" \
    --category "$DATASET" \
    --batch_size 16 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 评估完成！"
    echo ""
    echo "📈 关键指标："
    echo "   - HR@10: 命中率（Top-10推荐中包含真实物品的比例）"
    echo "   - NDCG@10: 归一化折损累积增益（考虑排名位置）"
    echo "   - CC: 约束检查（应该为0，表示所有生成的SID都有效）"
    echo ""
    echo "➡️  下一步（可选）："
    echo "   bash learn/step5_rl_train.sh"
else
    echo ""
    echo "❌ 评估失败！"
    exit 1
fi
