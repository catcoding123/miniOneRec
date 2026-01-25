#!/bin/bash

# ============================================================================
# 步骤6：评估RL模型并对比SFT
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  步骤6：评估RL模型并对比SFT                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

DATASET="Industrial_and_Scientific"
SFT_MODEL="./learn/output/sft"
RL_MODEL=${1:-"./learn/output/rl"}
DATA_DIR="./learn/mini_data"
TEST_FILE="$DATA_DIR/test.csv"

echo "📊 评估配置："
echo "   SFT模型：$SFT_MODEL"
echo "   RL模型：$RL_MODEL"
echo "   测试数据：$TEST_FILE"
echo ""

if [ ! -d "$RL_MODEL" ]; then
    echo "❌ RL模型不存在：$RL_MODEL"
    echo "   请先完成RL训练：bash learn/step5_rl_train.sh"
    exit 1
fi

# 评估RL模型
echo "🚀 评估RL模型..."
echo ""

python evaluate.py \
    --base_model_path "$RL_MODEL" \
    --test_data "$TEST_FILE" \
    --sid_index_path "$DATA_DIR/${DATASET}.index.json" \
    --item_meta_path "$DATA_DIR/${DATASET}.item.json" \
    --category "$DATASET" \
    --batch_size 16 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ RL模型评估完成！"
    echo ""
    echo "📊 性能对比："
    echo ""
    echo "请对比SFT和RL的指标："
    echo "   - HR@10的提升（RL应该比SFT更高）"
    echo "   - NDCG@10的提升（RL应该比SFT更高）"
    echo "   - CC应该都为0（所有生成都有效）"
    echo ""
    echo "💡 理解："
    echo "   RL通过GRPO算法，基于推荐指标直接优化，"
    echo "   因此通常能比纯SFT获得更好的推荐效果。"
    echo ""
    echo "🎉 完整流程体验完成！"
else
    echo ""
    echo "❌ 评估失败！"
    exit 1
fi
