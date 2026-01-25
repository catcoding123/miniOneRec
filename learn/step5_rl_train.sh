#!/bin/bash

# ============================================================================
# 步骤5：RL训练（强化学习优化）
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  步骤5：RL训练（强化学习优化）                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

DATASET="Industrial_and_Scientific"
SFT_MODEL=${1:-"./learn/output/sft"}
OUTPUT_DIR="./learn/output/rl"
DATA_DIR="./learn/mini_data"

# RL训练参数（针对mini数据集）
NUM_EPOCHS=2
LEARNING_RATE=1e-5
NUM_RETURN_SEQUENCES=4  # 每个prompt生成4个候选
BATCH_SIZE=4

echo "📝 RL训练配置："
echo "   SFT模型：$SFT_MODEL"
echo "   输出目录：$OUTPUT_DIR"
echo "   训练轮数：$NUM_EPOCHS"
echo "   候选数量：$NUM_RETURN_SEQUENCES"
echo "   学习率：$LEARNING_RATE"
echo ""

if [ ! -d "$SFT_MODEL" ]; then
    echo "❌ SFT模型不存在：$SFT_MODEL"
    echo "   请先完成SFT训练：bash learn/step3_sft_train.sh"
    exit 1
fi

echo "🚀 开始RL训练..."
echo ""
echo "⏳ 注意：RL训练涉及生成多个候选，速度较慢，请耐心等待..."
echo ""

python rl.py \
    --model_name_or_path "$SFT_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$DATA_DIR/train.csv" \
    --sid_index_path "$DATA_DIR/${DATASET}.index.json" \
    --item_meta_path "$DATA_DIR/${DATASET}.item.json" \
    --category "$DATASET" \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --max_length 512 \
    --save_steps 50 \
    --eval_steps 50 \
    --wandb_log False

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ RL训练完成！"
    echo ""
    echo "📂 模型保存在：$OUTPUT_DIR"
    echo ""
    echo "➡️  下一步："
    echo "   bash learn/step6_rl_evaluate.sh"
else
    echo ""
    echo "❌ RL训练失败！"
    exit 1
fi
