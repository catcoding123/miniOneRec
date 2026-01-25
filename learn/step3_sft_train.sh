#!/bin/bash

# ============================================================================
# 步骤3：SFT训练（监督微调）
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  步骤3：SFT训练（监督微调）                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 未检测到GPU！此步骤需要GPU支持。"
    exit 1
fi

echo "🔍 检查GPU状态："
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# 配置
DATASET="Industrial_and_Scientific"
BASE_MODEL=${1:-"./learn/qwen2.5-1.5B-local"}  # 本地模型路径
OUTPUT_DIR="/root/autodl-tmp/minionerec/sft"
DATA_DIR="./learn/mini_data"

# 训练参数（针对mini数据集优化）
BATCH_SIZE=8
MICRO_BATCH_SIZE=4
NUM_EPOCHS=3
LEARNING_RATE=3e-4
MAX_LEN=512

echo "📝 训练配置："
echo "   基础模型：$BASE_MODEL"
echo "   输出目录：$OUTPUT_DIR"
echo "   训练轮数：$NUM_EPOCHS"
echo "   批次大小：$BATCH_SIZE"
echo "   学习率：$LEARNING_RATE"
echo ""

# 检查模型是否存在
if [ ! -d "$BASE_MODEL" ]; then
    echo "⚠️  模型路径不存在：$BASE_MODEL"
    echo "   请先下载模型，或指定正确的模型路径"
    echo "   例如：bash learn/step3_sft_train.sh /path/to/model"
    exit 1
fi

echo "🚀 开始SFT训练..."
echo ""

# 运行训练（使用Python直接调用，方便调试）
python sft.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$DATA_DIR/train.csv" \
    --eval_file "$DATA_DIR/valid.csv" \
    --sid_index_path "$DATA_DIR/${DATASET}.index.json" \
    --item_meta_path "$DATA_DIR/${DATASET}.item.json" \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --cutoff_len $MAX_LEN \
    --category "$DATASET"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SFT训练完成！"
    echo ""
    echo "📂 模型保存在：$OUTPUT_DIR"
    echo ""
    echo "🔍 查看训练日志："
    echo "   cat $OUTPUT_DIR/train.log"
    echo ""
    echo "➡️  下一步："
    echo "   bash learn/step4_sft_evaluate.sh"
else
    echo ""
    echo "❌ SFT训练失败！请检查错误信息。"
    exit 1
fi
