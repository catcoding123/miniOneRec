#!/bin/bash

# ============================================================================
# 步骤1：准备Mini数据集
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  步骤1：准备Mini数据集（1000条训练样本）                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 数据集选择（可选：Industrial_and_Scientific 或 Office_Products）
DATASET=${1:-"Industrial_and_Scientific"}
N_SAMPLES=${2:-1000}

echo "📊 数据集：$DATASET"
echo "📈 采样数量：$N_SAMPLES"
echo ""

# 运行准备脚本
python learn/prepare_mini_data.py "$DATASET" "$N_SAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 数据准备完成！"
    echo ""
    echo "📂 数据位置：./learn/mini_data/"
    echo ""
    echo "🔍 查看数据样本："
    echo "   python learn/view_samples.py"
    echo ""
    echo "➡️  下一步："
    echo "   bash learn/step2_view_data.sh"
else
    echo ""
    echo "❌ 数据准备失败！"
    exit 1
fi
