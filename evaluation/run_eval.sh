#!/bin/bash
# ──────────────────────────────────────────────────────────
# 运行后端检测引擎的 PAN25 评估
# ──────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 数据集根目录 ───────────────────────────────────────────
PAN_DATASET_ROOT="${PAN_DATASET_ROOT:-$PROJECT_ROOT/dataset/PAN}"

# ── Hugging Face 镜像（国内网络） ───────────────────────────
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

# ── 数据集选择: spot-check | train | validation ──────────
DATASET="${DATASET:-spot-check}"   # 第二个参数可覆盖，见下方

# 如果有第二个命令行参数，用作数据集
if [ -n "$2" ]; then
  DATASET="$2"
fi

# 根据 DATASET 设置路径
case "$DATASET" in
  spot-check)
    _DATA_DIR="$PAN_DATASET_ROOT/pan25-generated-plagiarism-detection-spot-check/00_spot_check/00_spot_check"
    _TRUTH_DIR="$PAN_DATASET_ROOT/pan25-generated-plagiarism-detection-spot-check/00_spot_check/00_spot_check_truth"
    ;;
  train)
    _DATA_DIR="$PAN_DATASET_ROOT/pan25-generated-plagiarism-detection-train/01_train/01_train"
    _TRUTH_DIR="$PAN_DATASET_ROOT/pan25-generated-plagiarism-detection-train/01_train/01_train_truth"
    ;;
  validation)
    _DATA_DIR="$PAN_DATASET_ROOT/pan25-generated-plagiarism-detection-validation/02_validation/02_validation"
    _TRUTH_DIR="$PAN_DATASET_ROOT/pan25-generated-plagiarism-detection-validation/02_validation/02_validation_truth"
    ;;
  *)
    echo "错误: 未知数据集 '$DATASET'，可选: spot-check | train | validation"
    exit 1
    ;;
esac

# 允许环境变量覆盖
DATA_DIR="${DATA_DIR:-$_DATA_DIR}"
TRUTH_DIR="${TRUTH_DIR:-$_TRUTH_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/detections_backend_${DATASET}}"

# 引擎参数
SIM_THRESHOLD="${SIM_THRESHOLD:-0.55}"
RERANK_THRESHOLD="${RERANK_THRESHOLD:-0.3}"
TOP_K="${TOP_K:-5}"

echo "============================================"
echo "  PAN25 后端检测评估"
echo "============================================"
echo "  数据集:     $DATASET"
echo "  数据目录:   $DATA_DIR"
echo "  真值目录:   $TRUTH_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "  相似度阈值: $SIM_THRESHOLD"
echo "  Rerank阈值: $RERANK_THRESHOLD"
echo "  Top-K:      $TOP_K"
echo "  HF镜像:     $HF_ENDPOINT"
echo "============================================"

cd "$PROJECT_ROOT"

# 模式选择
MODE="${1:-full}"  # full | detect | eval | sweep

case "$MODE" in
  full)
    echo ">>> 完整模式：检测 + 评估"
    uv run python "$SCRIPT_DIR/run_pan25_eval.py" \
      --data-dir "$DATA_DIR" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --similarity-threshold "$SIM_THRESHOLD" \
      --rerank-threshold "$RERANK_THRESHOLD" \
      --top-k "$TOP_K" \
      --use-rerank
    ;;
  detect)
    echo ">>> 仅检测模式"
    uv run python "$SCRIPT_DIR/run_pan25_eval.py" \
      --data-dir "$DATA_DIR" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --similarity-threshold "$SIM_THRESHOLD" \
      --rerank-threshold "$RERANK_THRESHOLD" \
      --top-k "$TOP_K" \
      --use-rerank \
      --detect-only
    ;;
  eval)
    echo ">>> 仅评估模式（复用已有检测结果）"
    uv run python "$SCRIPT_DIR/run_pan25_eval.py" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --eval-only
    ;;
  sweep)
    echo ">>> 参数扫描模式"
    uv run python "$SCRIPT_DIR/run_pan25_eval.py" \
      --data-dir "$DATA_DIR" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --use-rerank \
      --sweep
    ;;
  no-rerank)
    echo ">>> 无 Rerank 模式"
    uv run python "$SCRIPT_DIR/run_pan25_eval.py" \
      --data-dir "$DATA_DIR" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "${OUTPUT_DIR}_no_rerank" \
      --similarity-threshold "$SIM_THRESHOLD" \
      --no-rerank
    ;;
  *)
    echo "用法: $0 <模式> [数据集]"
    echo ""
    echo "模式:"
    echo "  full      - 运行检测 + 评估（默认）"
    echo "  detect    - 仅运行检测，生成 XML"
    echo "  eval      - 仅运行评估（复用已有检测）"
    echo "  sweep     - 参数扫描，对比不同阈值"
    echo "  no-rerank - 不使用 CrossEncoder 精排"
    echo ""
    echo "数据集 (第二个参数，默认 spot-check):"
    echo "  spot-check  - Spot check 数据集 (50 对)"
    echo "  train       - 训练集 (62159 对)"
    echo "  validation  - 验证集 (7975 对)"
    echo ""
    echo "示例:"
    echo "  $0 full spot-check    # 在 spot-check 上完整评估"
    echo "  $0 full validation    # 在验证集上完整评估"
    echo "  $0 sweep train        # 在训练集上参数扫描"
    echo ""
    echo "环境变量："
    echo "  PAN_DATASET_ROOT  数据集根目录 (默认 $PROJECT_ROOT/dataset/PAN)"
    echo "  DATA_DIR          数据目录 (覆盖预设)"
    echo "  TRUTH_DIR         真值目录 (覆盖预设)"
    echo "  OUTPUT_DIR        输出目录 (覆盖预设)"
    echo "  HF_ENDPOINT       HuggingFace 镜像地址 (默认 https://hf-mirror.com)"
    echo "  SIM_THRESHOLD     相似度阈值 (默认 0.55)"
    echo "  RERANK_THRESHOLD  Rerank 阈值 (默认 0.3)"
    echo "  TOP_K             Top-K (默认 5)"
    exit 1
    ;;
esac

echo ""
echo "完成！"
