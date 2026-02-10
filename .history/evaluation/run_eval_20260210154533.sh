#!/bin/bash
# ──────────────────────────────────────────────────────────
# 运行后端检测引擎的 PAN25 评估
# ──────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认数据路径（spot check 数据集）
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/pan25-generated-plagiarism-detection-spot-check/00_spot_check/00_spot_check}"
TRUTH_DIR="${TRUTH_DIR:-$PROJECT_ROOT/pan25-generated-plagiarism-detection-spot-check/00_spot_check/00_spot_check_truth}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/detections_backend}"

# 引擎参数
SIM_THRESHOLD="${SIM_THRESHOLD:-0.55}"
RERANK_THRESHOLD="${RERANK_THRESHOLD:-0.3}"
TOP_K="${TOP_K:-5}"

echo "============================================"
echo "  PAN25 后端检测评估"
echo "============================================"
echo "  数据目录:   $DATA_DIR"
echo "  真值目录:   $TRUTH_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "  相似度阈值: $SIM_THRESHOLD"
echo "  Rerank阈值: $RERANK_THRESHOLD"
echo "============================================"

cd "$PROJECT_ROOT"

# 模式选择
MODE="${1:-full}"  # full | detect | eval | sweep

case "$MODE" in
  full)
    echo ">>> 完整模式：检测 + 评估"
    python "$SCRIPT_DIR/run_pan25_eval.py" \
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
    python "$SCRIPT_DIR/run_pan25_eval.py" \
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
    python "$SCRIPT_DIR/run_pan25_eval.py" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --eval-only
    ;;
  sweep)
    echo ">>> 参数扫描模式"
    python "$SCRIPT_DIR/run_pan25_eval.py" \
      --data-dir "$DATA_DIR" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --use-rerank \
      --sweep
    ;;
  no-rerank)
    echo ">>> 无 Rerank 模式"
    python "$SCRIPT_DIR/run_pan25_eval.py" \
      --data-dir "$DATA_DIR" \
      --truth-dir "$TRUTH_DIR" \
      --output-dir "${OUTPUT_DIR}_no_rerank" \
      --similarity-threshold "$SIM_THRESHOLD" \
      --no-rerank
    ;;
  *)
    echo "用法: $0 {full|detect|eval|sweep|no-rerank}"
    echo ""
    echo "  full      - 运行检测 + 评估（默认）"
    echo "  detect    - 仅运行检测，生成 XML"
    echo "  eval      - 仅运行评估（复用已有检测）"
    echo "  sweep     - 参数扫描，对比不同阈值"
    echo "  no-rerank - 不使用 CrossEncoder 精排"
    echo ""
    echo "环境变量："
    echo "  DATA_DIR          数据目录"
    echo "  TRUTH_DIR         真值目录"
    echo "  OUTPUT_DIR        输出目录"
    echo "  SIM_THRESHOLD     相似度阈值 (默认 0.55)"
    echo "  RERANK_THRESHOLD  Rerank 阈值 (默认 0.3)"
    echo "  TOP_K             Top-K (默认 5)"
    exit 1
    ;;
esac

echo ""
echo "完成！"
