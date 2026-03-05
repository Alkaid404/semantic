#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODE="${1:-full}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/dataset/PAN/pan25-generated-plagiarism-detection-train/01_train/01_train}"
TRUTH_DIR="${TRUTH_DIR:-$PROJECT_ROOT/dataset/PAN/pan25-generated-plagiarism-detection-train/01_train/01_train_truth}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/detections_train}"

echo "============================================"
echo "  PAN25 Train 基准评测"
echo "============================================"
echo "  模式:       $MODE"
echo "  数据目录:   $DATA_DIR"
echo "  真值目录:   $TRUTH_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "============================================"

DATA_DIR="$DATA_DIR" \
TRUTH_DIR="$TRUTH_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
"$SCRIPT_DIR/run_eval.sh" "$MODE"
