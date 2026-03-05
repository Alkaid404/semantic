#!/usr/bin/env python3
"""PAN25 评估适配器。

将后端的 SimilarityEngine 接入 PAN25 评估流程：
1. 读取 pairs 文件获取文档配对
2. 对每对文档运行 SimilarityEngine 检测
3. 输出 PAN 格式检测 XML
4. 调用 eval.py 计算 Recall / Precision / Granularity / PlagDet Score

用法：
    python run_pan25_eval.py \
        --data-dir  ../pan25-generated-plagiarism-detection-spot-check/00_spot_check/00_spot_check \
        --truth-dir ../pan25-generated-plagiarism-detection-spot-check/00_spot_check/00_spot_check_truth \
        --output-dir ./detections_backend \
        --similarity-threshold 0.55 \
        --rerank-threshold 0.3 \
        --use-rerank

也可以跳过检测、只运行评估（复用已有检测结果）：
    python run_pan25_eval.py \
        --truth-dir  ... \
        --output-dir ./detections_backend \
        --eval-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from xml.etree import ElementTree as ET

# ── 把 backend/ 加入 sys.path，以便 import app.* ─────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

# eval.py 所在目录
BASELINE_DIR = os.path.join(PROJECT_ROOT, "clef25", "pan25-baseline")
sys.path.insert(0, BASELINE_DIR)

# ── Hugging Face 镜像（国内网络） ───────────────────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 数据集根目录 ────────────────────────────────────────────
DATASET_ROOT = os.environ.get(
    "PAN_DATASET_ROOT",
    os.path.join(PROJECT_ROOT, "dataset", "PAN"),
)

# 数据集路径映射：dataset_name -> (data_dir, truth_dir)
DATASET_PRESETS: dict[str, tuple[str, str]] = {
    "spot-check": (
        os.path.join(DATASET_ROOT, "pan25-generated-plagiarism-detection-spot-check",
                     "00_spot_check", "00_spot_check"),
        os.path.join(DATASET_ROOT, "pan25-generated-plagiarism-detection-spot-check",
                     "00_spot_check", "00_spot_check_truth"),
    ),
    "train": (
        os.path.join(DATASET_ROOT, "pan25-generated-plagiarism-detection-train",
                     "01_train", "01_train"),
        os.path.join(DATASET_ROOT, "pan25-generated-plagiarism-detection-train",
                     "01_train", "01_train_truth"),
    ),
    "validation": (
        os.path.join(DATASET_ROOT, "pan25-generated-plagiarism-detection-validation",
                     "02_validation", "02_validation"),
        os.path.join(DATASET_ROOT, "pan25-generated-plagiarism-detection-validation",
                     "02_validation", "02_validation_truth"),
    ),
}


# ═══════════════════════════════════════════════════════════════
# 1. 数据 I/O
# ═══════════════════════════════════════════════════════════════

def load_pairs(pairs_file: str) -> list[tuple[str, str]]:
    """从 pairs 文件中读取 (susp_doc, src_doc) 对。"""
    pairs: list[tuple[str, str]] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def read_text(path: str) -> str:
    """读取文本文件。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════
# 2. 生成检测 XML（PAN 格式）
# ═══════════════════════════════════════════════════════════════

def generate_detection_xml(
    matches: list,
    susp_doc_name: str,
    src_doc_name: str,
    output_dir: str,
) -> str | None:
    """根据 MatchResult 列表，生成 PAN 格式的检测 XML 文件。

    文件名格式与真值保持一致：
        {susp_base}-{src_base}.xml
    """
    susp_base = os.path.splitext(susp_doc_name)[0]
    src_base = os.path.splitext(src_doc_name)[0]

    root = ET.Element("document", reference=susp_doc_name)

    # about 元素
    ET.SubElement(
        root, "feature",
        name="about",
        authors="",
        title="",
        lang="en",
    )

    detection_count = 0
    for m in matches:
        ET.SubElement(
            root, "feature",
            name="plagiarism",
            type="paraphrase",
            this_language="en",
            this_offset=str(m.suspect_offset),
            this_length=str(m.suspect_length),
            source_reference=src_doc_name,
            source_language="en",
            source_offset=str(m.source_offset),
            source_length=str(m.source_length),
        )
        detection_count += 1

    # 写入文件
    output_file = os.path.join(output_dir, f"{susp_base}-{src_base}.xml")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    return output_file, detection_count


# ═══════════════════════════════════════════════════════════════
# 3. 运行检测
# ═══════════════════════════════════════════════════════════════

def run_detection(
    pairs: list[tuple[str, str]],
    data_dir: str,
    output_dir: str,
    *,
    similarity_threshold: float = 0.55,
    rerank_threshold: float = 0.3,
    top_k: int = 5,
    use_rerank: bool = True,
) -> dict:
    """对所有文档对运行后端检测引擎，生成检测 XML。"""
    # 延迟导入，以便在 --eval-only 模式下无需加载模型
    from app.services.similarity_engine import SimilarityEngine

    os.makedirs(output_dir, exist_ok=True)

    engine = SimilarityEngine()

    susp_dir = os.path.join(data_dir, "susp")
    src_dir = os.path.join(data_dir, "src")

    stats = {
        "total_pairs": len(pairs),
        "processed": 0,
        "skipped": 0,
        "total_detections": 0,
        "pair_details": [],
    }

    print(f"\n{'='*60}")
    print(f"后端检测引擎 PAN25 评估")
    print(f"{'='*60}")
    print(f"文档对数：   {len(pairs)}")
    print(f"相似度阈值：  {similarity_threshold}")
    print(f"Rerank 阈值： {rerank_threshold}")
    print(f"Top-K：      {top_k}")
    print(f"使用 Rerank：  {use_rerank}")
    print(f"输出目录：    {output_dir}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for idx, (susp_doc, src_doc) in enumerate(pairs, 1):
        susp_path = os.path.join(susp_dir, susp_doc)
        src_path = os.path.join(src_dir, src_doc)

        # 检查文件是否存在
        if not os.path.exists(susp_path) or not os.path.exists(src_path):
            print(f"  [{idx}/{len(pairs)}] 跳过 {susp_doc} — 文件缺失")
            stats["skipped"] += 1
            continue

        # 读取全文
        susp_text = read_text(susp_path)
        src_text = read_text(src_path)

        # 运行引擎
        t0 = time.time()
        result = engine.check(
            src_text,
            susp_text,
            similarity_threshold=similarity_threshold,
            rerank_threshold=rerank_threshold,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        elapsed = time.time() - t0

        # 生成检测 XML
        output_file, det_count = generate_detection_xml(
            result.matches, susp_doc, src_doc, output_dir,
        )

        stats["processed"] += 1
        stats["total_detections"] += det_count
        stats["pair_details"].append({
            "susp": susp_doc,
            "src": src_doc,
            "detections": det_count,
            "similarity": result.similarity,
            "time": elapsed,
        })

        print(
            f"  [{idx}/{len(pairs)}] {susp_doc} ↔ {src_doc} "
            f"| 检测数: {det_count:3d} | 相似度: {result.similarity:.4f} "
            f"| 耗时: {elapsed:.2f}s"
        )

    total_time = time.time() - t_start
    print(f"\n检测完成 — 总耗时 {total_time:.1f}s, "
          f"处理 {stats['processed']} 对, "
          f"跳过 {stats['skipped']} 对, "
          f"共 {stats['total_detections']} 条检测\n")

    return stats


# ═══════════════════════════════════════════════════════════════
# 4. 运行评估
# ═══════════════════════════════════════════════════════════════

def run_evaluation(
    truth_dir: str,
    detection_dir: str,
    *,
    micro: bool = False,
    plag_tag: str = "plagiarism",
    det_tag: str = "plagiarism",
) -> dict:
    """调用 eval.py 中的评估函数，同时计算 Micro 和 Macro 全部指标。"""
    from eval import (
        extract_annotations_from_files,
        macro_avg_recall_and_precision,
        micro_avg_recall_and_precision,
        granularity,
        plagdet_score,
    )

    print(f"\n{'='*60}")
    print(f"PAN25 评估")
    print(f"{'='*60}")
    print(f"真值目录：   {truth_dir}")
    print(f"检测目录：   {detection_dir}")
    print(f"{'='*60}\n")

    # 提取标注
    print("读取真值标注...")
    cases = extract_annotations_from_files(truth_dir, plag_tag)
    print(f"  真值标注数：{len(cases)}")

    print("读取检测结果...")
    detections = extract_annotations_from_files(detection_dir, det_tag)
    print(f"  检测结果数：{len(detections)}")

    if len(cases) == 0:
        print("\n⚠ 警告：真值标注为空，无法评估！")
        return {}

    # 计算全部指标
    print("计算中...")
    macro_rec, macro_prec = macro_avg_recall_and_precision(cases, detections)
    micro_rec, micro_prec = micro_avg_recall_and_precision(cases, detections)
    gran = granularity(cases, detections)
    macro_score = plagdet_score(macro_rec, macro_prec, gran)
    micro_score = plagdet_score(micro_rec, micro_prec, gran)

    # 打印完整结果
    print(f"\n{'═'*60}")
    print(f"  PAN25 评估结果")
    print(f"{'═'*60}")
    print(f"  真值标注数：{len(cases)}")
    print(f"  检测结果数：{len(detections)}")
    print(f"{'─'*60}")
    print(f"  {'指标':<20} {'Macro':>12} {'Micro':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12}")
    print(f"  {'Recall':<20} {macro_rec:>12.6f} {micro_rec:>12.6f}")
    print(f"  {'Precision':<20} {macro_prec:>12.6f} {micro_prec:>12.6f}")
    print(f"  {'Granularity':<20} {gran:>12.6f} {gran:>12.6f}")
    print(f"  {'PlagDet Score':<20} {macro_score:>12.6f} {micro_score:>12.6f}")
    print(f"{'═'*60}\n")

    return {
        "macro_plagdet": macro_score,
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "micro_plagdet": micro_score,
        "micro_recall": micro_rec,
        "micro_precision": micro_prec,
        "granularity": gran,
        "num_cases": len(cases),
        "num_detections": len(detections),
    }


# ═══════════════════════════════════════════════════════════════
# 5. 参数扫描（可选）
# ═══════════════════════════════════════════════════════════════

def run_param_sweep(
    pairs: list[tuple[str, str]],
    data_dir: str,
    truth_dir: str,
    base_output_dir: str,
    *,
    sim_thresholds: list[float] | None = None,
    rerank_thresholds: list[float] | None = None,
    use_rerank: bool = True,
) -> list[dict]:
    """在多组阈值上运行检测 + 评估，生成对比表格。"""
    if sim_thresholds is None:
        sim_thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
    if rerank_thresholds is None:
        rerank_thresholds = [0.0, 0.2, 0.3, 0.5]

    results: list[dict] = []

    for sim_t in sim_thresholds:
        rr_list = rerank_thresholds if use_rerank else [0.0]
        for rr_t in rr_list:
            tag = f"sim{sim_t:.2f}_rr{rr_t:.2f}"
            out_dir = os.path.join(base_output_dir, tag)

            print(f"\n{'#'*60}")
            print(f"# 参数组合: sim_threshold={sim_t}, rerank_threshold={rr_t}")
            print(f"{'#'*60}")

            run_detection(
                pairs, data_dir, out_dir,
                similarity_threshold=sim_t,
                rerank_threshold=rr_t,
                use_rerank=use_rerank,
            )
            eval_result = run_evaluation(truth_dir, out_dir)
            eval_result["sim_threshold"] = sim_t
            eval_result["rerank_threshold"] = rr_t
            results.append(eval_result)

    # 打印对比表格
    print(f"\n{'='*120}")
    print("参数扫描结果汇总")
    print(f"{'='*120}")
    print(f"{'Sim_T':>8} {'RR_T':>8} │ {'Ma_PlagDet':>11} {'Ma_Recall':>11} {'Ma_Prec':>11} │ "
          f"{'Mi_PlagDet':>11} {'Mi_Recall':>11} {'Mi_Prec':>11} │ {'Gran':>8}")
    print(f"{'─'*8} {'─'*8} │ {'─'*11} {'─'*11} {'─'*11} │ "
          f"{'─'*11} {'─'*11} {'─'*11} │ {'─'*8}")
    for r in results:
        print(
            f"{r.get('sim_threshold', 0):>8.2f} "
            f"{r.get('rerank_threshold', 0):>8.2f} │ "
            f"{r.get('macro_plagdet', 0):>11.6f} "
            f"{r.get('macro_recall', 0):>11.6f} "
            f"{r.get('macro_precision', 0):>11.6f} │ "
            f"{r.get('micro_plagdet', 0):>11.6f} "
            f"{r.get('micro_recall', 0):>11.6f} "
            f"{r.get('micro_precision', 0):>11.6f} │ "
            f"{r.get('granularity', 0):>8.6f}"
        )
    print(f"{'='*120}\n")

    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将后端 SimilarityEngine 接入 PAN25 评估体系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 数据集选择
    parser.add_argument(
        "--dataset",
        choices=["spot-check", "train", "validation"],
        default="spot-check",
        help="选择预设数据集：spot-check(50对) / train(62159对) / validation(7975对)（默认 spot-check）",
    )

    # 路径参数（可覆盖 --dataset 预设）
    parser.add_argument(
        "--data-dir",
        default=None,
        help="包含 pairs / src / susp 的数据目录（留空则使用 --dataset 预设）",
    )
    parser.add_argument(
        "--truth-dir",
        default=None,
        help="真值 XML 目录（留空则使用 --dataset 预设）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="检测结果 XML 输出目录（留空则自动按数据集命名）",
    )

    # 引擎参数
    parser.add_argument("--similarity-threshold", type=float, default=0.55,
                        help="Embedding 余弦相似度阈值（默认 0.55）")
    parser.add_argument("--rerank-threshold", type=float, default=0.3,
                        help="CrossEncoder rerank 阈值（默认 0.3）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="每个疑似段落保留的 top-k 候选数（默认 5）")
    parser.add_argument("--use-rerank", action="store_true", default=True,
                        help="是否启用 CrossEncoder 精排（默认启用）")
    parser.add_argument("--no-rerank", dest="use_rerank", action="store_false",
                        help="禁用 CrossEncoder 精排")

    # 评估参数
    parser.add_argument("--micro", action="store_true",
                        help="使用微平均（默认宏平均）")
    parser.add_argument("--plag-tag", default="plagiarism",
                        help="真值标注标签后缀（默认 plagiarism）")
    parser.add_argument("--det-tag", default="plagiarism",
                        help="检测标注标签后缀（默认 plagiarism）")

    # 运行模式
    parser.add_argument("--eval-only", action="store_true",
                        help="仅运行评估（复用已有检测结果）")
    parser.add_argument("--detect-only", action="store_true",
                        help="仅运行检测（不评估）")
    parser.add_argument("--sweep", action="store_true",
                        help="参数扫描模式：在多组阈值上运行并对比")

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析数据集路径：优先使用显式 --data-dir/--truth-dir，否则从 --dataset 预设
    preset_data, preset_truth = DATASET_PRESETS[args.dataset]
    data_dir = args.data_dir or preset_data
    truth_dir = args.truth_dir or preset_truth
    output_dir = args.output_dir or os.path.join(
        SCRIPT_DIR, f"detections_backend_{args.dataset}",
    )

    pairs_file = os.path.join(data_dir, "pairs")

    print(f"\n数据集:  {args.dataset}")
    print(f"数据目录: {data_dir}")
    print(f"真值目录: {truth_dir}")
    print(f"输出目录: {output_dir}\n")

    if args.sweep:
        # 参数扫描模式
        pairs = load_pairs(pairs_file)
        run_param_sweep(
            pairs, data_dir, truth_dir,
            output_dir,
            use_rerank=args.use_rerank,
        )
        return

    if not args.eval_only:
        # 运行检测
        pairs = load_pairs(pairs_file)
        run_detection(
            pairs, data_dir, output_dir,
            similarity_threshold=args.similarity_threshold,
            rerank_threshold=args.rerank_threshold,
            top_k=args.top_k,
            use_rerank=args.use_rerank,
        )

    if not args.detect_only:
        # 运行评估
        run_evaluation(
            truth_dir,
            output_dir,
            micro=args.micro,
            plag_tag=args.plag_tag,
            det_tag=args.det_tag,
        )


if __name__ == "__main__":
    main()
