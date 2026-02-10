import os
import numpy as np
from vllm import LLM, SamplingParams
import re
from tqdm import tqdm
import time
import json
import argparse
from typing import List, Union

def get_gpu_count():
    """从 SLURM 环境变量读取可用 GPU 数量。"""
    gpus = os.getenv('SLURM_GPUS')
    if gpus is not None:
        return int(gpus)
    return 1  # 非 SLURM 环境默认使用 1 张 GPU

def paragraph_chunking(text):
    """按段落切分文本，并剔除参考文献段落。"""

    # 先移除 references/bibliography 等尾部内容
    references_pattern = r"(?si)(?:\n\n+|^)(?:references|bibliography|reference list|works cited)(?:\n\n+.*)?$"
    text = re.sub(references_pattern, "", text.strip())

    # 以空行切段，避免把公式块拆散
    paragraphs = re.split(r"\n\n(?!\s\n\s)", text)

    # 过滤空段落
    paragraphs = [p for p in paragraphs if p.strip()]

    return paragraphs


def compute_embeddings_with_llm(paragraphs: List[str], model: Union[str, LLM]) -> np.ndarray:
    """使用 vLLM 模型为段落列表生成嵌入向量。"""
    embeddings = []
    
    print(f"Computing embeddings for {len(paragraphs)} paragraphs...")

    outputs = model.embed(paragraphs)
    for output in outputs:
        embeddings.append(output.outputs.embedding)

    # vLLM 输出已经是向量列表，这里直接转为数组
    print("Embedding computation complete.")
    return np.array(embeddings)


def compute_and_save_embeddings(doc_list, doc_path, embeddings_path, llm_model):
    """为文档列表逐个生成嵌入并保存为 .npy 文件。"""
    print(f"Computing embeddings for {len(doc_list)} documents...")

    skipped_docs = []

    for doc in tqdm(doc_list, desc="Embedding documents"):
        doc_path_full = os.path.join(doc_path, doc)
        emb_file = os.path.join(embeddings_path, f"{doc}.npy")

        if os.path.exists(emb_file):
            print(f"Skipping {doc} because embeddings already exist")
            continue

        with open(doc_path_full, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = paragraph_chunking(text)
        try:
            # 生成段落嵌入并写盘
            embeddings = compute_embeddings_with_llm(paragraphs, llm_model)
            np.save(emb_file, embeddings)
            print(f"Saved embeddings for {doc} to {emb_file}")
        except Exception as e:
            # 出错时跳过该文档，继续处理后续文档
            print(f"Error embedding {doc}: {e}, skipping document because of error...")
            skipped_docs.append(doc)
            continue

    print(f"Skipped {len(skipped_docs)} documents: {skipped_docs}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate embeddings for documents using LLM')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', 
                       help='LLM model to use for generating embeddings.')
    parser.add_argument('--pairs_path', type=str, default='./pan25-plag_detect_test/03_test/pairs',
                       help='Path to the pairs file.')
    parser.add_argument('--src_path', type=str, default='./pan25-plag_detect_test/src',
                       help='Path to the source documents directory.')
    parser.add_argument('--susp_path', type=str, default='./pan25-plag_detect_test/susp',
                       help='Path to the suspicious documents directory.')
    parser.add_argument('--base_path', type=str, default='./pan25-plag_detect_test',
                       help='Base path for the dataset.')
    parser.add_argument('--train_or_test', type=str, default='test',
                    help='Whether to use the train or test set.')
    args = parser.parse_args()


    # 组装文件路径
    pairs_file = args.pairs_path
    src_path = args.src_path
    susp_path = args.susp_path
    embeddings_path = os.path.join(
        args.base_path, args.train_or_test + "_embeddings_" + args.model.replace("/", "_")
    )
    os.makedirs(embeddings_path, exist_ok=True)

    # 读取配对文件
    try:
        with open(pairs_file, "r", encoding="utf-8") as f:
            all_pairs = [line.strip().split() for line in f.readlines() if line.strip()]
            all_pairs = [p for p in all_pairs if len(p) == 2]
        print(f"Loaded {len(all_pairs)} pairs from {pairs_file}")
        if not all_pairs:
            print("Error: No valid pairs found. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: Pairs file not found at {pairs_file}. Exiting.")
        return
    except Exception as e:
        print(f"Error reading pairs file {pairs_file}: {e}. Exiting.")
        return

    # 汇总唯一文档列表
    susp_docs = sorted(set(pair[0] for pair in all_pairs))
    src_docs = sorted(set(pair[1] for pair in all_pairs))
    print(f"Unique suspicious docs: {len(susp_docs)}")
    print(f"Unique source docs: {len(src_docs)}")

    # 生成并保存所有文档的嵌入
    start_embedding_time = time.time()
    print("\n--- Computing and Saving Embeddings ---")
    
    # 初始化 vLLM 模型
    print(f"\n--- Initializing vLLM for embeddings ---")
    gpu_count = get_gpu_count()
    print(f"Using {gpu_count} GPUs for tensor parallelism")
    llm_model = LLM(model=args.model, tensor_parallel_size=gpu_count, task="embed", max_seq_len_to_capture=32768, enforce_eager=True)
    print(f"vLLM initialized with model: {args.model}")

    compute_and_save_embeddings(susp_docs, susp_path, embeddings_path, llm_model)
    compute_and_save_embeddings(src_docs, src_path, embeddings_path, llm_model)

    end_embedding_time = time.time()
    print(
        f"Embedding computation completed in {end_embedding_time - start_embedding_time:.2f} seconds"
    )

    # 保存运行摘要到 JSON
    summary = {
        "total_susp_docs": len(susp_docs),
        "total_src_docs": len(src_docs),
        "total_time_seconds": end_embedding_time - start_embedding_time,
        "embedding_model": args.model,
        "embeddings_directory": embeddings_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_file = os.path.join(embeddings_path, "embedding_summary.json")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved embedding summary to {summary_file}")
    except Exception as e:
        print(f"Error saving summary to {summary_file}: {e}")

    print("Embedding computation complete.")


if __name__ == "__main__":
    main()
