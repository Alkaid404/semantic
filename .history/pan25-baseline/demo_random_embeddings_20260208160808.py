import argparse
import os
import re
from typing import List

import numpy as np
from tqdm import tqdm


def paragraph_chunking(text: str) -> List[str]:
    """Split text into paragraphs and drop references section."""
    references_pattern = r"(?si)(?:\n\n+|^)(?:references|bibliography|reference list|works cited)(?:\n\n+.*)?$"
    text = re.sub(references_pattern, "", text.strip())
    paragraphs = re.split(r"\n\n(?!\s\n\s)", text)
    return [p for p in paragraphs if p.strip()]


def load_pairs(pairs_file: str) -> List[List[str]]:
    with open(pairs_file, "r", encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]


def generate_embeddings(paragraphs: List[str], dim: int, rng: np.random.RandomState) -> np.ndarray:
    if not paragraphs:
        return np.empty((0, dim), dtype=np.float32)
    emb = rng.normal(0.0, 1.0, size=(len(paragraphs), dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def compute_and_save(doc_list, doc_path, embeddings_path, dim, seed):
    os.makedirs(embeddings_path, exist_ok=True)
    for doc in tqdm(doc_list, desc=f"Embedding {os.path.basename(doc_path)}"):
        emb_file = os.path.join(embeddings_path, f"{doc}.npy")
        if os.path.exists(emb_file):
            continue
        with open(os.path.join(doc_path, doc), "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = paragraph_chunking(text)
        doc_seed = (hash(doc) + seed) % (2**32)
        rng = np.random.RandomState(doc_seed)
        embeddings = generate_embeddings(paragraphs, dim, rng)
        np.save(emb_file, embeddings)


def main():
    parser = argparse.ArgumentParser(description="Generate random embeddings for demo runs")
    parser.add_argument("--base_path", required=True, help="Dataset root with pairs, susp, src")
    parser.add_argument("--pairs_file", default=None, help="Pairs file path (default: base_path/pairs)")
    parser.add_argument("--output_dir", default="demo_embeddings_random", help="Output embeddings dir")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    pairs_file = args.pairs_file or os.path.join(args.base_path, "pairs")
    pairs = load_pairs(pairs_file)
    if not pairs:
        raise SystemExit("No valid pairs found")

    susp_docs = sorted(set(pair[0] for pair in pairs))
    src_docs = sorted(set(pair[1] for pair in pairs))

    embeddings_path = os.path.join(args.base_path, args.output_dir)
    compute_and_save(susp_docs, os.path.join(args.base_path, "susp"), embeddings_path, args.dim, args.seed)
    compute_and_save(src_docs, os.path.join(args.base_path, "src"), embeddings_path, args.dim, args.seed)

    print(f"Random embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
