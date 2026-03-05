"""核心配置与共享工具。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_device() -> str:
    """自动检测可用设备：优先 CUDA，否则 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Settings:
    """全局配置项。"""

    # ---- Embedding 模型 ----
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL",
        # "sentence-transformers/all-MiniLM-L6-v2"       # 22M, 384维
        "sentence-transformers/all-mpnet-base-v2"      # 110M, 768维
        # "intfloat/e5-large-v2"                           # 335M, 1024维
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", _default_device())

    # e5 系列模型要求对输入文本加前缀
    # 对称任务（相似度比对）统一用 "query: "；留空则不加前缀
    embedding_prompt: str = os.getenv("EMBEDDING_PROMPT", "query: ")

    # ---- Cross-encoder 精排模型 ----
    rerank_model_name: str = os.getenv(
        "RERANK_MODEL",
        # "cross-encoder/ms-marco-MiniLM-L-6-v2"         # 22M
        "cross-encoder/ms-marco-MiniLM-L-12-v2"        # 33M
        # "BAAI/bge-reranker-v2-m3"                        # 568M，多语言
    )
    # 默认 512，可兼容 ms-marco-MiniLM 系列上限并控制显存占用。
    # 如需更大长度可通过环境变量覆盖，实际会在服务层再按 tokenizer 上限裁剪。
    rerank_max_length: int = int(os.getenv("RERANK_MAX_LENGTH", "512"))
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "64"))

    # ---- 相似度阈值 ----
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

    # ---- Rerank 阈值（sigmoid 归一化后的概率值，0~1） ----
    rerank_threshold: float = float(os.getenv("RERANK_THRESHOLD", "0.55"))

    # ---- FAISS 检索 top-k ----
    faiss_top_k: int = int(os.getenv("FAISS_TOP_K", "5"))

    # ---- 向量存储路径 ----
    vector_store_dir: str = os.getenv(
        "VECTOR_STORE_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "vector_store"),
    )

    # ---- 段落切分 ----
    # 是否使用 pan25 风格的段落切分（按空行分段）；
    # 若为 False 则使用滑动窗口切分
    use_paragraph_chunking: bool = True
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "128"))


settings = Settings()
