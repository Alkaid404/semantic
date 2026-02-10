"""核心配置与共享工具。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """全局配置项。"""

    # ---- Embedding 模型 ----
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ---- Cross-encoder 精排模型 ----
    rerank_model_name: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ---- 相似度阈值 ----
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

    # ---- Rerank 阈值 ----
    rerank_threshold: float = float(os.getenv("RERANK_THRESHOLD", "0.3"))

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
