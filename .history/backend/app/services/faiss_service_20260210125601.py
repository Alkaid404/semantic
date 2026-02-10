"""FAISS 向量检索服务。

职责：
- 管理向量索引
- 使用 IndexFlatIP（内积 = cosine，在向量归一化后）进行检索
- 支持持久化存储

参考：pan25-baseline/cosine_baseline.py → compute_similarity_matrix()
"""
from __future__ import annotations

import os

import faiss
import numpy as np

from ..core import settings


class FaissService:
    """FAISS 索引封装。

    因为 EmbeddingService 默认输出 L2 归一化的向量，
    此处使用 IndexFlatIP（内积）等价于 cosine 相似度。
    """

    def __init__(self, dimension: int | None = None):
        self._dimension = dimension
        self._index: faiss.IndexFlatIP | None = None

    # ---- 初始化 / 重建索引 ----

    def _ensure_index(self, dim: int) -> faiss.IndexFlatIP:
        if self._index is None or self._index.d != dim:
            self._index = faiss.IndexFlatIP(dim)
            self._dimension = dim
        return self._index

    def reset(self) -> None:
        """清空索引。"""
        self._index = None

    # ---- 写入 ----

    def add_vectors(self, vectors: np.ndarray) -> None:
        """向索引中添加向量。

        vectors: shape (N, dim), dtype float32, 须已归一化。
        """
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return
        index = self._ensure_index(vectors.shape[1])
        index.add(vectors.astype(np.float32))

    # ---- 检索 ----

    def search(
        self, query_vectors: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """检索 top-k 相似向量。

        返回：
        - scores: (N_query, top_k) 相似度
        - indices: (N_query, top_k) 对应索引
        """
        if self._index is None or self._index.ntotal == 0:
            empty = np.empty((query_vectors.shape[0], 0), dtype=np.float32)
            return empty, empty.astype(np.int64)

        if top_k is None:
            top_k = settings.faiss_top_k
        # 保证 top_k 不超过索引中的向量数
        top_k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(
            query_vectors.astype(np.float32), top_k
        )
        return scores, indices

    # ---- 直接计算全量相似度矩阵（适合小规模场景） ----

    @staticmethod
    def cosine_similarity_matrix(
        query_vectors: np.ndarray, corpus_vectors: np.ndarray
    ) -> np.ndarray:
        """计算 query 与 corpus 之间的余弦相似度矩阵。

        与 pan25-baseline 中 compute_similarity_matrix 等价，
        但因向量已归一化，直接做内积即可。
        """
        return query_vectors @ corpus_vectors.T

    # ---- 持久化 ----

    def save(self, path: str | None = None) -> None:
        if self._index is None:
            return
        path = path or os.path.join(settings.vector_store_dir, "faiss.index")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self._index, path)

    def load(self, path: str | None = None) -> None:
        path = path or os.path.join(settings.vector_store_dir, "faiss.index")
        if os.path.exists(path):
            self._index = faiss.read_index(path)
            self._dimension = self._index.d
