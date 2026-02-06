"""相似度引擎（系统大脑）。

职责：
- 串联 chunk -> embed -> recall -> rerank -> score

查重率建议：
- matched_chunks / total_source_chunks
- 可升级为加权 token 覆盖率
"""
from __future__ import annotations

from .chunker import chunk_text
from .embedding_service import EmbeddingService
from .faiss_service import FaissService
from .rerank_service import RerankService


class SimilarityEngine:
    """完整流程编排器。"""

    def __init__(self):
        self._embedder = EmbeddingService()
        self._faiss = FaissService()
        self._reranker = RerankService()

    def check_plagiarism(self, source: str, suspects: list[str]):
        """运行完整流程并返回相似度结果。"""
        # 流程：chunk -> embed -> recall -> rerank -> score
        _ = chunk_text(source)
        # TODO: 实现完整流程与评分计算
        return {"similarity": 0.0, "matches": []}
