"""FAISS 向量检索服务。

职责：
- 管理向量索引
- 提供相似度检索

建议：
- 初期使用 IndexFlatL2
- 后期可升级 IVF 或 HNSW
"""
from __future__ import annotations


class FaissService:
    """FAISS 索引封装。"""

    def __init__(self):
        # TODO: 初始化索引并加载持久化数据
        pass

    def add_vectors(self, vectors):
        """向索引中添加向量。"""
        # TODO: 添加向量到索引
        pass

    def search(self, query_vectors, top_k: int = 5):
        """检索 top-k 相似向量。"""
        # TODO: 实现检索
        return []
