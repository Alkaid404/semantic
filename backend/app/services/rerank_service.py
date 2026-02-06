"""Cross-encoder 重排服务。

职责：
- 对候选对进行重排，提高查准率

目的：
- 修复仅靠 embedding 召回产生的误判
"""
from __future__ import annotations


class RerankService:
    """Cross-encoder 重排器封装。"""

    def __init__(self):
        # TODO: 加载 cross-encoder 模型
        pass

    def rerank(self, pairs):
        """对 (source_chunk, suspect_chunk) 进行重排。"""
        # TODO: 实现重排逻辑
        return []
