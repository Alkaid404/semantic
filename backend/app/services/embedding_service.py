"""Embedding 服务。

职责：
- 加载 embedding 模型
- 批量生成向量

建议：
- 支持 GPU 推理
- 批量编码提高吞吐
"""
from __future__ import annotations

import numpy as np


class EmbeddingService:
    """Embedding 模型封装。"""

    def __init__(self):
        # TODO: 加载模型权重
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为二维向量数组。"""
        # TODO: 实现批量编码
        return np.empty((0, 0), dtype=np.float32)
