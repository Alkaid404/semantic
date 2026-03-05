"""Embedding 服务。

职责：
- 加载 sentence-transformers 模型
- 批量生成向量
- 向量 L2 归一化，以便后续使用 cosine 相似度

参考：pan25-baseline/embeddings.py → compute_embeddings_with_llm()
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ..core import settings


class EmbeddingService:
    """Embedding 模型封装。"""

    def __init__(self, model_name: str | None = None, device: str | None = None, prompt: str | None = None):
        self._model_name = model_name or settings.embedding_model_name
        self._device = device or settings.embedding_device
        self._prompt = prompt if prompt is not None else settings.embedding_prompt
        self._model: SentenceTransformer | None = None

    # ---- 延迟加载 ----
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """将文本列表编码为二维向量数组 (N, dim)。

        参数：
        - texts:     原始文本列表
        - normalize: 是否对向量进行 L2 归一化（默认是，便于 cosine 检索）
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        # e5 系列模型需要对输入文本加 "query: " 前缀
        prompt = self._prompt
        if prompt:
            texts = [prompt + t for t in texts]

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        """返回向量维度。"""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
