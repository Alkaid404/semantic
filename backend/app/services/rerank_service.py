"""Cross-encoder 重排服务。

职责：
- 对候选对进行重排，提高查准率
- 使用 cross-encoder 对 (source_chunk, suspect_chunk) 打分
- 解决仅靠 embedding 召回产生的"语义接近但不抄袭"误判

目的：
- 修复仅靠 embedding 召回产生的误判
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from ..core import settings


class RerankService:
    """Cross-encoder 重排器封装。"""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.rerank_model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        pairs: list[tuple[str, str]],
        threshold: float | None = None,
    ) -> list[tuple[int, float]]:
        """对 (source_chunk, suspect_chunk) 对列表进行重排。

        参数：
        - pairs: [(source_text, suspect_text), ...]
        - threshold: 过滤阈值，低于此分数的对会被丢弃

        返回：
        - [(原始索引, cross-encoder 分数), ...] — 已按分数降序排列，
          并过滤掉低于阈值的结果。
        """
        if not pairs:
            return []

        if threshold is None:
            threshold = settings.rerank_threshold

        model = self._load_model()
        raw_scores = model.predict(pairs, show_progress_bar=False)
        if isinstance(raw_scores, np.ndarray):
            raw_scores = raw_scores.tolist()

        # MS-MARCO cross-encoder 输出的是原始 logits（范围约 -11~+11），
        # 需要通过 sigmoid 映射到 [0, 1] 概率区间
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        scores = [_sigmoid(s) for s in raw_scores]

        results = [
            (idx, score)
            for idx, score in enumerate(scores)
            if score >= threshold
        ]
        # 分数降序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
