"""IDF 加权服务。

职责：
- 基于当前文档对（source + suspect）的所有 chunk 构建 IDF 词汇表
- 提供 token → IDF 权重的映射
- 与 EmbeddingService 配合，在 token 级别对向量进行 IDF 加权后再 pooling

原理：
  IDF(t) = log( N / (1 + df(t)) )
  其中 N 为 chunk 总数，df(t) 为包含 token t 的 chunk 数量。
  IDF 值越高表示该词越稀有，在相似度计算中应被更重视。
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from transformers import PreTrainedTokenizerBase


class IDFService:
    """基于 chunk 语料的 IDF 权重管理器。"""

    def __init__(self):
        # token_id → IDF 值
        self._idf: dict[int, float] = {}
        self._default_idf: float = 1.0  # 未出现 token 的默认权重
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        chunks: list[str],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 512,
    ) -> "IDFService":
        """根据 chunk 列表构建 IDF 词表。

        参数：
        - chunks:     所有参与计算的 chunk 文本
        - tokenizer:  与 embedding 模型配套的分词器
        - max_length: 分词截断长度
        """
        n = len(chunks)
        if n == 0:
            self._fitted = True
            return self

        # 统计 document frequency — 每个 token 在多少个 chunk 中出现
        df: Counter[int] = Counter()

        for text in chunks:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            unique_ids = set(encoding["input_ids"])
            for tid in unique_ids:
                df[tid] += 1

        # 计算 IDF
        self._idf = {
            tid: math.log(n / (1.0 + freq))
            for tid, freq in df.items()
        }

        # 对 IDF 值做 min-max 归一化到 [0.1, 1.0]，避免极端权重
        if self._idf:
            vals = list(self._idf.values())
            min_v, max_v = min(vals), max(vals)
            span = max_v - min_v
            if span > 0:
                self._idf = {
                    tid: 0.1 + 0.9 * (v - min_v) / span
                    for tid, v in self._idf.items()
                }
            else:
                # 所有 IDF 值相同，均设为 1.0
                self._idf = {tid: 1.0 for tid in self._idf}

        self._default_idf = 0.1  # 完全未出现过的 token 给最低权重
        self._fitted = True
        return self

    def get_weights_for_tokens(self, token_ids: list[int]) -> np.ndarray:
        """返回与 token_ids 对应的 IDF 权重向量 (shape: len(token_ids),)。"""
        return np.array(
            [self._idf.get(tid, self._default_idf) for tid in token_ids],
            dtype=np.float32,
        )
