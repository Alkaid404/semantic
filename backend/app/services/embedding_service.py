"""Embedding 服务。

职责：
- 加载 sentence-transformers 模型
- 批量生成向量
- 向量 L2 归一化，以便后续使用 cosine 相似度
- 支持 IDF 加权编码：token 级嵌入 × IDF 权重 → 加权平均 pooling

参考：pan25-baseline/embeddings.py → compute_embeddings_with_llm()
"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..core import settings
from .idf_service import IDFService


class EmbeddingService:
    """Embedding 模型封装。"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self._model_name = model_name or settings.embedding_model_name
        self._device = device or settings.embedding_device
        self._model: SentenceTransformer | None = None

    # ---- 延迟加载 ----
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
        return self._model

    @property
    def tokenizer(self):
        """暴露底层分词器，供 IDFService.fit() 使用。"""
        model = self._load_model()
        return model.tokenizer

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """将文本列表编码为二维向量数组 (N, dim)。

        参数：
        - texts:     原始文本列表
        - normalize: 是否对向量进行 L2 归一化（默认是，便于 cosine 检索）
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def encode_idf(
        self,
        texts: list[str],
        idf: IDFService,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """IDF 加权编码：获取 token 级嵌入，按 IDF 权重做加权平均 pooling。

        步骤：
        1. tokenize → input_ids + attention_mask
        2. 通过 transformer backbone 得到 token_embeddings (N, seq_len, dim)
        3. 为每个 token 查 IDF 权重
        4. 加权平均 pooling：emb = Σ(w_i * h_i) / Σ(w_i)
        5. 可选 L2 归一化
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._load_model()
        device = model.device

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            # 分词
            encoded = model.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]  # (B, seq_len)
            attention_mask = encoded["attention_mask"]  # (B, seq_len)

            # 前向推理，取 token-level 输出
            with torch.no_grad():
                outputs = model[0].auto_model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                token_embeddings = outputs.last_hidden_state.cpu().numpy()
                # shape: (B, seq_len, dim)

            attn = attention_mask.numpy()  # (B, seq_len)
            ids = input_ids.numpy()  # (B, seq_len)

            for b in range(len(batch_texts)):
                seq_len = int(attn[b].sum())
                t_ids = ids[b, :seq_len].tolist()
                t_emb = token_embeddings[b, :seq_len, :]  # (seq_len, dim)

                # IDF 权重
                weights = idf.get_weights_for_tokens(t_ids)  # (seq_len,)
                # 加权平均 pooling
                w_sum = weights.sum()
                if w_sum > 0:
                    emb = (t_emb * weights[:, np.newaxis]).sum(axis=0) / w_sum
                else:
                    emb = t_emb.mean(axis=0)

                all_embeddings.append(emb)

        result = np.stack(all_embeddings, axis=0).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            result = result / norms

        return result

    @property
    def dimension(self) -> int:
        """返回向量维度。"""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
