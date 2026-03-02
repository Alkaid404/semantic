"""相似度引擎（系统大脑）。

职责：
- 串联 chunk → embed → recall → rerank → score
- 对单个可疑文档与源文档进行全流程比对
- 返回匹配段落、偏移量和整体相似度

流程对应 pan25-baseline/cosine_baseline.py:
  1. paragraph_chunking → 段落切分（带偏移）
  2. compute_embeddings  → 编码
  3. cosine_similarity   → 构建相似度矩阵
  4. threshold filter    → 收集检测对
  5. generate_xml        → 输出 PAN XML（可选）

本引擎在此基础上增加了 FAISS 召回 + CrossEncoder 精排两阶段。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import settings
from .chunker import Chunk, chunk_text
from .embedding_service import EmbeddingService
from .faiss_service import FaissService
from .rerank_service import RerankService


@dataclass
class MatchResult:
    """一个匹配对的完整信息。"""

    source_chunk: str
    suspect_chunk: str
    score: float  # 最终得分（rerank 后）
    cosine_score: float  # embedding 余弦相似度
    source_offset: int
    source_length: int
    suspect_offset: int
    suspect_length: int


@dataclass
class PlagiarismResult:
    """查重引擎的完整返回。"""

    similarity: float  # 文档级相似度 (0~1)
    matched_char_ratio: float  # 源文档被匹配到的字符比例
    matches: list[MatchResult]


class SimilarityEngine:
    """完整流程编排器。"""

    def __init__(
        self,
        embedder: EmbeddingService | None = None,
        faiss_svc: FaissService | None = None,
        reranker: RerankService | None = None,
    ):
        self._embedder = embedder or EmbeddingService()
        self._faiss = faiss_svc or FaissService()
        self._reranker = reranker or RerankService()

    # ------------------------------------------------------------------
    # 核心流程
    # ------------------------------------------------------------------

    def check(
        self,
        source_text: str,
        suspect_text: str,
        *,
        similarity_threshold: float | None = None,
        rerank_threshold: float | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> PlagiarismResult:
        """对一对 (source, suspect) 文档执行全流程查重。

        流程：
        1. 段落切分（同 pan25-baseline paragraph_chunking）
        2. Embedding 编码
        3. FAISS / 余弦矩阵召回候选对
        4. CrossEncoder 精排（可选）
        5. 合并去重、计算文档级分数
        """
        if similarity_threshold is None:
            similarity_threshold = settings.similarity_threshold
        if rerank_threshold is None:
            rerank_threshold = settings.rerank_threshold
        if top_k is None:
            top_k = settings.faiss_top_k

        # Step 1: 切分段落
        src_chunks = chunk_text(
            source_text,
            use_paragraph=settings.use_paragraph_chunking,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        susp_chunks = chunk_text(
            suspect_text,
            use_paragraph=settings.use_paragraph_chunking,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not src_chunks or not susp_chunks:
            return PlagiarismResult(similarity=0.0, matched_char_ratio=0.0, matches=[])

        # Step 2: 编码
        src_texts = [c.text for c in src_chunks]
        susp_texts = [c.text for c in susp_chunks]
        src_emb = self._embedder.encode(src_texts, normalize=True)
        susp_emb = self._embedder.encode(susp_texts, normalize=True)

        # Step 3: 余弦相似度矩阵召回
        #   sim_matrix[i][j] = cosine(susp_emb[i], src_emb[j])
        sim_matrix = FaissService.cosine_similarity_matrix(susp_emb, src_emb)

        # 收集候选对：每个 suspect 段落保留 top-k 最相似的 source 段落
        candidate_pairs: list[tuple[int, int, float]] = []  # (susp_idx, src_idx, cosine)
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix[i]
            # 取 top-k
            k = min(top_k, len(row))
            top_indices = np.argsort(row)[::-1][:k]
            for idx in top_indices:
                j = int(idx)
                cos_score = float(np.clip(row[j], 0.0, 1.0))
                if cos_score >= similarity_threshold:
                    candidate_pairs.append((i, j, cos_score))

        if not candidate_pairs:
            return PlagiarismResult(similarity=0.0, matched_char_ratio=0.0, matches=[])

        # Step 4: CrossEncoder 精排
        if use_rerank and candidate_pairs:
            rerank_pairs = [
                (susp_texts[si], src_texts[sj])
                for si, sj, _ in candidate_pairs
            ]
            rerank_results = self._reranker.rerank(
                rerank_pairs, threshold=rerank_threshold
            )
            # 根据 rerank 过滤
            filtered = []
            for orig_idx, rerank_score in rerank_results:
                si, sj, cos_score = candidate_pairs[orig_idx]
                filtered.append((si, sj, cos_score, rerank_score))
        else:
            filtered = [
                (si, sj, cos, cos) for si, sj, cos in candidate_pairs
            ]

        # Step 5: 去重 — 每个 suspect 段落只保留得分最高的一个匹配
        best_per_susp: dict[int, tuple[int, float, float]] = {}
        for si, sj, cos_score, final_score in filtered:
            if si not in best_per_susp or final_score > best_per_susp[si][1]:
                best_per_susp[si] = (sj, final_score, cos_score)

        # 构造 MatchResult
        matches: list[MatchResult] = []
        for si, (sj, final_score, cos_score) in sorted(best_per_susp.items()):
            s_chunk = susp_chunks[si]
            r_chunk = src_chunks[sj]
            matches.append(
                MatchResult(
                    source_chunk=r_chunk.text,
                    suspect_chunk=s_chunk.text,
                    score=final_score,
                    cosine_score=cos_score,
                    source_offset=r_chunk.offset,
                    source_length=r_chunk.length,
                    suspect_offset=s_chunk.offset,
                    suspect_length=s_chunk.length,
                )
            )

        # 文档级相似度
        similarity, matched_char_ratio = self._compute_document_score(
            matches, source_text, src_chunks
        )

        return PlagiarismResult(
            similarity=similarity,
            matched_char_ratio=matched_char_ratio,
            matches=matches,
        )

    # ------------------------------------------------------------------
    # 多疑似文档批量检查
    # ------------------------------------------------------------------

    def check_plagiarism(
        self, source_text: str, suspects: list[str]
    ) -> dict:
        """兼容旧路由接口：一个源文档 vs 多个疑似文档。"""
        all_matches: list[dict] = []
        max_similarity = 0.0

        for suspect in suspects:
            result = self.check(source_text, suspect)
            max_similarity = max(max_similarity, result.similarity)
            for m in result.matches:
                all_matches.append(
                    {
                        "source_chunk": m.source_chunk,
                        "suspect_chunk": m.suspect_chunk,
                        "score": round(m.score, 4),
                    }
                )

        return {
            "similarity": round(max_similarity, 4),
            "matches": all_matches,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_document_score(
        matches: list[MatchResult],
        source_text: str,
        src_chunks: list[Chunk],
    ) -> tuple[float, float]:
        """计算文档级相似度与匹配字符覆盖率。

        similarity = 匹配段落的平均最终得分（rerank 或 cosine）
        matched_char_ratio = 被匹配到的 source 字符数 / source 总长度
        """
        if not matches or not source_text:
            return 0.0, 0.0

        # 基于分数的加权平均
        total_score = sum(m.score for m in matches)
        avg_score = total_score / len(matches) if matches else 0.0

        # 字符覆盖率
        src_len = len(source_text)
        if src_len == 0:
            return avg_score, 0.0

        matched_chars = set()
        for m in matches:
            for i in range(m.source_offset, m.source_offset + m.source_length):
                matched_chars.add(i)
        char_ratio = len(matched_chars) / src_len

        return round(avg_score, 4), round(char_ratio, 4)
