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

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import settings
from .chunker import Chunk, chunk_text
from .embedding_service import EmbeddingService
from .faiss_service import FaissService
from .rerank_service import RerankService

logger = logging.getLogger(__name__)


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
        t0 = time.time()
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
        logger.info("Step1 chunking: %.2fs  src=%d  susp=%d",
                     time.time() - t0, len(src_chunks), len(susp_chunks))

        if not src_chunks or not susp_chunks:
            return PlagiarismResult(similarity=0.0, matched_char_ratio=0.0, matches=[])

        # Step 2: 编码
        t1 = time.time()
        src_texts = [c.text for c in src_chunks]
        susp_texts = [c.text for c in susp_chunks]
        src_emb = self._embedder.encode(src_texts, normalize=True)
        susp_emb = self._embedder.encode(susp_texts, normalize=True)
        logger.info("Step2 encoding: %.2fs", time.time() - t1)

        # Step 3: 余弦相似度矩阵召回
        t2 = time.time()
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
        logger.info("Step3 recall: %.2fs  candidates=%d", time.time() - t2, len(candidate_pairs))

        if not candidate_pairs:
            return PlagiarismResult(similarity=0.0, matched_char_ratio=0.0, matches=[])

        # Step 4: CrossEncoder 精排
        t3 = time.time()
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
        logger.info("Step4 rerank: %.2fs  filtered=%d", time.time() - t3, len(filtered))

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

        # Step 6: 后处理 — 源段落去重 + 相邻区间合并
        t5 = time.time()
        matches = self._dedup_by_source(matches)
        matches = self._merge_adjacent_matches(matches, gap_threshold=100)
        logger.info("Step6 postprocess: %.2fs  final_matches=%d",
                     time.time() - t5, len(matches))

        # 文档级相似度（综合双向覆盖率 + 平均匹配分）
        similarity, matched_char_ratio = self._compute_document_score(
            matches, source_text, suspect_text
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
    def _dedup_by_source(matches: list[MatchResult]) -> list[MatchResult]:
        """同一源段落匹配多个疑似段落时，只保留 rerank 分最高的一对。"""
        best: dict[tuple[int, int], MatchResult] = {}
        for m in matches:
            key = (m.source_offset, m.source_length)
            if key not in best or m.score > best[key].score:
                best[key] = m
        return sorted(best.values(), key=lambda x: x.suspect_offset)

    @staticmethod
    def _merge_adjacent_matches(
        matches: list[MatchResult],
        gap_threshold: int = 100,
    ) -> list[MatchResult]:
        """合并疑似文档中相邻/重叠的匹配区间。

        当两个匹配对在疑似文档中的间距 ≤ gap_threshold 字符时，
        合并为一个更大的匹配区间。源文档侧同样合并。
        合并后的 score 和 cosine_score 取按字符长度加权平均。
        """
        if not matches:
            return []

        sorted_m = sorted(matches, key=lambda x: x.suspect_offset)
        merged: list[MatchResult] = [sorted_m[0]]

        for curr in sorted_m[1:]:
            prev = merged[-1]
            prev_susp_end = prev.suspect_offset + prev.suspect_length
            curr_susp_end = curr.suspect_offset + curr.suspect_length

            if curr.suspect_offset <= prev_susp_end + gap_threshold:
                # 合并疑似文档区间
                new_susp_offset = prev.suspect_offset
                new_susp_end = max(prev_susp_end, curr_susp_end)
                new_susp_length = new_susp_end - new_susp_offset

                # 合并源文档区间
                prev_src_end = prev.source_offset + prev.source_length
                curr_src_end = curr.source_offset + curr.source_length
                new_src_offset = min(prev.source_offset, curr.source_offset)
                new_src_end = max(prev_src_end, curr_src_end)
                new_src_length = new_src_end - new_src_offset

                # 加权平均分数（按字符长度加权）
                w_prev = prev.suspect_length
                w_curr = curr.suspect_length
                total_w = w_prev + w_curr
                avg_score = (prev.score * w_prev + curr.score * w_curr) / total_w
                avg_cosine = (
                    prev.cosine_score * w_prev + curr.cosine_score * w_curr
                ) / total_w

                # 文本：取较长的源段落，疑似段落拼接
                merged[-1] = MatchResult(
                    source_chunk=(
                        prev.source_chunk
                        if prev.source_length >= curr.source_length
                        else curr.source_chunk
                    ),
                    suspect_chunk=(
                        prev.suspect_chunk + "\n\n" + curr.suspect_chunk
                    ),
                    score=round(avg_score, 4),
                    cosine_score=round(avg_cosine, 4),
                    source_offset=new_src_offset,
                    source_length=new_src_length,
                    suspect_offset=new_susp_offset,
                    suspect_length=new_susp_length,
                )
            else:
                merged.append(curr)

        return merged

    @staticmethod
    def _compute_document_score(
        matches: list[MatchResult],
        source_text: str,
        suspect_text: str,
    ) -> tuple[float, float]:
        """计算文档级相似度与匹配字符覆盖率。

        similarity = 综合源/疑似双向字符覆盖率 + 平均匹配分
        matched_char_ratio = 被匹配到的 source 字符数 / source 总长度
        """
        if not matches or not source_text:
            return 0.0, 0.0

        src_len = len(source_text.strip())
        susp_len = len(suspect_text.strip())
        if src_len == 0 or susp_len == 0:
            return 0.0, 0.0

        # 源文档字符覆盖
        matched_src_chars: set[int] = set()
        matched_susp_chars: set[int] = set()
        for m in matches:
            for i in range(m.source_offset, m.source_offset + m.source_length):
                matched_src_chars.add(i)
            for i in range(m.suspect_offset, m.suspect_offset + m.suspect_length):
                matched_susp_chars.add(i)

        src_coverage = len(matched_src_chars) / src_len
        susp_coverage = len(matched_susp_chars) / susp_len
        char_ratio = round(src_coverage, 4)

        # 综合评分：平均分 × 0.4 + 源覆盖 × 0.3 + 疑似覆盖 × 0.3
        avg_score = sum(m.score for m in matches) / len(matches)
        similarity = round(
            0.4 * avg_score + 0.3 * src_coverage + 0.3 * susp_coverage, 4
        )

        return similarity, char_ratio
