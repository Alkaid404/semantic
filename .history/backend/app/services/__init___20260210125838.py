"""核心算法服务。"""

from .chunker import Chunk, chunk_text, paragraph_chunking, sliding_window_chunking
from .embedding_service import EmbeddingService
from .faiss_service import FaissService
from .rerank_service import RerankService
from .similarity_engine import MatchResult, PlagiarismResult, SimilarityEngine

__all__ = [
    "Chunk",
    "chunk_text",
    "paragraph_chunking",
    "sliding_window_chunking",
    "EmbeddingService",
    "FaissService",
    "RerankService",
    "SimilarityEngine",
    "MatchResult",
    "PlagiarismResult",
]
