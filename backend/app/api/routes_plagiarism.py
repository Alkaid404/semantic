"""查重路由。

职责：
- 接收文件或文本上传
- 调用查重引擎
- 返回相似度结果
"""
from fastapi import APIRouter

from ..schemas.plagiarism_schema import PlagiarismRequest, PlagiarismResponse
from ..services.similarity_engine import SimilarityEngine

router = APIRouter()
_engine = SimilarityEngine()


@router.post("/check", response_model=PlagiarismResponse)
def check_plagiarism(payload: PlagiarismRequest):
    """执行完整流程进行查重。"""
    return _engine.check_plagiarism(payload.source_text, payload.suspects)
