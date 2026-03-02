"""查重路由。

职责：
- 接收文本或文件上传
- 调用查重引擎
- 返回 JSON 结果 / PAN XML 文件
- 自动将每次查重结果保存到 log 目录

路由：
  POST /check       — JSON 文本查重
  POST /check/files — 文件上传查重
  POST /check/xml   — 查重并返回 PAN XML
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, File, Response, UploadFile
from fastapi.responses import JSONResponse

from ..schemas.plagiarism_schema import (
    Match,
    PlagiarismRequest,
    PlagiarismResponse,
)
from ..services.similarity_engine import SimilarityEngine
from ..utils.file_parser import parse_file
from ..utils.xml_output import generate_pan_xml

router = APIRouter()
_engine = SimilarityEngine()

logger = logging.getLogger(__name__)

# ---- 日志保存目录 ----
_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "log")
os.makedirs(_LOG_DIR, exist_ok=True)


def _save_result_log(response: PlagiarismResponse, tag: str = "check") -> None:
    """将查重结果以 JSON 格式保存到 log 目录。"""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{tag}_{ts}.json"
        filepath = os.path.join(_LOG_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info("Result saved to %s", filepath)
    except Exception as e:
        logger.warning("Failed to save result log: %s", e)


@router.post("/check", response_model=PlagiarismResponse)
def check_plagiarism(payload: PlagiarismRequest):
    """通过 JSON body 进行查重，返回结果。"""

    # 兼容两种输入方式
    if payload.suspect_text:
        result = _engine.check(payload.source_text, payload.suspect_text)
        resp = PlagiarismResponse(
            similarity=result.similarity,
            matched_char_ratio=result.matched_char_ratio,
            matches=[
                Match(
                    source_chunk=m.source_chunk,
                    suspect_chunk=m.suspect_chunk,
                    score=round(m.score, 4),
                    cosine_score=round(m.cosine_score, 4),
                    source_offset=m.source_offset,
                    source_length=m.source_length,
                    suspect_offset=m.suspect_offset,
                    suspect_length=m.suspect_length,
                )
                for m in result.matches
            ],
        )
        _save_result_log(resp, "check")
        return resp

    # 多个疑似文档
    if payload.suspects:
        all_matches: list[Match] = []
        max_sim = 0.0
        max_char_ratio = 0.0
        for suspect in payload.suspects:
            result = _engine.check(payload.source_text, suspect)
            max_sim = max(max_sim, result.similarity)
            max_char_ratio = max(max_char_ratio, result.matched_char_ratio)
            for m in result.matches:
                all_matches.append(
                    Match(
                        source_chunk=m.source_chunk,
                        suspect_chunk=m.suspect_chunk,
                        score=round(m.score, 4),
                        cosine_score=round(m.cosine_score, 4),
                        source_offset=m.source_offset,
                        source_length=m.source_length,
                        suspect_offset=m.suspect_offset,
                        suspect_length=m.suspect_length,
                    )
                )
        resp = PlagiarismResponse(
            similarity=max_sim,
            matched_char_ratio=max_char_ratio,
            matches=all_matches,
        )
        _save_result_log(resp, "check_multi")
        return resp

    return PlagiarismResponse(similarity=0.0, matched_char_ratio=0.0, matches=[])


@router.post("/check/files", response_model=PlagiarismResponse)
async def check_plagiarism_files(
    source_file: UploadFile = File(..., description="源文档（txt）"),
    suspect_file: UploadFile = File(..., description="可疑文档（txt）"),
):
    """上传两个文件进行查重。"""
    source_text = await parse_file(source_file)
    suspect_text = await parse_file(suspect_file)

    result = _engine.check(source_text, suspect_text)
    resp = PlagiarismResponse(
        similarity=result.similarity,
        matched_char_ratio=result.matched_char_ratio,
        matches=[
            Match(
                source_chunk=m.source_chunk,
                suspect_chunk=m.suspect_chunk,
                score=round(m.score, 4),
                cosine_score=round(m.cosine_score, 4),
                source_offset=m.source_offset,
                source_length=m.source_length,
                suspect_offset=m.suspect_offset,
                suspect_length=m.suspect_length,
            )
            for m in result.matches
        ],
    )
    _save_result_log(resp, "check_files")
    return resp


@router.post("/check/xml")
async def check_plagiarism_xml(
    source_file: UploadFile = File(..., description="源文档（txt）"),
    suspect_file: UploadFile = File(..., description="可疑文档（txt）"),
):
    """上传两个文件进行查重，并返回 PAN XML 格式结果。"""
    source_text = await parse_file(source_file)
    suspect_text = await parse_file(suspect_file)

    susp_name = suspect_file.filename or "suspect.txt"
    src_name = source_file.filename or "source.txt"

    result = _engine.check(source_text, suspect_text)

    xml_str = generate_pan_xml(
        matches=result.matches,
        susp_doc_name=susp_name,
        src_doc_name=src_name,
    )
    return Response(content=xml_str, media_type="application/xml")
