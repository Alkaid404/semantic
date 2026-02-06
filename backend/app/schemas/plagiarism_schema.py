"""查重结果的 Pydantic 数据模型。"""
from __future__ import annotations

from pydantic import BaseModel


class Match(BaseModel):
    source_chunk: str
    suspect_chunk: str
    score: float


class PlagiarismResponse(BaseModel):
    similarity: float
    matches: list[Match]


class PlagiarismRequest(BaseModel):
    source_text: str
    suspects: list[str]
