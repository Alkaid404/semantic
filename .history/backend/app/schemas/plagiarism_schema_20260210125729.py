"""查重结果的 Pydantic 数据模型。

扩展字段参照 PAN25 检测结果格式，增加了字符偏移与长度信息，
以便生成 PAN XML 输出与可视化高亮。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Match(BaseModel):
    """单条匹配记录。"""

    source_chunk: str = Field(description="源文档中被匹配到的段落文本")
    suspect_chunk: str = Field(description="可疑文档中被匹配到的段落文本")
    score: float = Field(description="最终相似度得分 (rerank 后)")
    cosine_score: float = Field(0.0, description="Embedding 余弦相似度")
    source_offset: int = Field(0, description="源文档字符偏移")
    source_length: int = Field(0, description="源文档匹配长度")
    suspect_offset: int = Field(0, description="可疑文档字符偏移")
    suspect_length: int = Field(0, description="可疑文档匹配长度")


class PlagiarismResponse(BaseModel):
    """查重响应。"""

    similarity: float = Field(description="文档级相似度 (0~1)")
    matched_char_ratio: float = Field(
        0.0, description="源文档被匹配到的字符比例"
    )
    matches: list[Match] = Field(default_factory=list)


class PlagiarismRequest(BaseModel):
    """查重请求：传入源文本 + 一个或多个可疑文本。"""

    source_text: str = Field(description="源文档文本")
    suspect_text: str = Field(
        "", description="单个可疑文档文本（二选一）"
    )
    suspects: list[str] = Field(
        default_factory=list,
        description="多个可疑文档文本列表（二选一）",
    )
