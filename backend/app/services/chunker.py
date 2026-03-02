"""文本切块服务。

职责：
- 将文本按语义切块
- 支持两种模式：
    1. Pan25 风格段落切分（按空行分段, 去参考文献）
    2. 滑动窗口切分（fallback）
- 返回带字符偏移的切块，用于生成 PAN XML 输出

参考：pan25-baseline/cosine_baseline.py → paragraph_chunking()
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """一个文本切块，附带在原文中的字符位置信息。"""

    text: str
    offset: int  # 在原文中的起始字符偏移
    length: int  # 字符长度


# ---------- 元数据段落检测 ----------

# 匹配作者行：以 "By" 开头，后接若干人名（用 and / , 连接）
_AUTHOR_PATTERN = re.compile(
    r"^(?:by|authors?)\s*[:\-]?\s*"          # "By" / "Author:" / "Authors -"
    r"(?:[A-Z][a-z.\-]+[\s,]+(?:and\s+)?)+", # 后跟人名列表
    re.IGNORECASE,
)

# 匹配纯元数据行：日期行、机构行、邮箱、DOI、标题式短行等
_META_PATTERNS = [
    # 邮箱地址行
    re.compile(r"^[\w.\-]+@[\w.\-]+\.\w+", re.IGNORECASE),
    # DOI / arXiv
    re.compile(
        r"^(?:doi|arxiv|https?://(?:doi\.org|arxiv\.org))\s*[:\s]",
        re.IGNORECASE,
    ),
    # 纯日期行
    re.compile(
        r"^(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+\d{1,2},?\s+\d{4}$",
        re.IGNORECASE,
    ),
    # "Received: ... / Accepted: ..." 行
    re.compile(r"^(?:received|accepted|published|submitted)\s*:", re.IGNORECASE),
]


def _is_metadata_chunk(text: str) -> bool:
    """判断一段文本是否为元数据（作者行、邮箱、DOI 等），而非正文内容。"""
    stripped = text.strip()

    # 作者列表检测（最常见的误匹配来源）
    if _AUTHOR_PATTERN.match(stripped):
        # 进一步验证：如果 "and" / "," 连接的人名 ≥2 个且整段无正文句子，基本确认为作者行
        separators = len(re.findall(r"\band\b|,", stripped, re.IGNORECASE))
        has_period_sentence = bool(re.search(r"[.!?]\s+[A-Z]", stripped))
        if separators >= 2 and not has_period_sentence:
            return True

    # 其他元数据模式
    for pat in _META_PATTERNS:
        if pat.match(stripped):
            return True

    # 内容密度检测：如果字母字符占比太低（如纯数字/符号行），视为非内容
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 0 and alpha_count / len(stripped) < 0.4:
        return True

    return False


# ---------- Pan25 风格：按段落切分 ----------

def paragraph_chunking(text: str, min_chars: int = 15) -> list[Chunk]:
    """按段落切分文本，并丢弃 references/bibliography 等尾部段落。

    复用 pan25-baseline 的逻辑，在此基础上额外记录每段在原文中的
    字符偏移与长度，用于后续生成 PAN XML 检测结果。

    参数：
    - min_chars: 段落最小字符数阈值，低于此长度的段落将被丢弃（默认 15）
    """
    cleaned = text.strip()

    # 移除参考文献段
    references_pattern = (
        r"(?si)(?:\n\n+|^)"
        r"(?:references|bibliography|reference list|works cited)"
        r"(?:\n\n+.*)?$"
    )
    cleaned = re.sub(references_pattern, "", cleaned)

    # 以空行切段（保持公式块与上下文在同一段）
    parts = re.split(r"\n\n(?!\s\n\s)", cleaned)

    chunks: list[Chunk] = []
    search_start = 0
    for part in parts:
        stripped = part.strip()
        if not stripped or len(stripped) < min_chars:
            continue
        # 跳过元数据段落（作者行、邮箱、DOI 等）
        if _is_metadata_chunk(stripped):
            continue
        # 在原文中定位该段落的偏移
        idx = text.find(stripped, search_start)
        if idx == -1:
            # fallback：从头搜索
            idx = text.find(stripped)
        offset = idx if idx != -1 else search_start
        chunks.append(Chunk(text=stripped, offset=offset, length=len(stripped)))
        if idx != -1:
            search_start = idx + len(stripped)
    return chunks


# ---------- 滑动窗口切分 ----------

def sliding_window_chunking(
    text: str, chunk_size: int = 512, overlap: int = 128, min_chars: int = 15
) -> list[Chunk]:
    """使用滑动窗口切分文本，每个窗口覆盖 chunk_size 个字符，
    相邻窗口之间有 overlap 个字符的重叠，以保留上下文。

    参数：
    - min_chars: 切块最小字符数阈值，低于此长度的切块将被丢弃（默认 15）
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(chunk_size - overlap, 1)
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end].strip()
        if chunk_text and len(chunk_text) >= min_chars and not _is_metadata_chunk(chunk_text):
            chunks.append(Chunk(text=chunk_text, offset=pos, length=end - pos))
        pos += step
    return chunks


# ---------- 统一入口 ----------

def chunk_text(
    text: str,
    *,
    use_paragraph: bool = True,
    chunk_size: int = 512,
    overlap: int = 128,
    min_chars: int = 15,
) -> list[Chunk]:
    """切分文本的统一入口。

    参数：
    - use_paragraph: True → Pan25 段落模式；False → 滑动窗口模式
    - chunk_size / overlap: 仅在滑动窗口模式下生效
    - min_chars: 切块最小字符数阈值，低于此长度的切块将被丢弃（默认 15）
    """
    if use_paragraph:
        return paragraph_chunking(text, min_chars=min_chars)
    return sliding_window_chunking(text, chunk_size, overlap, min_chars=min_chars)
