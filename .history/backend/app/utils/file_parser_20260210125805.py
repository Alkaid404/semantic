"""文件解析工具。

支持：
- txt — 纯文本
- pdf（可扩展）
- docx（可扩展）
"""
from __future__ import annotations

from fastapi import UploadFile


async def parse_file(file: UploadFile) -> str:
    """将上传文件解析为文本。

    当前仅支持 txt 纯文本；后续可扩展 pdf/docx。
    """
    raw = await file.read()

    # 尝试 UTF-8，失败后回退 latin-1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    return text


def read_text_file(path: str) -> str:
    """从本地路径读取纯文本文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
