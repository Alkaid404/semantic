"""文本切块服务。

职责：
- 将文本按语义切块

建议：
- 使用滑动窗口避免语义断裂。
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """使用滑动窗口切分文本。

    接口：
    - text: 原始文本
    - chunk_size: 每个块的最大长度
    - overlap: 片段重叠长度，用于保留上下文
    """
    # TODO: 实现滑动窗口切块。
    return []
