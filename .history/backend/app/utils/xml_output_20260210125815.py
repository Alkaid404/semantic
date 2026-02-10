"""PAN XML 格式输出。

生成与 pan25-baseline/cosine_baseline.py → generate_xml() 兼容的
检测结果 XML 文件，用于 PAN 评测系统。

PAN XML 格式示例：
  <document reference="suspicious-document020468.txt">
    <feature name="about" authors="" title="" lang="en" .../>
    <feature name="plagiarism"
             type="paraphrase"
             this_language="en"
             this_offset="117" this_length="1465"
             source_reference="source-document020468.txt"
             source_language="en"
             source_offset="82" source_length="1677"/>
  </document>
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from ..services.similarity_engine import MatchResult


def generate_pan_xml(
    matches: list[MatchResult],
    susp_doc_name: str,
    src_doc_name: str,
) -> str:
    """将检测到的匹配段落对写成 PAN XML 格式字符串。

    参考：pan25-baseline/cosine_baseline.py → generate_xml()

    参数：
    - matches:       MatchResult 列表
    - susp_doc_name: 可疑文档文件名 (如 "suspicious-document020468.txt")
    - src_doc_name:  源文档文件名   (如 "source-document020468.txt")

    返回：
    - XML 字符串
    """
    root = ET.Element("document", reference=susp_doc_name)

    # about 元素
    ET.SubElement(
        root,
        "feature",
        name="about",
        authors="",
        title="",
        lang="en",
        similarity="",
        severity="",
    )

    # 每个 match 对应一个 plagiarism feature
    for m in matches:
        ET.SubElement(
            root,
            "feature",
            name="plagiarism",
            type="paraphrase",
            this_language="en",
            this_offset=str(m.suspect_offset),
            this_length=str(m.suspect_length),
            source_reference=src_doc_name,
            source_language="en",
            source_offset=str(m.source_offset),
            source_length=str(m.source_length),
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)

    # 写入字符串
    from io import BytesIO

    buf = BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def save_pan_xml(
    matches: list[MatchResult],
    susp_doc_name: str,
    src_doc_name: str,
    output_path: str,
) -> str:
    """将检测结果写入 PAN XML 文件。

    返回输出文件路径。
    """
    import os

    xml_str = generate_pan_xml(matches, susp_doc_name, src_doc_name)

    # 输出文件名：suspicious-xxx-plagiarized-source-xxx.xml
    base_susp = os.path.splitext(susp_doc_name)[0]
    base_src = os.path.splitext(src_doc_name)[0]
    output_file = os.path.join(
        output_path, f"{base_susp}-plagiarized-{base_src}.xml"
    )

    os.makedirs(output_path, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return output_file
