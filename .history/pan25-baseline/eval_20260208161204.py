#! /usr/bin/python
# 版权所有 (C) 2009 webis.de。保留所有权利。
"""抄袭检测评测指标。
源实现参考：https://github.com/pan-webis-de/pan-code/blob/master/sepln09/pan09-plagiarism-detection-performance-measures.py

本模块实现了召回率、精确率与粒度（granularity）等指标，依据文献 [1] 的定义。
指标既支持宏平均也支持微平均，两者都以“真实抄袭标注”与“检测结果标注”为输入。
检测结果会与真实标注比较，从而评估不同检测算法的准确性。
此外，`plagdet_score` 会将召回率、精确率与粒度综合成一个评分，用于排序算法。

参数 `cases` 与 `detections` 都应为 `Annotation` 实例的可迭代集合。
`Annotation` 是一个 7 元组，示例：
>>> Annotation('suspicious-document00001.txt', 10000, 1000, \
...            'source-document00001.txt', 5000, 1100, True)
前 3 个字段表示可疑文档中的字符偏移与长度，后 3 个字段表示源文档中的偏移与长度。
最后一个字段表示该标注是否为外部检测；若为内在检测，源文档相关字段应为空/0。

本模块还包含从 XML 文件中提取抄袭标注的函数，兼容 PAN 评测格式 [2,3]。

[1]  Martin Potthast, Benno Stein, Alberto Barron-Cedeno, and Paolo Rosso.
    An Evaluation Framework for Plagiarism Detection.
    In Proceedings of COLING 2010, Beijing, China. ACL.

[2]  Martin Potthast, Benno Stein, Andreas Eiselt, Alberto Barron-Cedeno,
    and Paolo Rosso. Overview of PAN 09 Plagiarism Detection.
    SEPLN 2009 Workshop. CEUR-WS.org.

[3]  Martin Potthast, Benno Stein, Andreas Eiselt, Alberto Barron-Cedeno,
    and Paolo Rosso. Overview of PAN 10 Plagiarism Detection.
    PAN at CLEF 2010.
"""

from __future__ import division

__author__ = "Martin Potthast"
__email__ = "martin.potthast at uni-weimar dot de"
__version__ = "1.3"
__all__ = [
    "macro_avg_recall_and_precision",
    "micro_avg_recall_and_precision",
    "granularity",
    "plagdet_score",
    "Annotation",
]


from collections import namedtuple
import getopt
import glob
import math
from numpy import int8 as npint8
from numpy.ma import zeros, sum as npsum
import os
import sys
import unittest
import xml.dom.minidom


TREF, TOFF, TLEN = "this_reference", "this_offset", "this_length"
SREF, SOFF, SLEN = "source_reference", "source_offset", "source_length"
EXT = "is_external"
Annotation = namedtuple("Annotation", [TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT])
TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT = range(7)


def macro_avg_recall_and_precision(cases, detections):
    """返回 (rec, prec)：基于宏平均的召回率与精确率。"""
    return macro_avg_recall(cases, detections), macro_avg_precision(cases, detections)


def micro_avg_recall_and_precision(cases, detections):
    """返回 (rec, prec)：基于微平均的召回率与精确率。"""
    if len(cases) == 0 and len(detections) == 0:
        return 1, 1
    if len(cases) == 0 or len(detections) == 0:
        return 0, 0
    num_plagiarized, num_detected, num_plagiarized_detected = 0, 0, 0  # 字符数
    num_plagiarized += count_chars(cases)
    num_detected += count_chars(detections)
    detections = true_detections(cases, detections)
    num_plagiarized_detected += count_chars(detections)
    rec, prec = 0, 0
    if num_plagiarized > 0:
        rec = num_plagiarized_detected / num_plagiarized
    if num_detected > 0:
        prec = num_plagiarized_detected / num_detected
    return rec, prec


def granularity(cases, detections):
    """计算检测结果的粒度（一个真实案例被拆成多个检测的程度）。"""
    if len(detections) == 0:
        return 1
    detections_per_case = list()
    case_index = index_annotations(cases)
    det_index = index_annotations(detections)
    for tref in case_index:
        cases, detections = case_index[tref], det_index.get(tref, False)
        if not detections:  # 该文档没有检测结果
            continue
        for case in cases:
            num_dets = sum((is_overlapping(case, det) for det in detections))
            detections_per_case.append(num_dets)
    detected_cases = sum((num_dets > 0 for num_dets in detections_per_case))
    if detected_cases == 0:
        return 1
    return sum(detections_per_case) / detected_cases


def plagdet_score(rec, prec, gran):
    """综合召回率、精确率与粒度，得到可排序的 plagdet 分数。"""
    if (rec == 0 and prec == 0) or prec < 0 or rec < 0 or gran < 1:
        return 0
    return ((2 * rec * prec) / (rec + prec)) / math.log(1 + gran, 2)


def macro_avg_recall(cases, detections):
    """计算宏平均召回率。"""
    if len(cases) == 0 and len(detections) == 0:
        return 1
    elif len(cases) == 0 or len(detections) == 0:
        return 0
    num_cases, recall_per_case = len(cases), list()
    case_index = index_annotations(cases)
    det_index = index_annotations(detections)
    for tref in case_index:
        cases, detections = case_index[tref], det_index.get(tref, False)
        if not detections:  # 该文档没有检测结果
            continue
        for case in cases:
            recall_per_case.append(case_recall(case, detections))
    return sum(recall_per_case) / num_cases


def case_recall(case, detections):
    """计算单个真实案例的召回率。"""
    num_detected_plagiarized = overlapping_chars(case, detections)
    num_plagiarized = case[TLEN] + case[SLEN]
    return num_detected_plagiarized / num_plagiarized


def macro_avg_precision(cases, detections):
    """计算宏平均精确率。"""
    # 注意：宏平均精确率等价于交换参数计算宏平均召回率
    return macro_avg_recall(detections, cases)


def true_detections(cases, detections):
    """仅保留与真实案例重叠的检测，并裁剪为实际重叠的片段。"""
    true_dets = list()
    case_index = index_annotations(cases)
    det_index = index_annotations(detections)
    for tref in case_index:
        cases, detections = case_index[tref], det_index.get(tref, False)
        if not detections:  # 该文档没有检测结果
            continue
        for case in cases:
            case_dets = (det for det in detections if is_overlapping(case, det))
            true_case_dets = (overlap_annotation(case, det) for det in case_dets)
            true_dets.extend(true_case_dets)
    return true_dets


def overlap_annotation(ann1, ann2):
    """返回 ann1 与 ann2 的重叠部分标注。"""
    tref, sref, ext = ann1[TREF], ann1[SREF], ann1[EXT] and ann2[EXT]
    toff, tlen, soff, slen = 0, 0, 0, 0
    if is_overlapping(ann1, ann2):
        toff, tlen = overlap_chars(ann1, ann2, TOFF, TLEN)
        if ext:
            soff, slen = overlap_chars(ann1, ann2, SOFF, SLEN)
    return Annotation(tref, toff, tlen, sref, soff, slen, ext)


def overlap_chars(ann1, ann2, xoff, xlen):
    """根据偏移与长度字段，计算两段标注的重叠区间。"""
    overlap_start, overlap_length = 0, 0
    max_ann = ann1 if ann1[xoff] >= ann2[xoff] else ann2
    min_ann = ann1 if ann1[xoff] < ann2[xoff] else ann2
    if min_ann[xoff] + min_ann[xlen] > max_ann[xoff]:
        overlap_start = max_ann[xoff]
        overlap_end = min(min_ann[xoff] + min_ann[xlen], max_ann[xoff] + max_ann[xlen])
        overlap_length = overlap_end - overlap_start
    return overlap_start, overlap_length


def count_chars(annotations):
    """统计标注覆盖的字符数（重叠字符只计一次）。"""
    num_chars = count_chars2(annotations, TREF, TOFF, TLEN)
    num_chars += count_chars2(annotations, SREF, SOFF, SLEN)
    return num_chars


def count_chars2(annotations, xref, xoff, xlen):
    """在指定引用字段上统计覆盖字符数。"""
    num_chars = 0
    max_length = max((ann[xoff] + ann[xlen] for ann in annotations))
    char_bits = zeros(max_length, dtype=bool)
    xref_index = index_annotations(annotations, xref)
    for xref in xref_index:
        annotations = xref_index[xref]
        char_bits[:] = False
        for ann in annotations:
            char_bits[ann[xoff] : ann[xoff] + ann[xlen]] = True
        num_chars += npsum(char_bits)
    return num_chars


def overlapping_chars(ann1, annotations):
    """统计 ann1 与一组标注之间的重叠字符数。"""
    annotations = [ann2 for ann2 in annotations if is_overlapping(ann1, ann2)]
    if len(annotations) == 0 or not isinstance(ann1, Annotation):
        return 0
    this_overlaps = zeros(ann1[TLEN], dtype=bool)
    source_overlaps = zeros(ann1[SLEN], dtype=bool)
    for ann2 in annotations:
        mark_overlapping_chars(this_overlaps, ann1, ann2, TOFF, TLEN)
        mark_overlapping_chars(source_overlaps, ann1, ann2, SOFF, SLEN)
    return npsum(this_overlaps) + npsum(source_overlaps)


def mark_overlapping_chars(char_bits, ann1, ann2, xoff, xlen):
    """将 ann2 与 ann1 的重叠部分在布尔数组中标记为 True。"""
    offset_difference = ann2[xoff] - ann1[xoff]
    overlap_start = min(max(0, offset_difference), ann1[xlen])
    overlap_end = min(max(0, offset_difference + ann2[xlen]), ann1[xlen])
    char_bits[overlap_start:overlap_end] = True


def is_overlapping(ann1, ann2):
    """判断 ann2 是否与 ann1 存在重叠。"""
    detected = (
        ann1[TREF] == ann2[TREF]
        and ann2[TOFF] + ann2[TLEN] > ann1[TOFF]
        and ann2[TOFF] < ann1[TOFF] + ann1[TLEN]
    )
    if ann1[EXT] == True and ann2[EXT] == True:
        detected = (
            detected
            and ann1[SREF] == ann2[SREF]
            and ann2[SOFF] + ann2[SLEN] > ann1[SOFF]
            and ann2[SOFF] < ann1[SOFF] + ann1[SLEN]
        )
    return detected


def index_annotations(annotations, xref=TREF):
    """构建倒排索引：引用值 -> 标注列表。"""
    index = dict()
    for ann in annotations:
        index.setdefault(ann[xref], []).append(ann)
    return index


def extract_annotations_from_files(path, tagname):
    """从目录中读取 XML，并提取指定标签的标注集合。"""
    if not os.path.exists(path):
        print("Path not accessible:", path)
        sys.exit(2)
    annotations = set()
    xmlfiles = glob.glob(os.path.join(path, "*.xml"))
    xmlfiles.extend(glob.glob(os.path.join(path, os.path.join("*", "*.xml"))))
    for xmlfile in xmlfiles:
        annotations.update(extract_annotations_from_file(xmlfile, tagname))
    return annotations


def extract_annotations_from_file(xmlfile, tagname):
    """从单个 XML 文件中提取指定标签的标注集合。"""
    doc = xml.dom.minidom.parse(xmlfile)
    annotations = set()
    if not doc.documentElement.hasAttribute("reference"):
        return annotations
    t_ref = doc.documentElement.getAttribute("reference")
    for node in doc.documentElement.childNodes:
        if (
            node.nodeType == xml.dom.Node.ELEMENT_NODE
            and node.hasAttribute("name")
            and node.getAttribute("name").endswith(tagname)
        ):
            ann = extract_annotation_from_node(node, t_ref)
            if ann:
                annotations.add(ann)
    return annotations


def extract_annotation_from_node(xmlnode, t_ref):
    """从 XML 节点构造一个标注对象。"""
    if not (
        xmlnode.hasAttribute("this_offset") and xmlnode.hasAttribute("this_length")
    ):
        return False
    t_off = int(xmlnode.getAttribute("this_offset"))
    t_len = int(xmlnode.getAttribute("this_length"))
    s_ref, s_off, s_len, ext = "", 0, 0, False
    if (
        xmlnode.hasAttribute("source_reference")
        and xmlnode.hasAttribute("source_offset")
        and xmlnode.hasAttribute("source_length")
    ):
        s_ref = xmlnode.getAttribute("source_reference")
        s_off = int(xmlnode.getAttribute("source_offset"))
        s_len = int(xmlnode.getAttribute("source_length"))
        ext = True
    return Annotation(
        t_ref.replace(".txt", ""),
        t_off,
        t_len,
        s_ref.replace(".txt", ""),
        s_off,
        s_len,
        ext,
    )


class TestPerfMeasures(unittest.TestCase):
    """评测指标的单元测试。"""

    ann1 = Annotation("tref1", 0, 100, "sref1", 0, 100, True)
    ann2 = Annotation("tref1", 0, 100, "", 0, 0, False)
    ann3 = Annotation("tref1", 100, 100, "sref1", 100, 100, True)
    ann4 = Annotation("tref1", 0, 200, "sref1", 0, 200, True)
    ann5 = Annotation("tref1", 0, 1, "sref1", 0, 1, True)
    ann6 = Annotation("tref1", 99, 1, "sref1", 99, 1, True)
    ann7 = Annotation("tref2", 0, 100, "sref2", 0, 100, True)
    ann8 = Annotation("tref2", 0, 100, "", 0, 0, False)
    ann9 = Annotation("tref2", 50, 100, "sref2", 50, 100, True)
    ann10 = Annotation("tref2", 25, 75, "sref2", 25, 75, True)

    def test_macro_averaged_recall(self):
        self.assertEqual(1, macro_avg_recall([], []))
        self.assertEqual(0, macro_avg_recall(["sth"], []))
        self.assertEqual(0, macro_avg_recall([], ["sth"]))
        self.assertEqual(1, macro_avg_recall([self.ann1], [self.ann1]))
        self.assertEqual(1, macro_avg_recall([self.ann2], [self.ann2]))
        self.assertEqual(0.5, macro_avg_recall([self.ann1, self.ann7], [self.ann1]))
        self.assertEqual(0.5, macro_avg_recall([self.ann2, self.ann8], [self.ann2]))
        self.assertEqual(0, macro_avg_recall([self.ann1], [self.ann7]))
        self.assertEqual(0, macro_avg_recall([self.ann2], [self.ann8]))

    def test_case_recall(self):
        self.assertEqual(0, case_recall(self.ann1, []))
        self.assertEqual(1, case_recall(self.ann1, [self.ann1]))
        self.assertEqual(0.5, case_recall(self.ann1, [self.ann2]))
        self.assertEqual(0, case_recall(self.ann1, [self.ann3]))
        self.assertEqual(1, case_recall(self.ann1, [self.ann4]))
        self.assertEqual(1, case_recall(self.ann1, [self.ann4, self.ann7]))
        self.assertEqual(0, case_recall(self.ann1, [self.ann7, self.ann9]))
        self.assertEqual(0.5, case_recall(self.ann7, [self.ann9]))
        self.assertEqual(0.75, case_recall(self.ann7, [self.ann10]))
        self.assertEqual(0.75, case_recall(self.ann7, [self.ann9, self.ann10]))

    def test_macro_averaged_precision(self):
        self.assertEqual(1, macro_avg_precision([], []))
        self.assertEqual(0, macro_avg_precision(["sth"], []))
        self.assertEqual(0, macro_avg_precision([], ["sth"]))
        self.assertEqual(1, macro_avg_precision([self.ann1], [self.ann1]))
        self.assertEqual(1, macro_avg_precision([self.ann2], [self.ann2]))
        self.assertEqual(1, macro_avg_precision([self.ann1, self.ann7], [self.ann1]))
        self.assertEqual(1, macro_avg_precision([self.ann2, self.ann8], [self.ann2]))
        self.assertEqual(0.5, macro_avg_precision([self.ann1], [self.ann4]))
        self.assertEqual(1, macro_avg_precision([self.ann7], [self.ann10]))
        self.assertEqual(1, macro_avg_precision([self.ann7], [self.ann10]))
        self.assertEqual(
            0.75, macro_avg_precision([self.ann7], [self.ann9, self.ann10])
        )
        self.assertEqual(0.25, macro_avg_precision([self.ann1], [self.ann3, self.ann4]))

    def test_granularity(self):
        self.assertEqual(1, granularity([], []))
        self.assertEqual(1, granularity([self.ann1], [self.ann2]))
        self.assertEqual(1, granularity([self.ann1], [self.ann2, self.ann3]))
        self.assertEqual(2, granularity([self.ann1], [self.ann2, self.ann3, self.ann4]))
        self.assertEqual(
            1.5, granularity([self.ann1, self.ann3], [self.ann2, self.ann4])
        )

    def test_plagdet_score(self):
        self.assertEqual(0, plagdet_score(-1, 0, 0))
        self.assertEqual(0, plagdet_score(0, -1, 0))
        self.assertEqual(0, plagdet_score(0, 0, -1))
        self.assertEqual(0, plagdet_score(0, 0, 1))
        self.assertEqual(0, plagdet_score(0, 1, 1))
        self.assertEqual(0, plagdet_score(1, 0, 1))
        self.assertEqual(1, plagdet_score(1, 1, 1))
        self.assertEqual(2 / 3, plagdet_score(0.5, 1, 1))
        self.assertEqual(2 / 3, plagdet_score(1, 0.5, 1))
        self.assertAlmostEqual(0.63092975, plagdet_score(1, 1, 2))
        self.assertAlmostEqual(0.23659865, plagdet_score(0.25, 0.75, 2))

    def test_is_overlapping(self):
        self.assertTrue(is_overlapping(self.ann1, self.ann2))
        self.assertFalse(is_overlapping(self.ann1, self.ann3))
        self.assertTrue(is_overlapping(self.ann1, self.ann4))
        self.assertFalse(is_overlapping(self.ann1, self.ann7))
        self.assertFalse(is_overlapping(self.ann1, self.ann8))
        self.assertFalse(is_overlapping(self.ann1, self.ann9))
        self.assertFalse(is_overlapping(self.ann1, self.ann10))
        self.assertTrue(is_overlapping(self.ann1, self.ann5))
        self.assertTrue(is_overlapping(self.ann1, self.ann6))

    def test_index_annotations(self):
        index = index_annotations([self.ann1, self.ann7, self.ann2, self.ann8])
        self.assertEqual([self.ann1, self.ann2], index.get("tref1"))
        self.assertEqual([self.ann7, self.ann8], index.get("tref2"))


def usage():
    """打印命令行用法说明。"""
    print(
        """\
Usage: perfmeasures.py [options]

Options:
      --micro      Compute micro-averaged recall and precision,
                   default: macro-averaged recall and precision
  -p, --plag-path  Path to the XML files with plagiarism annotations
      --plag-tag   Tag name suffix of plagiarism annotations,
                   default: 'plagiarism'
  -d, --det-path   Path to the XML files with detection annotations
      --det-tag    Tag name of the detection annotations,
                   default: 'detected-plagiarism'
  -h, --help       Show this message
"""
    )


def parse_options():
    """解析命令行参数。"""
    try:
        long_options = [
            "micro",
            "plag-path=",
            "plag-tag=",
            "det-path=",
            "det-tag=",
            "help",
        ]
        opts, _ = getopt.getopt(sys.argv[1:], "p:d:h", long_options)
    except:
        usage()
        sys.exit(2)
    micro_averaged = False
    plag_path, det_path = "undefined", "undefined"
    plag_tag_name, det_tag_name = "plagiarism", "detected-plagiarism"
    for opt, arg in opts:
        if opt in ("--micro"):
            micro_averaged = True
        elif opt in ("-p", "--plag-path"):
            plag_path = arg
        elif opt == "--plag-tag":
            plag_tag_name = arg
        elif opt in ("-d", "--det-path"):
            det_path = arg
        elif opt == "--det-tag":
            det_tag_name = arg
        elif opt in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "Unknown option."
    if plag_path == "undefined":
        print("Plagiarism path undefined. Use option -p or --plag-path.")
        sys.exit()
    if det_path == "undefined":
        print("Detections path undefined. Use option -d or --det-path.")
        sys.exit()
    return (micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name)


def main(micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name):
    """模块入口：读取标注与检测结果并计算指标。"""
    print("Reading", plag_path)
    cases = extract_annotations_from_files(plag_path, plag_tag_name)
    print("Have", len(cases), "cases")
    print("Reading", det_path)
    detections = extract_annotations_from_files(det_path, det_tag_name)
    print("Have", len(detections), "detections")
    print("Processing... (this may take a while)")
    rec, prec = 0, 0
    if micro_averaged:
        rec, prec = micro_avg_recall_and_precision(cases, detections)
    else:
        rec, prec = macro_avg_recall_and_precision(cases, detections)
    gran = granularity(cases, detections)
    print("Plagdet Score", plagdet_score(rec, prec, gran))
    print("Recall", rec)
    print("Precision", prec)
    print("Granularity", gran)


if __name__ == "__main__":
    main(*parse_options())
