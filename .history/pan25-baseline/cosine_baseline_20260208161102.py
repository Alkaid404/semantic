"""
MIT License

Copyright (c) 2024 Jan Philip Wahle (https://jpwahle.com/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import re
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 基础配置
BASE_PATH = "./pan25-plag_detect_test"

EMBEDDING_THRESHOLDS = {
    "meta-llama_Llama-3.3-70B-Instruct": 0.727,  # 预先计算的阈值
    "Linq-AI-Research_Linq-Embed-Mistral": 0.697,  # 预先计算的阈值
    "Alibaba-NLP_gte-Qwen2-7B-instruct": 0.596,  # 预先计算的阈值
}


def load_pairs(pairs_file_path: str):
    """从 pairs 文件读取可疑文档与源文档的配对列表。"""
    with open(pairs_file_path, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    return pairs

def load_embeddings(susp_doc, src_doc, embeddings_path):
    """读取可疑文档与源文档的预计算段落嵌入。"""
    susp_emb_file = os.path.join(embeddings_path, f"{susp_doc}.npy")
    src_emb_file = os.path.join(embeddings_path, f"{src_doc}.npy")
    susp_emb = np.load(susp_emb_file, allow_pickle=True)
    src_emb = np.load(src_emb_file, allow_pickle=True)
    return susp_emb, src_emb

def paragraph_chunking(text):
    """按段落切分文本，并丢弃 references/bibliography 等尾部段落。"""

    # 先移除参考文献段，避免其干扰段落对齐
    references_pattern = r"(?si)(?:\n\n+|^)(?:references|bibliography|reference list|works cited)(?:\n\n+.*)?$"
    text = re.sub(references_pattern, "", text.strip())

    # 以空行切段，保持公式块与上下文在同一段
    paragraphs = re.split(r"\n\n(?!\s\n\s)", text)

    # 过滤空段落
    paragraphs = [p for p in paragraphs if p.strip()]

    return paragraphs

def get_plagiarism_annotations(susp_doc, src_doc, truths_path, base_path=BASE_PATH):
    """从真值 XML 中提取抄袭段落对，并映射为段落索引。"""
    truth_file = os.path.join(
        truths_path, f"{os.path.splitext(susp_doc)[0]}-{os.path.splitext(src_doc)[0]}.xml"
    )
    if not os.path.exists(truth_file):
        return set()
    
    tree = ET.parse(truth_file)
    root = tree.getroot()
    plagiarism_pairs = set()
    
    for feature in root.findall(".//feature[@name='plagiarism']"):
        susp_offset = int(feature.get("this_offset"))
        src_offset = int(feature.get("source_offset"))

        # 将字符偏移映射到段落索引
        susp_file_path = os.path.join(base_path, "susp", f"{os.path.splitext(susp_doc)[0]}.txt")
        src_file_path = os.path.join(base_path, "src", f"{os.path.splitext(src_doc)[0]}.txt")
        
        # 将字符偏移定位到所属段落
        def map_offset_to_paragraph(file_path, offset):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 复用段落切分逻辑，保证与嵌入生成一致
            paragraphs = paragraph_chunking(text)
            
            # 累积段落长度，定位 offset 落在哪一段
            current_pos = 0
            for i, para in enumerate(paragraphs):
                para_len = len(para) + 2  # +2 计入段落之间的换行
                if current_pos <= offset < current_pos + para_len:
                    return i
                current_pos += para_len
            
            return -1  # 未找到对应段落
            
        # 计算抄袭对应的段落索引对
        susp_para_idx = map_offset_to_paragraph(susp_file_path, susp_offset)
        src_para_idx = map_offset_to_paragraph(src_file_path, src_offset)

        plagiarism_pairs.add((susp_para_idx, src_para_idx))
    
    return plagiarism_pairs

def compute_similarity_matrix(susp_emb, src_emb):
    """计算可疑段落与源段落的余弦相似度矩阵。"""
    return cosine_similarity(susp_emb, src_emb)

def collect_similarities(pairs, embeddings_path, truths_path, base_path=BASE_PATH):
    """统计抄袭/非抄袭段落对的相似度分布。"""
    plag_similarities = []
    non_plag_similarities = []
    
    for susp_doc, src_doc in tqdm(pairs, desc="Processing pairs"):
        # 读取段落嵌入
        try:
            susp_emb, src_emb = load_embeddings(susp_doc, src_doc, embeddings_path)
        except Exception as e:
            print(f"Error loading embeddings for {susp_doc}-{src_doc}: {e}")
            continue
        if susp_emb.size == 0 or src_emb.size == 0:
            continue
        
        # 计算相似度矩阵
        sim_matrix = compute_similarity_matrix(susp_emb, src_emb)
        
        # 读取真值段落对
        plag_pairs = get_plagiarism_annotations(susp_doc, src_doc, truths_path, base_path=base_path)
        
        # 遍历矩阵，按真值标注归类相似度
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                sim = sim_matrix[i, j]
                # 判断 (i, j) 是否为抄袭段落对
                is_plag = (i, j) in plag_pairs
                if is_plag:
                    plag_similarities.append(sim)
                else:
                    non_plag_similarities.append(sim)
    
    return np.array(plag_similarities), np.array(non_plag_similarities)

def plot_distributions(plag_sim, non_plag_sim):
    """绘制抄袭/非抄袭相似度分布图。"""
    plt.figure(figsize=(10, 6))
    sns.histplot(plag_sim, color='red', label='Plagiarism', kde=True, alpha=0.5)
    sns.histplot(non_plag_sim, color='blue', label='Non-Plagiarism', kde=True, alpha=0.5)
    plt.title("Cosine Similarity Distributions")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("cosine_similarity_distributions.png")

def find_optimal_threshold(plag_sim, non_plag_sim):
    """在 0~1 上搜索阈值，使抄袭与非抄袭分离度最大。"""
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_score = -1
    
    for t in thresholds:
        plag_above = np.mean(plag_sim >= t)
        non_plag_below = np.mean(non_plag_sim < t)
        score = plag_above + non_plag_below
        if score > best_score:
            best_score = score
            best_threshold = t
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Plagiarism above threshold: {np.mean(plag_sim >= best_threshold):.3f}")
    print(f"Non-plagiarism below threshold: {np.mean(non_plag_sim < best_threshold):.3f}")
    return best_threshold

def find_text_offset(document, text, label=""):
    """在原文中定位子串的字符偏移与长度，必要时进行模糊匹配。"""
    text = text.strip()
    if not text:
        return None
    try:
        offset = document.find(text)
        if offset != -1:
            return offset, len(text)
        norm_text = re.sub(r"\s+", " ", text).strip()
        norm_document = re.sub(r"\s+", " ", document)
        offset = norm_document.find(norm_text)
        if offset != -1:
            non_space_count = len(re.sub(r"\s", "", norm_document[:offset]))
            orig_offset = -1
            current_non_space = 0
            for i, char in enumerate(document):
                if not char.isspace():
                    current_non_space += 1
                if current_non_space > non_space_count:
                    search_start = max(0, i - len(text) // 2)
                    orig_offset = document.find(text, search_start)
                    if orig_offset != -1:
                        found_norm = re.sub(
                            r"\s+", " ", document[orig_offset : orig_offset + len(text)]
                        ).strip()
                        if SequenceMatcher(None, norm_text, found_norm).ratio() > 0.95:
                            return orig_offset, len(text)
                        else:
                            orig_offset = -1
                    break
            if orig_offset != -1:
                return orig_offset, len(text)
        matcher = SequenceMatcher(None, document, text, autojunk=False)
        match = matcher.find_longest_match(0, len(document), 0, len(text))
        if match.size >= len(text) * 0.85 and matcher.ratio() > 0.85:
            return match.a, match.size
        else:
            print(
                f"Warning: Could not reliably find {label} text '{text[:50]}...' (ratio: {matcher.ratio():.2f}, size: {match.size}) in document."
            )
            return None
    except Exception as e:
        print(f"Error finding offset for {label} text '{text[:50]}...': {e}")
        return None

def generate_xml(
    detections,
    susp_doc_name,
    src_doc_name,
    output_path,
    full_susp_text,
    full_src_text,
    params=None,
):
    """将检测到的段落对写成 PAN XML 格式文件。"""
    if params is None:
        params = {}
    root = ET.Element("document", reference=susp_doc_name)
    ET.SubElement(
        root,
        "feature",
        name="about",
        authors="",
        title="",
        lang="en",
        similarity="",
        severity="",
        prompt_tokens="",
        output_tokens="",
    )
    detection_count = 0
    for susp_text_det, src_text_det in detections:
        susp_result = find_text_offset(full_susp_text, susp_text_det, "suspicious")
        src_result = find_text_offset(full_src_text, src_text_det, "source")
        if susp_result and src_result:
            susp_offset, susp_length = susp_result
            src_offset, src_length = src_result
            ET.SubElement(
                root,
                "feature",
                name="plagiarism",
                type="paraphrase",
                this_language="en",
                this_offset=str(susp_offset),
                this_length=str(susp_length),
                source_reference=src_doc_name,
                source_language="en",
                source_offset=str(src_offset),
                source_length=str(src_length),
            )
            detection_count += 1
        else:
            print(
                f"Skipping detection: <susp>{susp_text_det[:50]}...</susp><src>{src_text_det[:50]}...</src>"
            )
    # 在输出文件名中追加参数信息，便于区分不同设置
    output_suffix = "".join(
        [f"-{k}{v}" for k, v in params.items() if k not in ["llm_model"]]
    )
    output_file_base = f"{os.path.splitext(susp_doc_name)[0]}-plagiarized-{os.path.splitext(src_doc_name)[0]}{output_suffix}.xml"
    output_file = os.path.join(output_path, output_file_base)
    try:
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"Generated detection file ({detection_count} features): {output_file}")
    except Exception as e:
        print(f"Error writing XML file {output_file}: {e}")
        return None
    return output_file

def predict_and_save_detections(
    pairs,
    threshold,
    embeddings_path,
    output_dir="detections_cosine",
    base_path=BASE_PATH,
):
    """根据阈值生成检测结果，每个可疑段只保留最相似的源段。"""
    
    # 从嵌入目录推断模型名称，拼到输出目录
    model_name = os.path.basename(embeddings_path).replace("embeddings_", "")
    output_dir = f"detections_cosine_embeddings_{model_name}"
    
    # 创建输出目录
    os.makedirs(os.path.join(base_path, output_dir), exist_ok=True)
    
    print(f"Generating predictions using threshold: {threshold:.3f}")
    
    for susp_doc, src_doc in tqdm(pairs, desc="Processing pairs"):
        
        try:
            # 读取嵌入
            susp_emb, src_emb = load_embeddings(susp_doc, src_doc, embeddings_path)
        except Exception as e:
            print(f"Error loading embeddings for {susp_doc}-{src_doc}: {e}. Skipping pair.")
            continue
        
        if susp_emb.size == 0 or src_emb.size == 0:
            print(f"Skipping {susp_doc}-{src_doc}: embeddings not found")
            continue
        
        # 读取原文全文
        with open(os.path.join(base_path, "susp", susp_doc), "r", encoding="utf-8") as f:
            full_susp_text = f.read()
        with open(os.path.join(base_path, "src", src_doc), "r", encoding="utf-8") as f:
            full_src_text = f.read()
        
        # 切分段落
        susp_paragraphs = paragraph_chunking(full_susp_text)
        src_paragraphs = paragraph_chunking(full_src_text)
        
        # 计算相似度矩阵
        sim_matrix = compute_similarity_matrix(susp_emb, src_emb)
        
        # 收集超过阈值的段落对
        detections = []
        # 每个可疑段只保留最相似的源段
        for i in range(sim_matrix.shape[0]):
            if i >= len(susp_paragraphs):
                continue
                
            # 选择相似度最高的源段落
            max_sim_idx = np.argmax(sim_matrix[i])
            max_sim = sim_matrix[i, max_sim_idx]
            
            # 仅当相似度超过阈值才加入检测结果
            if max_sim >= threshold and max_sim_idx < len(src_paragraphs):
                detections.append((susp_paragraphs[i], src_paragraphs[max_sim_idx]))
        
        # 生成 XML
        params = {
            "threshold": f"{threshold:.3f}",
            "method": "cosine",
        }
        
        output_file = generate_xml(
            detections, susp_doc, src_doc, 
            os.path.join(base_path, output_dir),
            full_susp_text, full_src_text, params
        )
        
    print(f"Predictions saved to {os.path.join(base_path, output_dir)}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run plagiarism detection with cosine similarity')
    parser.add_argument('--test_embeddings', type=str, required=True, help='Path to test embeddings directory.')
    parser.add_argument('--test_pairs_file', type=str, required=True, help='Path to test pairs file.')
    parser.add_argument('--test_truths_path', type=str, required=False, help='Path to test truths directory.')
    parser.add_argument('--train_embeddings', type=str, required=False, help='Path to train embeddings directory, if not provided, use precomputed thresholds.')
    parser.add_argument('--train_pairs_file', type=str, required=False, help='Path to train pairs file. Required if --train_embeddings is provided.')
    parser.add_argument('--train_truths_path', type=str, required=False, help='Path to train truths directory. Required if --train_embeddings is provided.')
    parser.add_argument('--base_path', type=str, default=BASE_PATH, help='Base path for src/susp text and outputs.')
    args = parser.parse_args()
    
    # 读取测试集配对
    test_pairs = load_pairs(args.test_pairs_file)
    print(f"Loaded {len(test_pairs)} test pairs")
    
    if args.train_embeddings:
        if not args.train_truths_path:
            parser.error("--train_truths_path is required when using --train_embeddings")
        if not args.train_pairs_file:
            parser.error("--train_pairs_file is required when using --train_embeddings")
            
        # 读取训练集配对
        train_pairs = load_pairs(args.train_pairs_file)
        print(f"Loaded {len(train_pairs)} train pairs")
        
        # 用训练集统计分布并寻找最优阈值
        plag_sim, non_plag_sim = collect_similarities(
            train_pairs,
            args.train_embeddings,
            args.train_truths_path,
            base_path=args.base_path,
        )
        threshold = find_optimal_threshold(plag_sim, non_plag_sim)
    else:
        # 使用预置阈值
        model_name = os.path.basename(args.test_embeddings).replace("embeddings_", "")
        threshold = EMBEDDING_THRESHOLDS[model_name]

    # 使用测试集生成检测结果
    predict_and_save_detections(
        test_pairs,
        threshold,
        args.test_embeddings,
        base_path=args.base_path,
    )

if __name__ == "__main__":
    main()