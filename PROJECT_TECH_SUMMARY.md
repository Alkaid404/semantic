# 语义查重系统 — 技术要点与项目总结

> 基于深度语义理解的文档抄袭检测系统，采用 **两阶段检索-精排架构**，实现高精度、可解释的全文查重。

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心流程（六步 Pipeline）](#3-核心流程六步-pipeline)
4. [技术要点与亮点](#4-技术要点与亮点)
5. [关键配置参数](#5-关键配置参数)
6. [技术栈](#6-技术栈)
7. [项目结构](#7-项目结构)

---

## 1. 项目概述

本项目是一个面向 **PAN25 抄袭检测评测** 场景的语义查重系统，核心思路是：

- **不依赖字面匹配**：传统查重工具依赖 n-gram / 指纹比对，对深层改写（paraphrase）检测能力弱。本系统通过 Transformer 编码文本语义向量，即使词汇完全不同也能捕捉语义相似。
- **两阶段架构**：先用轻量 Bi-Encoder 做大规模向量召回（O(n·m) → O(n·k)），再用重量级 Cross-Encoder 对候选对做精确打分，兼顾效率与精度。
- **全链路可解释**：返回的每个匹配对均含余弦分数（召回阶段）与 Rerank 分数（精排阶段），以及精确的字符偏移量，便于可视化与评测。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   React 前端                         │
│  文本/文件输入 → 调用 API → 高亮展示 + 明细表格       │
└──────────────────────┬──────────────────────────────┘
                       │  HTTP (Axios, 300s timeout)
                       ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI 后端 (Uvicorn)                   │
│                                                      │
│  routes_plagiarism.py                                │
│    ├─ POST /check       (JSON 文本查重)               │
│    ├─ POST /check/files (文件上传查重)                 │
│    └─ POST /check/xml   (PAN XML 格式输出)            │
│                                                      │
│  similarity_engine.py ← 核心编排器                    │
│    ├─ chunker.py          段落/滑动窗口切分            │
│    ├─ embedding_service.py  Sentence-Transformer 编码 │
│    ├─ faiss_service.py      FAISS 向量召回             │
│    └─ rerank_service.py     Cross-Encoder 精排         │
└─────────────────────────────────────────────────────┘
```

**前后端分离**：前端 React + Vite，后端 FastAPI + Uvicorn。模型在应用启动时通过 FastAPI lifespan 预加载，避免首次请求冷启动。

---

## 3. 核心流程（六步 Pipeline）

`SimilarityEngine.check()` 是系统的大脑，一次完整的查重经历以下六个步骤：

### Step 1: 段落切分（Chunking）

```
源文档 / 疑似文档 → chunk_text() → List[Chunk(text, offset, length)]
```

- **主模式：段落切分**（`use_paragraph_chunking=True`）  
  按空行 (`\n\n`) 分段，模拟 PAN25 baseline 的 `paragraph_chunking`。
- **备选模式：滑动窗口切分**  
  固定窗口大小（512 字符）+ 步长重叠（128 字符），适合无明显段落边界的文本。
- **最小字符阈值**：段落字符数 < 15 时丢弃，避免标题、编号等噪声。
- **元数据过滤**：`_is_metadata_chunk()` 通过正则识别作者行、邮箱、DOI、日期等元数据信息，过滤非正文内容，减少误匹配。
- **参考文献去除**：遇到 `References` / `Bibliography` 等标题时截断后续内容。

每个 Chunk 记录 `(text, offset, length)`，offset 用于后续高亮与 PAN XML 输出。

### Step 2: 语义编码（Embedding）

```
List[str] → SentenceTransformer.encode() → np.ndarray (N, 768)
```

- **模型**：`sentence-transformers/all-mpnet-base-v2`（110M 参数，768 维）
- **L2 归一化**：编码后的向量自动 L2 归一化，使得内积 = 余弦相似度。
- **批量编码**：`batch_size=64`，一次性编码全部 chunks。

### Step 3: 余弦召回（Cosine Recall）

```
sim_matrix = susp_emb @ src_emb.T    # 形状 (n_susp, n_src)
```

- **FAISS IndexFlatIP**（内积索引）：由于向量已 L2 归一化，内积即余弦相似度。
- **Top-K 筛选**：对每个疑似段落，取余弦最高的 k=5 个源段落。
- **阈值过滤**：余弦 < 0.55 的候选直接淘汰。
- **分数钳位**：`np.clip(score, 0.0, 1.0)`，防止浮点精度导致的越界值。

### Step 4: Cross-Encoder 精排（Rerank）

```
候选对 (susp_chunk, src_chunk) → CrossEncoder.predict() → sigmoid → 概率分数
```

- **模型**：`cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Sigmoid 归一化**：CrossEncoder 原始输出是 logits（范围约 -11 ~ +11），需经 sigmoid 映射到 [0, 1] 概率空间。
- **阈值过滤**：精排分 < 0.55 的候选被淘汰。
- 精排阶段的 Cross-Encoder 能同时看到两个文本，做 token-level 交互注意力，比 Bi-Encoder 的独立编码更精确。

### Step 5: 疑似段落去重（Per-Suspect Dedup）

每个疑似段落可能和多个源段落匹配。此步骤只保留 **rerank 分最高的一个匹配对**，避免同个疑似段落被重复报告。

### Step 6: 后处理（Post-Processing）

```
matches → _dedup_by_source() → _merge_adjacent_matches() → final matches
```

- **源段落去重** (`_dedup_by_source`)：同一个源段落被多个疑似段落匹配时，只保留得分最高的一对，解决"一源多配"冗余。
- **相邻区间合并** (`_merge_adjacent_matches`)：疑似文档中间距 ≤ 100 字符的匹配区间合并为一个大区间，分数按字符长度加权平均。解决"段落碎片化"问题。
- **文档级评分** (`_compute_document_score`)：  
  $$ \text{similarity} = 0.4 \times \overline{\text{score}} + 0.3 \times \frac{|\text{matched_src_chars}|}{|\text{src}|} + 0.3 \times \frac{|\text{matched_susp_chars}|}{|\text{susp}|} $$
  综合平均匹配分与双向字符覆盖率，比单纯取平均分更稳健。

### 全流程时序图

```
Source Doc ──┐                    Suspect Doc ──┐
             ▼                                  ▼
        ┌─────────┐                       ┌─────────┐
        │ Step 1  │  paragraph_chunking   │ Step 1  │
        │ Chunk   │                       │ Chunk   │
        └────┬────┘                       └────┬────┘
             ▼                                  ▼
        ┌─────────┐                       ┌─────────┐
        │ Step 2  │  SentenceTransformer  │ Step 2  │
        │ Encode  │  all-mpnet-base-v2    │ Encode  │
        └────┬────┘                       └────┬────┘
             │           ┌──────────┐          │
             └──────────►│ Step 3   │◄─────────┘
                         │ Cosine   │
                         │ Recall   │  top-k=5, threshold=0.55
                         └────┬─────┘
                              ▼
                         ┌──────────┐
                         │ Step 4   │
                         │ Rerank   │  CrossEncoder + sigmoid
                         └────┬─────┘
                              ▼
                         ┌──────────┐
                         │ Step 5   │
                         │ Dedup    │  per-suspect best match
                         └────┬─────┘
                              ▼
                         ┌──────────┐
                         │ Step 6   │
                         │ Post-    │  source dedup + merge
                         │ process  │  + document scoring
                         └────┬─────┘
                              ▼
                      PlagiarismResult
                  (similarity, matches[])
```

---

## 4. 技术要点与亮点

### 4.1 两阶段检索-精排架构

| 阶段 | 模型                                    | 复杂度                    | 作用           |
| ---- | --------------------------------------- | ------------------------- | -------------- |
| 召回 | Bi-Encoder (all-mpnet-base-v2)          | O(n+m) 编码 + O(n·k) 检索 | 快速缩小候选集 |
| 精排 | Cross-Encoder (ms-marco-MiniLM-L-12-v2) | O(candidates)             | 精确语义匹配   |

Bi-Encoder 独立编码两段文本再做点积，速度快但精度有限；Cross-Encoder 将两段文本拼接后联合编码，做 full attention 交互，精度高但速度慢。两阶段结合实现 **效率与精度的平衡**。

### 4.2 Sigmoid 归一化

Cross-Encoder 的 `model.predict()` 返回的是 **原始 logits**，不是概率值，范围约 [-11, +11]。直接作为分数会导致：

- 分数可能 > 1.0（显示超过 100%）
- 无法与余弦阈值统一比较

**解决方案**：在 rerank_service 中对 logits 施加 sigmoid 映射：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

映射后分数严格落在 [0, 1]，可直接与阈值比较、作为百分制展示。

### 4.3 元数据智能过滤

学术文档中的作者行（如 `"John Doe, Jane Smith"`）、邮箱、DOI、日期等元数据，虽然语义结构类似（"人名列表"），但不应被视为抄袭。

`_is_metadata_chunk()` 通过多条件识别：

- **正则模式**：检测邮箱、DOI、ISO 日期格式
- **作者行模式**：`"名 姓, 名 姓"` 多人列表格式
- **字母密度**：字母字符 < 50% 的文本块视为非正文（如参考文献编号、表格数据）

### 4.4 后处理优化

| 问题                                  | 解决方案                                                   | 效果               |
| ------------------------------------- | ---------------------------------------------------------- | ------------------ |
| 同源多配：一个源段落命中 7 个疑似段落 | `_dedup_by_source()`：按 (offset, length) 去重，保留最高分 | 消除冗余匹配       |
| 段落碎片化：相邻的匹配段落被分开报告  | `_merge_adjacent_matches(gap≤100)`：合并相邻区间           | 输出更连贯         |
| 评分失真：只看平均分忽略覆盖范围      | 双向覆盖率 + 平均分加权                                    | 更准确反映抄袭程度 |

### 4.5 字符级偏移量定位

从切分到最终结果，全程追踪每个 chunk 的 `(offset, length)`：

- **PAN XML 兼容**：可直接生成评测系统所需的 `this_offset / source_offset` 格式
- **前端高亮**：基于偏移量的 `buildHighlightSegments()` 将原始文本切割为高亮/普通片段，支持重叠区间合并

### 4.6 全流程性能日志

每个步骤均记录耗时（`logger.info("StepN xxx: %.2fs")`），便于快速定位性能瓶颈。典型耗时分布：

| 步骤                 | 典型耗时 | 说明                 |
| -------------------- | -------- | -------------------- |
| Step1 Chunking       | < 0.01s  | 纯文本处理           |
| Step2 Encoding       | 1~5s     | 取决于段落数量和设备 |
| Step3 Cosine Recall  | < 0.1s   | 矩阵乘法             |
| Step4 Rerank         | 0.5~3s   | 取决于候选对数量     |
| Step5+6 Post-process | < 0.01s  | 内存操作             |

### 4.7 结果持久化

每次查重结果自动以 JSON 格式保存到 `backend/log/` 目录，文件名包含时间戳（精确到微秒），便于回溯分析和评测对比。

### 4.8 多格式输出

- **JSON API**：`/check` 和 `/check/files` 返回结构化 JSON，含所有匹配对和元信息
- **PAN XML**：`/check/xml` 返回 PAN 评测系统兼容的 XML 格式，可直接提交评测
- **前端可视化**：高亮标注 + 对比网格 + 明细表格

---

## 5. 关键配置参数

| 参数                     | 默认值                                    | 说明                                |
| ------------------------ | ----------------------------------------- | ----------------------------------- |
| `embedding_model_name`   | `sentence-transformers/all-mpnet-base-v2` | Bi-Encoder 模型 (110M 参数, 768 维) |
| `rerank_model_name`      | `cross-encoder/ms-marco-MiniLM-L-12-v2`   | Cross-Encoder 精排模型              |
| `similarity_threshold`   | 0.55                                      | 余弦相似度召回阈值                  |
| `rerank_threshold`       | 0.55                                      | Rerank 精排阈值 (sigmoid 后概率值)  |
| `faiss_top_k`            | 5                                         | 每个疑似段落召回的源段落数          |
| `chunk_size`             | 512                                       | 滑动窗口切分的窗口大小（字符）      |
| `chunk_overlap`          | 128                                       | 滑动窗口切分的重叠大小（字符）      |
| `use_paragraph_chunking` | True                                      | 是否使用段落切分（否则用滑动窗口）  |
| `min_chars`              | 15                                        | 段落最小字符数阈值                  |
| `gap_threshold`          | 100                                       | 相邻区间合并的最大间距（字符）      |

所有阈值均支持环境变量覆盖，便于部署时调参。

---

## 6. 技术栈

| 层            | 技术                                      | 用途                |
| ------------- | ----------------------------------------- | ------------------- |
| **前端**      | React + Vite + Axios                      | 交互界面、高亮展示  |
| **后端框架**  | FastAPI + Uvicorn                         | RESTful API 服务    |
| **Embedding** | sentence-transformers (all-mpnet-base-v2) | 语义向量编码        |
| **Rerank**    | CrossEncoder (ms-marco-MiniLM-L-12-v2)    | 精排打分            |
| **向量检索**  | FAISS (IndexFlatIP)                       | 高效内积相似度计算  |
| **数值计算**  | NumPy                                     | 矩阵运算、归一化    |
| **评测**      | PAN XML + Docker 评测脚本                 | 兼容 PAN25 评测标准 |

---

## 7. 项目结构

```
semantic/
├── backend/
│   ├── main.py                     # Uvicorn 启动入口
│   └── app/
│       ├── main.py                 # FastAPI 应用 (lifespan 预加载模型)
│       ├── core/__init__.py        # Settings 全局配置
│       ├── api/
│       │   └── routes_plagiarism.py # 路由 (/check, /check/files, /check/xml)
│       ├── services/
│       │   ├── chunker.py          # 段落切分 + 元数据过滤
│       │   ├── embedding_service.py # SentenceTransformer 编码
│       │   ├── faiss_service.py    # FAISS 索引 + cosine matrix
│       │   ├── rerank_service.py   # CrossEncoder + sigmoid 归一化
│       │   └── similarity_engine.py # 六步流程编排器 (系统大脑)
│       ├── schemas/
│       │   └── plagiarism_schema.py # Pydantic 请求/响应模型
│       └── utils/
│           ├── file_parser.py      # 文件解析
│           └── xml_output.py       # PAN XML 格式生成
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # 主页面 (输入 + 高亮 + 表格)
│   │   ├── services/api.js         # Axios 封装 + 高亮片段构建
│   │   └── index.css               # 样式
│   └── vite.config.js
├── evaluation/
│   ├── run_pan25_eval.py           # PAN25 评测脚本
│   └── Dockerfile                  # 评测环境容器
├── start.sh                        # 一键启动脚本
├── pyproject.toml                  # Python 项目配置
└── PROJECT_TECH_SUMMARY.md         # 本文件
```

---

