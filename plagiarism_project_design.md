# 语义结构级文本查重系统 --- 项目设计文档

## 一、项目目标

构建一个**结构级语义查重系统**，支持用户输入原始文本，并导入一个或多个疑似抄袭文档，通过
**chunk-level embedding + FAISS + CrossEncoder rerank**
实现高精度语义匹配，最终返回：

-   疑似抄袭片段高亮
-   文档整体查重率
-   语义相似度评分
-   可扩展的后端检索架构

本项目采用**前后端分离架构**，暂不包含实验评估模块。

------------------------------------------------------------------------

# 二、技术选型

## 后端

-   **FastAPI**：高性能异步接口
-   **FAISS**：向量检索
-   **SentenceTransformers / GloVe**：语义向量
-   **CrossEncoder**：精排提高准确率
-   **SQLite / PostgreSQL（可升级）**：元数据存储

## 前端

-   **React + Vite**
-   **Axios**
-   **Tailwind / Ant Design（任选）**

------------------------------------------------------------------------

# 三、整体架构

    Frontend (React)
          │
          ▼
    FastAPI Gateway
          │
     ├── Embedding Service
     ├── Retrieval Service (FAISS)
     ├── Rerank Service (CrossEncoder)
     └── Similarity Engine

系统流程：

1.  用户上传原文 + 疑似文档
2.  后端进行 chunk 切分
3.  生成向量
4.  FAISS 粗召回
5.  CrossEncoder 精排
6.  计算查重率
7.  返回高亮结果

------------------------------------------------------------------------

# 四、项目目录结构（核心）

    plagiarism-detector/
    │
    ├── backend/
    │   ├── app/
    │   │   ├── main.py
    │   │   ├── api/
    │   │   ├── core/
    │   │   ├── services/
    │   │   ├── models/
    │   │   ├── schemas/
    │   │   └── utils/
    │   │
    │   ├── vector_store/
    │   └── requirements.txt
    │
    ├── frontend/
    │   ├── src/
    │   │   ├── components/
    │   │   ├── pages/
    │   │   ├── services/
    │   │   └── App.jsx
    │
    └── README.md

------------------------------------------------------------------------

# 五、后端详细设计

## 1️⃣ `main.py` --- 服务入口

**职责：** - 初始化 FastAPI - 注册路由 - 加载模型 - 启动时加载 FAISS

**接口：**

``` python
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

------------------------------------------------------------------------

## 2️⃣ api/ --- 路由层

### `routes_plagiarism.py`

**职责：** - 接收文件上传 - 调用查重服务 - 返回结果

**接口：**

### POST `/check`

    Request:
    {
      "source_text": "...",
      "suspects": ["doc1", "doc2"]
    }

    Response:
    {
      "similarity": 0.42,
      "matches": [
        {
          "source_chunk": "...",
          "suspect_chunk": "...",
          "score": 0.83
        }
      ]
    }

------------------------------------------------------------------------

## 3️⃣ services/ --- 核心算法层

这是项目最重要的部分。

------------------------------------------------------------------------

### `chunker.py`

**职责：** - 将文本按语义切块

**接口：**

``` python
def chunk_text(text: str, chunk_size=200, overlap=50) -> list[str]:
    pass
```

建议：使用 sliding window，避免语义断裂。

------------------------------------------------------------------------

### `embedding_service.py`

**职责：** - 加载 embedding 模型 - 批量生成向量

**接口：**

``` python
class EmbeddingService:
    def encode(self, texts: list[str]) -> np.ndarray:
        pass
```

优化建议： - GPU 推理 - batch encode

------------------------------------------------------------------------

### `faiss_service.py`

**职责：** - 管理向量索引 - 提供相似度搜索

**接口：**

``` python
class FaissService:

    def add_vectors(self, vectors):
        pass

    def search(self, query_vectors, top_k=5):
        pass
```

建议： - 初期：IndexFlatL2 - 中期：IVF / HNSW

------------------------------------------------------------------------

### `rerank_service.py`

**职责：** - CrossEncoder 精排，提高查准率

**接口：**

``` python
class RerankService:

    def rerank(self, pairs):
        pass
```

作用： 解决 embedding "语义接近但不抄袭"的误判问题。

------------------------------------------------------------------------

### ⭐ `similarity_engine.py`（系统大脑）

**职责：** - 串联整个查重流程

**接口：**

``` python
class SimilarityEngine:

    def check_plagiarism(self, source, suspects):
        '''
        Pipeline:
        chunk → embed → recall → rerank → compute score
        '''
```

查重率建议算法：

    plagiarism_rate =
        matched_chunks / total_source_chunks

可升级为 **加权 token 覆盖率**。

------------------------------------------------------------------------

## 4️⃣ schemas/ --- 数据协议

### `plagiarism_schema.py`

``` python
class Match(BaseModel):
    source_chunk: str
    suspect_chunk: str
    score: float

class PlagiarismResponse(BaseModel):
    similarity: float
    matches: list[Match]
```

作用：

✔ 自动生成 OpenAPI\
✔ 前后端类型一致

------------------------------------------------------------------------

## 5️⃣ utils/

### `file_parser.py`

支持：

-   txt
-   pdf（后续）
-   docx（后续）

接口：

``` python
def parse_file(file) -> str:
    pass
```

------------------------------------------------------------------------

# 六、向量存储目录

    vector_store/
    ├── faiss.index
    └── metadata.db

职责：

-   持久化索引
-   避免每次重建

------------------------------------------------------------------------

# 七、前端设计

## 页面布局

    ┌──────────────┬──────────────┐
    │ 原始文本输入 │ 疑似文档上传 │
    │              │              │
    └──────────────┴──────────────┘
            ↓ 点击查重
            ↓
       返回高亮结果 + 查重率

------------------------------------------------------------------------

## 组件结构

### components/

#### `TextUploader.jsx`

-   输入文本
-   上传 txt

#### `SuspectUploader.jsx`

-   支持多个文件

#### `ResultViewer.jsx`

-   高亮重复片段
-   显示相似度

------------------------------------------------------------------------

### services/api.js

``` javascript
export async function checkPlagiarism(data){
    return axios.post("/check", data)
}
```

------------------------------------------------------------------------

# 八、系统可扩展方向（强烈建议复试讲）

如果面试官问深度，这部分是关键。

### ⭐ 升级1：结构级查重

-   段落图匹配
-   discourse similarity

### ⭐ 升级2：多阶段检索

    Embedding Recall → CrossEncoder → LLM Judge

### ⭐ 升级3：分布式向量库

-   Milvus
-   Weaviate

### ⭐ 升级4：GPU 推理优化

------------------------------------------------------------------------

# 九、项目亮点总结（可直接复述）

> 本系统不是简单 API 调用，而是构建了一套完整的语义检索架构，通过
> chunk-level embedding 与 CrossEncoder
> 精排，实现结构级文本查重，在保证召回率的同时显著提高查准率，并具备良好的工程扩展性。

------------------------------------------------------------------------

# 十、推荐开发顺序（避免踩坑）

1️⃣ FastAPI skeleton\
2️⃣ chunk + embedding 跑通\
3️⃣ FAISS 检索\
4️⃣ rerank\
5️⃣ similarity engine\
6️⃣ 前端页面\
7️⃣ 高亮显示

**不要一开始就追求复杂模型。**

工程 \> 模型。

------------------------------------------------------------------------


