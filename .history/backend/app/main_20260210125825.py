"""FastAPI 入口。

职责：
- 初始化 FastAPI
- 注册路由
- 启动时预加载模型（embedding + cross-encoder）
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes_plagiarism import _engine, router as plagiarism_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """在应用启动时预加载模型，加速首次请求。"""
    # 预热 embedding 模型
    _engine._embedder.encode(["warmup"], normalize=True)
    # 预热 cross-encoder
    _engine._reranker.rerank([("warmup a", "warmup b")], threshold=-999)
    print("✓ 模型加载完成")
    yield


app = FastAPI(title="Plagiarism Detector", lifespan=lifespan)

# CORS — 允许前端开发服务器通信
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """轻量健康检查。"""
    return {"status": "ok"}


app.include_router(plagiarism_router)
