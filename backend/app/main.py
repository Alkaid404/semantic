"""FastAPI 入口。

职责：
- 初始化 FastAPI
- 注册路由
- 加载模型
- 启动时加载 FAISS 索引
"""
from fastapi import FastAPI

from .api.routes_plagiarism import router as plagiarism_router

app = FastAPI(title="Plagiarism Detector")


@app.get("/health")
def health_check():
    """轻量健康检查。"""
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    """启动时加载模型和向量索引。"""
    # TODO: 加载 embedding 模型、cross-encoder 和 FAISS 索引。
    pass


app.include_router(plagiarism_router)
