"""后端启动脚本。

用法:
    python -m backend.main
    # 或
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
