#!/usr/bin/env bash
# 一键启动前后端服务
# 用法: ./start.sh
# 停止: Ctrl+C（会同时终止前后端进程）

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

# 颜色
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
  echo -e "\n${YELLOW}正在停止所有服务...${NC}"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
  echo -e "${GREEN}所有服务已停止${NC}"
}
trap cleanup EXIT INT TERM

# ---------- 后端 ----------
echo -e "${CYAN}[1/2] 启动后端 (FastAPI :8000)...${NC}"
cd "$BACKEND_DIR"

# 安装依赖（仅缺失时会安装）
uv pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q

# 启动后端（后台运行，日志带前缀）
uv run python main.py 2>&1 | sed "s/^/[backend] /" &
BACKEND_PID=$!

# ---------- 前端 ----------
echo -e "${CYAN}[2/2] 启动前端 (Vite :5173)...${NC}"
cd "$FRONTEND_DIR"

# 安装依赖（仅 node_modules 不存在时）
if [ ! -d "node_modules" ]; then
  echo -e "${YELLOW}安装前端依赖...${NC}"
  npm install --silent
fi

npm run dev 2>&1 | sed "s/^/[frontend] /" &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  项目已启动！${NC}"
echo -e "${GREEN}  前端: http://localhost:5173${NC}"
echo -e "${GREEN}  后端: http://localhost:8000${NC}"
echo -e "${GREEN}  API 文档: http://localhost:8000/docs${NC}"
echo -e "${GREEN}  按 Ctrl+C 停止所有服务${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

wait
