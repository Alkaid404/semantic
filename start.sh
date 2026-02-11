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

# 依赖已通过根目录 pyproject.toml 管理

# 启动后端（后台运行，日志带前缀）
uv run python main.py 2>&1 | sed "s/^/[backend] /" &
BACKEND_PID=$!

echo -e "${CYAN}等待后端初始化完成 (加载 AI 模型可能需要 10-30 秒)...${NC}"
# 循环检查后端端口是否响应 (检查 /docs 接口)
while ! curl -s --head --request GET http://localhost:8000/docs | grep "200 OK" > /dev/null; do
  sleep 2
done
echo -e "${GREEN}后端已就绪！${NC}"

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
