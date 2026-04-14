#!/bin/bash
echo "================================================"
echo "  LLM 从零开始 - 交互式学习平台"
echo "================================================"
echo ""

# Start backend
echo "[1/2] 启动后端 (FastAPI + PyTorch)..."
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

sleep 2

# Start frontend
echo "[2/2] 启动前端 (Next.js)..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "================================================"
echo "  应用已启动:"
echo "  前端: http://localhost:3000"
echo "  后端 API: http://localhost:8000"
echo "  API 文档: http://localhost:8000/docs"
echo "================================================"
echo ""
echo "按 Ctrl+C 停止所有服务"

wait $BACKEND_PID $FRONTEND_PID
