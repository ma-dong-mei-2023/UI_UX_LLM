@echo off
echo ================================================
echo   LLM 从零开始 - 交互式学习平台
echo ================================================
echo.

echo [1/2] 启动后端 (FastAPI + PyTorch)...
start "LLM Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo 等待后端启动...
timeout /t 3 /nobreak >nul

echo [2/2] 启动前端 (Next.js)...
start "LLM Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ================================================
echo   应用已启动:
echo   前端: http://localhost:3000
echo   后端 API: http://localhost:8000
echo   API 文档: http://localhost:8000/docs
echo ================================================
echo.
pause
