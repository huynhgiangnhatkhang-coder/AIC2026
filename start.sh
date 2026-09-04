#!/bin/bash

echo "Đang khởi động Backend (FastAPI) trên cổng 8000..."
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Đang khởi động Frontend (Vite) trên cổng 5173..."
cd AIChallenge2026-master
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
npm run dev -- --host &
FRONTEND_PID=$!

cd ..

echo "=================================================="
echo "Hệ thống đã khởi động!"
echo "👉 Frontend (UI): http://localhost:5173"
echo "👉 Backend (API): http://localhost:8000"
echo "Ấn Ctrl+C để dừng cả 2 hệ thống."
echo "=================================================="

# Hàm xử lý khi nhấn Ctrl+C
cleanup() {
    echo ""
    echo "Đang tắt các dịch vụ..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Chờ tiến trình
wait
