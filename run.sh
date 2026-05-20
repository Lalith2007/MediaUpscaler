#!/usr/bin/env bash
# ============================================================
# MediaUpscaler — Unified Runner (starts both servers)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

echo ""
echo "🎬 MediaUpscaler — Starting Servers"
echo "===================================="

# Cleanup function to kill both servers on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers…"
    [ -n "$AI_PID"  ] && kill "$AI_PID"  2>/dev/null && echo "   → AI server (PID $AI_PID) stopped."
    [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null && echo "   → Web server (PID $WEB_PID) stopped."
    wait 2>/dev/null
    echo "👋 All servers stopped. Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---- Start AI API Server (port 5001) ----
echo "🧠 Starting AI API server on http://localhost:5001 …"
python ai_api/api_server.py &
AI_PID=$!
sleep 2

# ---- Start Web App Server (port 8000) ----
echo "🌐 Starting Web App server on http://localhost:8000 …"
python app.py &
WEB_PID=$!
sleep 1

echo ""
echo "✅ Both servers are running!"
echo "   🌐 Web App:   http://localhost:8000"
echo "   🧠 AI API:    http://localhost:5001"
echo ""
echo "   Press Ctrl+C to stop both servers."
echo ""

# Wait for either process to exit
wait
