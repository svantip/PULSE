#!/bin/bash

# Start Urgency Classification Services
# This script starts both the FastAPI backend and Gradio UI

echo "🚀 Starting Urgency Classification Services..."
echo ""

# Check if requirements are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "⚠️  Gradio not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start FastAPI server in the background
echo "📡 Starting FastAPI server on http://localhost:8000..."
python ../controller.py &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 5

# Start Gradio UI
echo "🎨 Starting Gradio UI on http://localhost:7860..."
echo ""
echo "✅ Services are starting!"
echo "   - API: http://localhost:8000"
echo "   - UI:  http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Run Gradio (this will block)
python ../app.py

# Cleanup: kill FastAPI when Gradio stops
kill $FASTAPI_PID 2>/dev/null
echo ""
echo "👋 Services stopped"
