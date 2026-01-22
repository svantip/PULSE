#!/bin/bash

# Quick Start Script for Slack Integration
# This script helps you set up and test the Slack integration

set -e

echo "================================================"
echo "  🚀 PULSE Slack Integration Quick Start"
echo "================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your Slack credentials:"
    echo "   - SLACK_BOT_TOKEN (from Slack App OAuth & Permissions)"
    echo "   - SLACK_SIGNING_SECRET (from Slack App Basic Information)"
    echo "   - SLACK_SUPPORT_CHANNEL (channel name or ID)"
    echo ""
    read -p "Press Enter once you've updated .env..."
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
else
    pip install -r requirements.txt
fi
echo "✅ Dependencies installed"
echo ""

# Test imports
echo "🔍 Testing imports..."
python3 -c "
import sys
try:
    from slack_sdk import WebClient
    print('   ✅ slack_sdk installed')
except ImportError as e:
    print(f'   ❌ Error importing slack_sdk: {e}')
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print('   ✅ transformers installed')
except ImportError as e:
    print(f'   ❌ Error importing transformers: {e}')
    sys.exit(1)

try:
    from fastapi import FastAPI
    print('   ✅ fastapi installed')
except ImportError as e:
    print(f'   ❌ Error importing fastapi: {e}')
    sys.exit(1)

print('   ✅ All required packages imported successfully')
"
echo ""

# Check if .env is configured
echo "🔑 Checking .env configuration..."
if grep -q "xoxb-your-bot-token-here" .env; then
    echo "   ⚠️  WARNING: SLACK_BOT_TOKEN not configured"
    echo "   Please update .env with your actual Slack token"
    echo ""
fi

if grep -q "your-signing-secret-here" .env; then
    echo "   ⚠️  WARNING: SLACK_SIGNING_SECRET not configured"
    echo "   Please update .env with your actual Slack signing secret"
    echo ""
fi

# Ask user what they want to do
echo "================================================"
echo "What would you like to do?"
echo "================================================"
echo ""
echo "1. Start the API server"
echo "2. Run test suite (without Slack)"
echo "3. View setup documentation"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting API server..."
        echo ""
        echo "The server will be available at: http://localhost:8000"
        echo "Slack webhook endpoint: http://localhost:8000/slack/events"
        echo ""
        echo "📋 For local development with Slack:"
        echo "   1. Open a new terminal"
        echo "   2. Run: ngrok http 8000"
        echo "   3. Use the ngrok HTTPS URL in your Slack app settings"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo ""
        sleep 2
        python3 controller.py
        ;;
    2)
        echo ""
        echo "🧪 Running test suite..."
        echo ""
        python3 test_slack_integration.py
        ;;
    3)
        echo ""
        echo "📖 Opening setup documentation..."
        echo ""
        if command -v open &> /dev/null; then
            open SLACK_INTEGRATION.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open SLACK_INTEGRATION.md
        else
            cat SLACK_INTEGRATION.md
        fi
        ;;
    4)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac
