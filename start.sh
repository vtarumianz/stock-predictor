#!/bin/bash

# Stock Predictor - Quick Start Script for Linux/Mac

echo "🚀 Stock Predictor - Quick Start"
echo "=================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Dependencies installed successfully"
echo ""

# Run Streamlit
echo "🎉 Starting Stock Predictor..."
echo ""
echo "💡 Tip: The app will open in your browser at http://localhost:8501"
echo "💡 To stop the app, press Ctrl + C"
echo ""

streamlit run app.py
