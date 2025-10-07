#!/bin/bash

# JADE RAG Setup Script
echo "================================================"
echo "JADE RAG System - Local Setup"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

echo "✓ Docker found"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To run the system:"
echo "1. Start ChromaDB:"
echo "   docker run -d -p 8000:8000 -v ./chroma_data:/chroma/chroma chromadb/chroma:latest"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the server:"
echo "   export CHROMA_HOST=localhost"
echo "   export CHROMA_PORT=8000"
echo "   python app.py"
echo ""
echo "4. Open your browser to http://localhost:5000"
echo ""
echo "================================================"
