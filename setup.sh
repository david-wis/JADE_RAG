#!/bin/bash

echo "🚀 Setting up JADE RAG System..."

# Create data directory
mkdir -p data

# Check if Ollama is running locally
echo "🔍 Checking if Ollama is running locally..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running locally"
    
    # Check if the model exists
    if ollama list | grep -q "gpt-oss:20b"; then
        echo "✅ Model gpt-oss:20b is already available"
    else
        echo "📥 Pulling Ollama model: gpt-oss:20b"
        ollama pull gpt-oss:20b
    fi
else
    echo "❌ Ollama is not running locally"
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo "Then pull the model:"
    echo "  ollama pull gpt-oss:20b"
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "To start the system:"
echo "  docker-compose up -d"
echo ""
echo "To access the UI:"
echo "  http://localhost:8001/ui"
echo ""
echo "To ingest notebooks:"
echo "  curl -X POST http://localhost:8001/ingest"
