#!/bin/bash

echo "🚀 Setting up JADE RAG System locally..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Check if Ollama is running
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
fi

# Start Weaviate
echo "🗄️ Starting Weaviate..."
docker-compose up -d weaviate

# Wait for Weaviate to be ready
echo "⏳ Waiting for Weaviate to be ready..."
sleep 10

# Check if Weaviate is ready
if curl -s http://localhost:8080/v1/meta > /dev/null 2>&1; then
    echo "✅ Weaviate is ready"
else
    echo "❌ Weaviate is not ready. Please check the logs:"
    echo "  docker-compose logs weaviate"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the RAG server:"
echo "  source venv/bin/activate"
echo "  python run_local.py"
echo ""
echo "To access the UI:"
echo "  http://localhost:8001/ui"
echo ""
echo "To ingest notebooks:"
echo "  curl -X POST http://localhost:8001/ingest"

