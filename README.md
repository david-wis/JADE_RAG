# JADE RAG System

A Retrieval-Augmented Generation (RAG) system for JADE course materials using Weaviate, OpenAI/Ollama, and FastAPI.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Weaviate only)
- Either:
  - **OpenAI API Key** (recommended for better performance)
  - **Or Ollama installed locally** (for offline usage)
- At least 8GB RAM

### 1. Configuration

First, configure your environment by copying the template file:

```bash
# Copy the environment template
cp env.template .env
```

Edit the `.env` file and configure your settings:

#### Option A: Using OpenAI (Recommended)

```bash
# Set AI provider to OpenAI
AI_PROVIDER=openai

# Add your OpenAI API key
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4

# Other settings remain the same
WEAVIATE_URL=http://localhost:8080
SERVER_HOST=0.0.0.0
SERVER_PORT=8001
```

#### Option B: Using Ollama (Local)

```bash
# Set AI provider to Ollama
AI_PROVIDER=ollama

# Configure Ollama settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
MODEL_NAME=gpt-oss:20b

# Other settings
WEAVIATE_URL=http://localhost:8080
SERVER_HOST=0.0.0.0
SERVER_PORT=8001
```

#### Text Splitter Configuration

The system supports configurable text splitting strategies for processing Jupyter notebooks:

```bash
# Text Splitter Configuration
TEXT_SPLITTER_STRATEGY=cell_based  # "cell_based" or "langchain"
CHUNK_SIZE=1500
CHUNK_OVERLAP=250
MIN_CHUNK_SIZE=100
```

### 2. Quick Setup

```bash
# Run the setup script
./setup_local.sh
```

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Weaviate
docker-compose up -d

# If using Ollama, start Ollama (if not already running)
# Skip this step if using OpenAI
ollama serve
ollama pull gpt-oss:20b
```

### 4. Start the RAG Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python run_local.py
```

### 5. Ingest Course Materials

```bash
# Ingest all notebooks from the Clases/ directory
curl -X POST http://localhost:8001/ingest
```

### 6. Access the UI

Open your browser and go to: <http://localhost:8001/ui>

## Usage

### Web Interface

1. Navigate to <http://localhost:8001/ui>
2. Ask questions about the course content
3. View answers with source references

### API Endpoints

- `GET /` - Health check
- `POST /query` - Ask a question
- `POST /ingest` - Re-ingest notebooks
- `GET /health` - System health status

### Example API Usage

```bash
# Ask a question
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are variables in Python?", "max_results": 5}'

# Check system health
curl http://localhost:8001/health
```

## Configuration

Environment variables can be set in the `.env` file:

### AI Provider Settings

- `AI_PROVIDER`: Choose between "openai" or "ollama" (default: ollama)

### OpenAI Settings (when AI_PROVIDER=openai)

- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)
- `OPENAI_BASE_URL`: OpenAI API base URL (default: <https://api.openai.com/v1>)

### Ollama Settings (when AI_PROVIDER=ollama)

- `OLLAMA_HOST`: Ollama host (default: localhost)
- `OLLAMA_PORT`: Ollama port (default: 11434)
- `MODEL_NAME`: Ollama model name (default: gpt-oss:20b)

### General Settings

- `WEAVIATE_URL`: Weaviate server URL (default: <http://localhost:8080>)
- `SERVER_HOST`: Server host (default: 0.0.0.0)
- `SERVER_PORT`: Server port (default: 8001)
- `NOTEBOOKS_DIR`: Directory containing notebooks (default: /app/notebooks)

## Development

### Local Development

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Weaviate
docker-compose up -d weaviate

# Run the server
python run_local.py
```

### Adding New Notebooks

1. Place new `.ipynb` files in the `Clases/` directory
2. Call the ingest endpoint: `curl -X POST http://localhost:8001/ingest`