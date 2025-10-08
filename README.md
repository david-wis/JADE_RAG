# JADE RAG System

A Retrieval-Augmented Generation (RAG) system for JADE course materials using Weaviate, Ollama, and FastAPI.

## Features

- 📚 **Notebook Processing**: Automatically extracts content from Jupyter notebooks in the `Clases/` directory
- 🔍 **Semantic Search**: Uses Weaviate for vector storage and retrieval
- 🤖 **AI-Powered Q&A**: Integrates with Ollama using the `gpt-oss:20b` model
- 🌐 **Web UI**: Simple and intuitive interface for asking questions
- 🐍 **Local Python**: Runs locally without Docker complexity

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI UI    │    │    Weaviate     │    │  Local Ollama   │
│   (Port 8001)   │◄──►│   (Port 8080)   │    │   (Port 11434)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  RAG System     │
                    │  (Embeddings +  │
                    │   Retrieval)    │
                    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Weaviate only)
- Ollama installed locally
- At least 8GB RAM

### 1. Quick Setup

```bash
# Run the setup script
./setup_local.sh
```

### 2. Manual Setup (Alternative)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Weaviate
docker-compose up -d weaviate

# Start Ollama (if not already running)
ollama serve
ollama pull gpt-oss:20b
```

### 3. Start the RAG Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python run_local.py
```

### 4. Ingest Course Materials

```bash
# Ingest all notebooks from the Clases/ directory
curl -X POST http://localhost:8001/ingest
```

### 5. Access the UI

Open your browser and go to: http://localhost:8001/ui

## Usage

### Web Interface

1. Navigate to http://localhost:8001/ui
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

Environment variables can be set in the `config.py` file or as environment variables:

- `WEAVIATE_URL`: Weaviate server URL (default: http://localhost:8080)
- `OLLAMA_HOST`: Ollama host (default: localhost)
- `OLLAMA_PORT`: Ollama port (default: 11434)
- `MODEL_NAME`: Ollama model name (default: gpt-oss:20b)
- `SERVER_HOST`: Server host (default: 0.0.0.0)
- `SERVER_PORT`: Server port (default: 8001)

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

## Troubleshooting

### Common Issues

1. **Ollama model not found**: Make sure Ollama is running locally and the model is pulled
   ```bash
   ollama serve
   ollama pull gpt-oss:20b
   ```

2. **Weaviate connection issues**: Check if Weaviate is running
   ```bash
   docker-compose logs weaviate
   curl http://localhost:8080/v1/meta
   ```

3. **GPU not detected**: Ensure NVIDIA Docker runtime is installed
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs weaviate
```

## File Structure

```
JADE_RAG/
├── Clases/                 # Jupyter notebooks directory
├── data/                   # Persistent data storage
├── venv/                   # Python virtual environment
├── docker-compose.yml      # Weaviate Docker configuration
├── main.py                # FastAPI application
├── rag_system.py          # Core RAG logic with Weaviate
├── config.py              # Configuration settings
├── run_local.py           # Local server runner
├── setup_local.sh         # Local setup script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the JADE educational system.