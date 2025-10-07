# JADE_RAG

A Retrieval-Augmented Generation (RAG) system built with ChromaDB for querying Jupyter notebooks using semantic search.

## Features

- 🚀 **Docker Compose Setup**: Easy deployment with ChromaDB and Flask server
- 📓 **Jupyter Notebook Processing**: Automatically convert notebooks into embeddings
- 🔍 **Semantic Search**: Query your notebooks using natural language
- 🎨 **Web UI**: Beautiful and intuitive Q&A interface
- 💾 **Persistent Storage**: Keep your embeddings across restarts

## Architecture

- **ChromaDB**: Vector database for storing and querying embeddings
- **Flask**: Web server for the Q&A API and UI
- **Sentence Transformers**: Generate embeddings from notebook content
- **Docker Compose**: Orchestrate ChromaDB and the RAG server

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for local development)

### Running with Docker Compose

**Note:** The Docker Compose setup requires significant disk space (~16GB) due to ML dependencies. If you encounter disk space issues, use the lightweight or local setup method below.

**Full Stack (ChromaDB + Flask in containers):**
```bash
docker compose up --build
```

**Lightweight (ChromaDB only - Recommended):**
```bash
# Start only ChromaDB in Docker
docker compose -f docker-compose.light.yml up

# In another terminal, run Flask locally
pip install -r requirements.txt
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

### Running Locally (Development)

**Recommended for development and testing - requires less disk space.**

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or use the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

2. Start ChromaDB (using Docker):
```bash
docker run -d -p 8000:8000 -v ./chroma_data:/chroma/chroma chromadb/chroma:latest
```

3. Run the Flask application:
```bash
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
python app.py
```

4. Access the UI at `http://localhost:5000`

## Usage

### Adding Notebooks

1. Place your `.ipynb` files in the `notebooks/` directory
2. Click the "Process Notebooks" button in the web UI
3. The system will extract content from all cells (code and markdown)

### Querying

1. Type your question in the search box
2. Press Enter or click "Search"
3. View the most relevant notebook cells with:
   - Source notebook name
   - Cell type (code/markdown)
   - Cell index
   - Similarity score

### API Endpoints

- `GET /`: Main Q&A interface
- `POST /api/query`: Submit a question
  ```json
  {
    "question": "How do I create a DataFrame?",
    "n_results": 5
  }
  ```
- `POST /api/upload`: Process notebooks from the notebooks directory
- `GET /api/stats`: Get collection statistics

## Project Structure

```
JADE_RAG/
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Flask application container
├── requirements.txt        # Python dependencies
├── app.py                  # Flask web application
├── rag_system.py          # RAG system with ChromaDB
├── notebook_processor.py   # Jupyter notebook processor
├── templates/
│   └── index.html         # Web UI
├── notebooks/             # Place your .ipynb files here
│   └── sample_notebook.ipynb
└── chroma_data/           # ChromaDB persistent storage (auto-created)
```

## Example Queries

- "How do I create a pandas DataFrame?"
- "Show me examples of linear regression"
- "What machine learning models are used?"
- "How to calculate mean and statistics?"

## Configuration

### Environment Variables

- `CHROMA_HOST`: ChromaDB host (default: `chromadb` in Docker, `localhost` locally)
- `CHROMA_PORT`: ChromaDB port (default: `8000`)

### Customization

- **Embedding Model**: Edit `rag_system.py` to change the model (default: `all-MiniLM-L6-v2`)
- **Collection Name**: Modify `collection_name` in `rag_system.py`
- **Number of Results**: Adjust `n_results` in API calls

## Technology Stack

- **Python 3.11**
- **ChromaDB**: Vector database
- **Flask**: Web framework
- **Sentence Transformers**: Embedding generation
- **Docker & Docker Compose**: Containerization
- **nbformat**: Jupyter notebook parsing

## Troubleshooting

### ChromaDB Connection Issues

If you see connection errors:
```bash
# Check if ChromaDB is running
docker-compose ps

# View ChromaDB logs
docker-compose logs chromadb

# Restart services
docker-compose restart
```

### No Results Found

- Ensure notebooks are processed (click "Process Notebooks")
- Check that notebooks contain content in cells
- Try different query phrasings

### Port Already in Use

Change ports in `docker-compose.yml`:
```yaml
ports:
  - "5001:5000"  # Change Flask port
  - "8001:8000"  # Change ChromaDB port
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details