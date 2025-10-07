# JADE RAG Usage Guide

This guide provides detailed instructions for using the JADE RAG system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Adding Notebooks](#adding-notebooks)
3. [Querying the System](#querying-the-system)
4. [API Reference](#api-reference)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### Option 1: Local Setup (Recommended)

This method is faster and uses less disk space.

```bash
# 1. Clone the repository
git clone https://github.com/david-wis/JADE_RAG.git
cd JADE_RAG

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start ChromaDB
docker run -d -p 8000:8000 chromadb/chroma:latest

# 4. Run the server
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
python app.py
```

### Option 2: Docker Compose

This method runs everything in containers but requires ~16GB disk space.

```bash
# 1. Clone the repository
git clone https://github.com/david-wis/JADE_RAG.git
cd JADE_RAG

# 2. Start all services
docker compose up --build
```

## Adding Notebooks

### Step 1: Place Notebooks

Place your `.ipynb` files in the `notebooks/` directory:

```bash
cp /path/to/your/notebook.ipynb notebooks/
```

### Step 2: Process Notebooks

1. Open the web UI at `http://localhost:5000`
2. Click the **"📤 Process Notebooks"** button
3. Wait for confirmation that notebooks were processed
4. The document count will update automatically

### What Gets Indexed?

The system extracts and indexes:
- **Markdown cells**: Full text content
- **Code cells**: Complete code
- **Cell outputs**: Text outputs from code execution
- **Metadata**: Notebook name, cell type, cell index

## Querying the System

### Using the Web UI

1. Navigate to `http://localhost:5000`
2. Type your question in the search box
3. Press Enter or click **"Search"**
4. View results with:
   - Relevance ranking
   - Source notebook name
   - Cell type and index
   - Similarity score

### Example Queries

Here are some example questions you can ask:

**General Code Questions:**
- "How do I create a pandas DataFrame?"
- "Show me examples of data visualization"
- "What libraries are imported?"

**Specific Implementation Questions:**
- "How is the data preprocessed?"
- "What machine learning models are used?"
- "Show me the training loop code"

**Data Analysis Questions:**
- "What statistics are calculated?"
- "How is the data cleaned?"
- "What are the main insights?"

### Understanding Results

Each result shows:
- **Rank**: Relevance ranking (1 = most relevant)
- **Source**: Which notebook the content came from
- **Cell Type**: `code` or `markdown`
- **Cell Index**: Position in the notebook
- **Distance**: Lower = more similar (0 = exact match)

## API Reference

### POST /api/query

Query the RAG system.

**Request:**
```json
{
  "question": "How do I create a DataFrame?",
  "n_results": 5
}
```

**Response:**
```json
{
  "question": "How do I create a DataFrame?",
  "results": [
    {
      "rank": 1,
      "content": "[code] import pandas as pd...",
      "source": "data_analysis.ipynb",
      "cell_type": "code",
      "cell_index": 3,
      "distance": 0.2345
    }
  ],
  "count": 5
}
```

### POST /api/upload

Process all notebooks in the notebooks directory.

**Response:**
```json
{
  "message": "Notebooks processed successfully",
  "count": 42
}
```

### GET /api/stats

Get collection statistics.

**Response:**
```json
{
  "name": "jupyter_notebooks",
  "count": 42
}
```

## Programmatic Usage

### Using the Test Script

The repository includes a test script for batch processing:

```bash
python test_rag.py
```

This script will:
1. Connect to ChromaDB
2. Process all notebooks in `notebooks/`
3. Run sample queries
4. Display results

### Custom Python Scripts

```python
from rag_system import RAGSystem
from notebook_processor import NotebookProcessor

# Initialize RAG system
rag = RAGSystem(chroma_host='localhost', chroma_port='8000')

# Process a single notebook
documents, metadatas = NotebookProcessor.process_notebook('my_notebook.ipynb')

# Generate IDs
ids = [f"my_notebook_cell_{i}" for i in range(len(documents))]

# Add to RAG system
rag.add_documents(documents, metadatas, ids)

# Query
results = rag.query("How do I load data?", n_results=5)

# Access results
for doc, metadata, distance in zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
):
    print(f"Distance: {distance:.4f}")
    print(f"Source: {metadata['source']}")
    print(f"Content: {doc[:100]}...")
    print()
```

## Troubleshooting

### ChromaDB Connection Errors

**Symptom:** "RAG system not initialized" or connection errors

**Solution:**
```bash
# Check if ChromaDB is running
docker ps | grep chroma

# If not running, start it
docker run -d -p 8000:8000 chromadb/chroma:latest

# Check if port 8000 is in use
lsof -i :8000
```

### No Results Found

**Symptom:** Queries return no results

**Solutions:**
1. Ensure notebooks are processed (click "Process Notebooks")
2. Check that notebooks contain content
3. Try different query phrasings
4. Verify document count is > 0

### Port Already in Use

**Symptom:** "Address already in use" error

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change the port in app.py
app.run(host='0.0.0.0', port=5001)
```

### Docker Compose Build Fails

**Symptom:** Docker build runs out of space

**Solution:**
Use the local setup method instead:
```bash
# Clean up Docker
docker system prune -af

# Use local setup
pip install -r requirements.txt
docker run -d -p 8000:8000 chromadb/chroma:latest
python app.py
```

### Import Errors

**Symptom:** "ModuleNotFoundError" or import errors

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# If using a virtual environment, activate it first
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

## Advanced Configuration

### Changing the Embedding Model

Edit `rag_system.py`:

```python
# Default model (fast, good quality)
self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Higher quality (slower)
self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Faster (lower quality)
self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
```

### Adjusting Number of Results

In the web UI, results are limited to 5 by default. To change this:

1. Edit `templates/index.html`
2. Find `n_results: 5` in the `submitQuery()` function
3. Change to your desired number

### Persistent ChromaDB Storage

To keep data across ChromaDB restarts:

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/chroma_data:/chroma/chroma \
  chromadb/chroma:latest
```

## Best Practices

1. **Organize Notebooks**: Use descriptive names for your notebooks
2. **Clean Content**: Remove unnecessary output before indexing
3. **Regular Updates**: Re-process notebooks when content changes
4. **Query Optimization**: Be specific in your questions
5. **Review Results**: Check multiple results for complete answers

## Next Steps

- Explore the sample notebook in `notebooks/sample_notebook.ipynb`
- Add your own notebooks and experiment with queries
- Customize the UI in `templates/index.html`
- Extend the RAG system in `rag_system.py`

For more information, see the main [README.md](README.md).
