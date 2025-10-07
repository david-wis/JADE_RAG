# JADE RAG Quick Reference

## 🚀 Installation

### Lightweight (Recommended)
```bash
docker compose -f docker-compose.light.yml up -d
pip install -r requirements.txt
export CHROMA_HOST=localhost CHROMA_PORT=8000
python app.py
```

### Full Docker
```bash
docker compose up --build
```

### Local Only
```bash
pip install -r requirements.txt
docker run -d -p 8000:8000 chromadb/chroma:latest
export CHROMA_HOST=localhost CHROMA_PORT=8000
python app.py
```

## 📖 Core Commands

| Action | Command |
|--------|---------|
| Start ChromaDB | `docker compose -f docker-compose.light.yml up -d` |
| Start Flask | `python app.py` |
| Run Tests | `python test_rag.py` |
| Stop All | `docker compose down` |
| Clean Docker | `docker system prune -af` |

## 🌐 URLs

| Service | URL |
|---------|-----|
| Web UI | http://localhost:5000 |
| ChromaDB | http://localhost:8000 |
| API Query | POST http://localhost:5000/api/query |
| API Upload | POST http://localhost:5000/api/upload |
| API Stats | GET http://localhost:5000/api/stats |

## 📁 File Structure

```
notebooks/          → Place .ipynb files here
templates/          → Web UI templates
chroma_data/        → ChromaDB storage (auto-created)
app.py              → Flask server
rag_system.py       → RAG logic
notebook_processor.py → Notebook parser
```

## 🔧 Environment Variables

```bash
export CHROMA_HOST=localhost    # ChromaDB hostname
export CHROMA_PORT=8000        # ChromaDB port
```

## 📊 API Examples

### Query
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a DataFrame?", "n_results": 5}'
```

### Upload
```bash
curl -X POST http://localhost:5000/api/upload
```

### Stats
```bash
curl http://localhost:5000/api/stats
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Port in use | `lsof -i :5000` then `kill -9 <PID>` |
| ChromaDB not running | `docker ps` then restart if needed |
| No results | Click "Process Notebooks" button |
| Import errors | `pip install -r requirements.txt` |
| Disk space issues | Use lightweight setup |

## 💡 Example Queries

- "How do I load data from a CSV?"
- "Show me data visualization examples"
- "What preprocessing steps are used?"
- "How is the model trained?"
- "What are the main insights?"

## 🎯 Quick Workflow

1. **Add notebooks** → Copy .ipynb to `notebooks/`
2. **Start system** → Run Docker + Flask
3. **Process** → Click "Process Notebooks" in UI
4. **Query** → Ask questions in search box
5. **Review** → See ranked results with sources

## 📚 Documentation Links

- [README.md](README.md) - Full project overview
- [USAGE.md](USAGE.md) - Detailed usage guide
- [setup.sh](setup.sh) - Automated setup

## ⚙️ Configuration

### Change Embedding Model
Edit `rag_system.py`:
```python
self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Default
# or
self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Better
```

### Change Port
Edit `app.py`:
```python
app.run(host='0.0.0.0', port=5001)  # Changed from 5000
```

### Adjust Results Count
Edit `templates/index.html`:
```javascript
n_results: 10  // Changed from 5
```

## 🔄 Common Tasks

### Reset Database
```bash
docker compose down
rm -rf chroma_data/
docker compose up -d
```

### View Logs
```bash
docker compose logs chromadb      # ChromaDB logs
docker compose logs rag-server    # Flask logs
```

### Update Code
```bash
git pull
pip install -r requirements.txt --upgrade
docker compose restart
```

## 🎓 Learning Resources

1. Try the sample notebook in `notebooks/sample_notebook.ipynb`
2. Experiment with different query styles
3. Check the USAGE.md for advanced features
4. Explore the API with curl or Postman

---

**Quick Help**: For detailed information, see README.md or USAGE.md
