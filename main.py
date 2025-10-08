import os
import uvicorn
import nbformat
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from rag_system import RAGSystem
from config import SERVER_HOST, SERVER_PORT

# Initialize RAG system
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system
    rag_system = RAGSystem()
    await rag_system.initialize()
    yield
    # Shutdown (if needed)
    # Add any cleanup code here

app = FastAPI(title="JADE RAG System", version="1.0.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    question: str

@app.get("/")
async def root():
    return {"message": "JADE RAG System is running"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        result = await rag_system.query(request.question, max_results=request.max_results)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_notebooks():
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        result = await rag_system.ingest_notebooks()
        return {"message": f"Successfully ingested {result['count']} notebook chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_initialized": rag_system is not None}

@app.get("/notebook/{filename}")
async def get_notebook_content(filename: str):
    """Get full notebook content for display in modal"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        # Find the notebook file
        notebooks_dir = "./Clases"
        notebook_path = os.path.join(notebooks_dir, filename)
        
        if not os.path.exists(notebook_path):
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Extract full notebook content
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        cells = []
        for cell_idx, cell in enumerate(notebook.cells):
            cell_data = {
                "index": cell_idx,
                "type": cell.cell_type,
                "content": cell.source.strip()
            }
            cells.append(cell_data)
        
        return {
            "filename": filename,
            "cells": cells
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the UI
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JADE RAG Q&A</title>
        <script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js"></script>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #555;
            }
            input, textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
                box-sizing: border-box;
            }
            textarea {
                height: 100px;
                resize: vertical;
            }
            button {
                background: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                margin-right: 10px;
            }
            button:hover {
                background: #0056b3;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .loading {
                display: none;
                text-align: center;
                color: #666;
                margin: 20px 0;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }
            .sources {
                margin-top: 15px;
                padding: 15px;
                background: #e9ecef;
                border-radius: 6px;
            }
            .source {
                margin-bottom: 10px;
                padding: 10px;
                background: white;
                border-radius: 4px;
                border-left: 3px solid #28a745;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .source:hover {
                background: #f8f9fa;
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .source-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
            }
            .source-title {
                font-weight: 600;
                color: #333;
                flex: 1;
            }
            .confidence-badge {
                background: #007bff;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 10px;
            }
            .confidence-high {
                background: #28a745;
            }
            .confidence-medium {
                background: #ffc107;
                color: #333;
            }
            .confidence-low {
                background: #dc3545;
            }
            .source-content {
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }
            .error {
                color: #dc3545;
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }
            /* Markdown styling */
            .markdown-content h1, .markdown-content h2, .markdown-content h3, .markdown-content h4, .markdown-content h5, .markdown-content h6 {
                color: #333;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .markdown-content h1 { font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .markdown-content h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .markdown-content h3 { font-size: 1.3em; }
            .markdown-content code {
                background: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            .markdown-content pre {
                background: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                overflow-x: auto;
                margin: 10px 0;
            }
            .markdown-content pre code {
                background: none;
                padding: 0;
                border-radius: 0;
            }
            .markdown-content blockquote {
                border-left: 4px solid #007bff;
                margin: 10px 0;
                padding-left: 15px;
                color: #666;
            }
            .markdown-content table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            .markdown-content table th, .markdown-content table td {
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }
            .markdown-content table th {
                background: #f8f9fa;
                font-weight: 600;
            }
            .markdown-content ul, .markdown-content ol {
                margin: 10px 0;
                padding-left: 20px;
            }
            .markdown-content li {
                margin: 5px 0;
            }
            .markdown-content hr {
                border: none;
                border-top: 2px solid #eee;
                margin: 20px 0;
            }
            .markdown-content strong {
                font-weight: 600;
                color: #333;
            }
            .markdown-content em {
                font-style: italic;
            }
            /* Modal popup styling */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            .modal-content {
                background-color: white;
                margin: 2% auto;
                padding: 20px;
                border-radius: 10px;
                width: 90%;
                max-width: 1200px;
                max-height: 90vh;
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }
            .modal-title {
                font-size: 1.5em;
                font-weight: 600;
                color: #333;
            }
            .close {
                color: #aaa;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
                line-height: 1;
            }
            .close:hover {
                color: #000;
            }
            .notebook-content {
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                white-space: pre-wrap;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .highlighted-content {
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(255,193,7,0.3);
            }
            .cell-separator {
                border-top: 2px solid #007bff;
                margin: 20px 0;
                padding-top: 10px;
            }
            .cell-info {
                background: #e9ecef;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
                color: #666;
                margin-bottom: 10px;
                display: inline-block;
            }
            .loading-modal {
                text-align: center;
                padding: 40px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 JADE RAG Q&A System</h1>
            
            <div class="input-group">
                <label for="question">Ask a question about the course content:</label>
                <textarea id="question" placeholder="e.g., What are variables in Python? How do I use functions?"></textarea>
            </div>
            
            <button onclick="askQuestion()">Ask Question</button>
            <button onclick="ingestNotebooks()">Re-ingest Notebooks</button>
            
            <div class="loading" id="loading">
                <p>🤔 Thinking...</p>
            </div>
            
            <div id="result"></div>
        </div>

        <!-- Modal for notebook content -->
        <div id="notebookModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title" id="modalTitle">Notebook Content</div>
                    <span class="close" onclick="closeModal()">&times;</span>
                </div>
                <div id="modalBody">
                    <div class="loading-modal">Loading notebook content...</div>
                </div>
            </div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value.trim();
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                loading.style.display = 'block';
                result.innerHTML = '';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            max_results: 5
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    displayResult(data);
                } catch (error) {
                    result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                } finally {
                    loading.style.display = 'none';
                }
            }
            
            async function ingestNotebooks() {
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                loading.style.display = 'block';
                result.innerHTML = '';
                
                try {
                    const response = await fetch('/ingest', {
                        method: 'POST'
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    result.innerHTML = `<div class="result"><h3>✅ Success</h3><p>${data.message}</p></div>`;
                } catch (error) {
                    result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                } finally {
                    loading.style.display = 'none';
                }
            }
            
            function displayResult(data) {
                const result = document.getElementById('result');
                
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<div class="sources"><h4>📚 Sources:</h4>';
                    data.sources.forEach((source, index) => {
                        const confidence = source.confidence || 0;
                        let confidenceClass = 'confidence-low';
                        if (confidence >= 70) {
                            confidenceClass = 'confidence-high';
                        } else if (confidence >= 50) {
                            confidenceClass = 'confidence-medium';
                        }
                        
                        sourcesHtml += `
                            <div class="source" data-filename="${source.metadata.filename}" data-cell-index="${source.metadata.cell_index}" data-source-content="${encodeURIComponent(source.content)}">
                                <div class="source-header">
                                    <div class="source-title">${source.metadata.filename || 'Unknown'}</div>
                                    <span class="confidence-badge ${confidenceClass}">${confidence}%</span>
                                </div>
                                <div class="source-content">${source.content.substring(0, 200)}...</div>
                            </div>
                        `;
                    });
                    sourcesHtml += '</div>';
                }
                
                // Render markdown content
                const renderedAnswer = marked.parse(data.answer);
                
                result.innerHTML = `
                    <div class="result">
                        <h3>💡 Answer:</h3>
                        <div class="markdown-content">${renderedAnswer}</div>
                        ${sourcesHtml}
                    </div>
                `;
            }
            
            // Allow Enter key to submit
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    askQuestion();
                }
            });
            
            // Modal functions
            async function openNotebookModal(filename, cellIndex, sourceContent) {
                const modal = document.getElementById('notebookModal');
                const modalTitle = document.getElementById('modalTitle');
                const modalBody = document.getElementById('modalBody');
                
                modalTitle.textContent = filename;
                modalBody.innerHTML = '<div class="loading-modal">Loading notebook content...</div>';
                modal.style.display = 'block';
                
                try {
                    const response = await fetch(`/notebook/${filename}`);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    displayNotebookContent(data, cellIndex, sourceContent);
                } catch (error) {
                    modalBody.innerHTML = `<div class="error">Error loading notebook: ${error.message}</div>`;
                }
            }
            
            function displayNotebookContent(notebookData, targetCellIndex, sourceContent) {
                const modalBody = document.getElementById('modalBody');
                let content = '';
                
                notebookData.cells.forEach((cell, index) => {
                    const isTargetCell = index === targetCellIndex;
                    const cellClass = isTargetCell ? 'highlighted-content' : 'notebook-content';
                    const cellTypeLabel = cell.type === 'markdown' ? 'Markdown' : 'Code';
                    
                    content += `
                        <div class="cell-separator">
                            <div class="cell-info">Cell ${index} - ${cellTypeLabel}</div>
                            <div class="${cellClass}">${escapeHtml(cell.content)}</div>
                        </div>
                    `;
                });
                
                modalBody.innerHTML = content;
                
                // Scroll to the highlighted cell
                const highlightedCell = modalBody.querySelector('.highlighted-content');
                if (highlightedCell) {
                    highlightedCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
            
            function closeModal() {
                document.getElementById('notebookModal').style.display = 'none';
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('notebookModal');
                if (event.target === modal) {
                    closeModal();
                }
            }
            
            // Event delegation for source clicks
            document.addEventListener('click', function(event) {
                const sourceElement = event.target.closest('.source');
                if (sourceElement) {
                    const filename = sourceElement.getAttribute('data-filename');
                    const cellIndex = parseInt(sourceElement.getAttribute('data-cell-index'));
                    const sourceContent = decodeURIComponent(sourceElement.getAttribute('data-source-content'));
                    openNotebookModal(filename, cellIndex, sourceContent);
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
