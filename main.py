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
from code_generator import CodeExampleGenerator
from config import SERVER_HOST, SERVER_PORT

# Initialize RAG system and code generator
rag_system = None
code_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system, code_generator
    rag_system = RAGSystem()
    await rag_system.initialize()
    
    code_generator = CodeExampleGenerator(rag_system)
    await code_generator.initialize()
    
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

class CodeGenerationRequest(BaseModel):
    requirement: str
    num_examples: Optional[int] = 3

class CodeGenerationResponse(BaseModel):
    requirement: str
    examples: List[dict]
    theory_sources: List[dict]
    num_examples: int
    has_theory_improvement: bool
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "JADE RAG System is running"}

@app.get("/config")
async def get_config():
    """Get current text splitter configuration"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    return {
        "text_splitter_strategy": rag_system.text_splitter.strategy,
        "chunk_size": rag_system.text_splitter.chunk_size,
        "chunk_overlap": rag_system.text_splitter.chunk_overlap,
        "min_chunk_size": rag_system.text_splitter.min_chunk_size,
        "langchain_enabled": rag_system.text_splitter.strategy != "cell_based",
        "reranking_enabled": rag_system.enable_reranking,
        "reranking_model": rag_system.reranking_model_name if rag_system.enable_reranking else None,
        "initial_retrieval_count": rag_system.initial_retrieval_count,
        "final_retrieval_count": rag_system.final_retrieval_count
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        result = await rag_system.query(request.question, max_results=request.max_results or 5)
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

@app.post("/generate-code", response_model=CodeGenerationResponse)
async def generate_code_examples(request: CodeGenerationRequest):
    """Generate code examples based on a rubric requirement"""
    try:
        if not code_generator:
            raise HTTPException(status_code=500, detail="Code generator not initialized")
        
        result = await code_generator.generate_examples(
            request.requirement, 
            request.num_examples or 3
        )
        
        return CodeGenerationResponse(
            requirement=result["requirement"],
            examples=result["examples"],
            theory_sources=result["theory_sources"],
            num_examples=result["num_examples"],
            has_theory_improvement=result["has_theory_improvement"],
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "rag_initialized": rag_system is not None,
        "code_generator_initialized": code_generator is not None
    }

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

# Serve the Q&A UI
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    with open("templates/qa_ui.html", "r", encoding="utf-8") as f:
        return f.read()

# Serve the Code Generation UI
@app.get("/code-gen", response_class=HTMLResponse)
async def get_code_gen_ui():
    with open("templates/code_gen_ui.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
