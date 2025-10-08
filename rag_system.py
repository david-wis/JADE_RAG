import os
import asyncio
import json
from typing import List, Dict, Any, Optional
import weaviate
import ollama
import nbformat
from sentence_transformers import SentenceTransformer
import logging
from config import WEAVIATE_URL, OLLAMA_HOST, OLLAMA_PORT, MODEL_NAME, NOTEBOOKS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.weaviate_url = WEAVIATE_URL
        self.ollama_host = OLLAMA_HOST
        self.ollama_port = OLLAMA_PORT
        self.model_name = MODEL_NAME
        
        self.client = None
        self.embedding_model = None
        self.ollama_client = None
        
    async def initialize(self):
        """Initialize the RAG system components"""
        try:
            # Initialize Weaviate client
            self.client = weaviate.Client(url=self.weaviate_url)
            
            # Check if Weaviate is running
            if not self.client.is_ready():
                raise Exception("Weaviate is not ready. Please start Weaviate first.")
            
            # Create or get collection
            self._create_collection()
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Ollama client
            self.ollama_client = ollama.AsyncClient(
                host=f"http://{self.ollama_host}:{self.ollama_port}"
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _create_collection(self):
        """Create or get the JADE notebooks collection in Weaviate"""
        collection_name = "JadeNotebooks"
        
        # Check if collection exists
        if self.client.schema.exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return
        
        # Create collection schema
        collection_schema = {
            "class": collection_name,
            "description": "JADE course notebook content",
            "vectorizer": "none",  # We'll provide our own embeddings
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content of the notebook cell"
                },
                {
                    "name": "filename",
                    "dataType": ["string"],
                    "description": "The name of the notebook file"
                },
                {
                    "name": "cell_type",
                    "dataType": ["string"],
                    "description": "The type of cell (markdown, code)"
                },
                {
                    "name": "cell_index",
                    "dataType": ["int"],
                    "description": "The index of the cell in the notebook"
                },
                {
                    "name": "notebook_path",
                    "dataType": ["string"],
                    "description": "The full path to the notebook file"
                }
            ]
        }
        
        self.client.schema.create_class(collection_schema)
        logger.info(f"Created collection {collection_name}")
    
    def extract_notebook_content(self, notebook_path: str) -> List[Dict[str, Any]]:
        """Extract content from a Jupyter notebook"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            chunks = []
            filename = os.path.basename(notebook_path)
            
            for cell_idx, cell in enumerate(notebook.cells):
                if cell.cell_type == 'markdown':
                    content = cell.source.strip()
                    if content:
                        chunks.append({
                            'content': content,
                            'metadata': {
                                'filename': filename,
                                'cell_type': 'markdown',
                                'cell_index': cell_idx,
                                'notebook_path': notebook_path
                            }
                        })
                elif cell.cell_type == 'code':
                    # Extract code and comments
                    code_content = cell.source.strip()
                    if code_content:
                        chunks.append({
                            'content': f"Code:\n{code_content}",
                            'metadata': {
                                'filename': filename,
                                'cell_type': 'code',
                                'cell_index': cell_idx,
                                'notebook_path': notebook_path
                            }
                        })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing notebook {notebook_path}: {e}")
            return []
    
    async def ingest_notebooks(self) -> Dict[str, Any]:
        """Ingest all notebooks from the Clases directory"""
        try:
            notebooks_dir = NOTEBOOKS_DIR
            if not os.path.exists(notebooks_dir):
                notebooks_dir = "./Clases"
            
            if not os.path.exists(notebooks_dir):
                raise FileNotFoundError(f"Notebooks directory not found: {notebooks_dir}")
            
            all_chunks = []
            notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]
            
            logger.info(f"Found {len(notebook_files)} notebook files")
            
            for notebook_file in notebook_files:
                notebook_path = os.path.join(notebooks_dir, notebook_file)
                chunks = self.extract_notebook_content(notebook_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {notebook_file}: {len(chunks)} chunks")
            
            if not all_chunks:
                logger.warning("No content found in notebooks")
                return {"count": 0, "message": "No content found"}
            
            # Clear existing collection
            try:
                self.client.schema.delete_class("JadeNotebooks")
                self._create_collection()
            except:
                pass
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            documents = [chunk['content'] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to Weaviate
            logger.info("Adding to Weaviate...")
            with self.client.batch as batch:
                batch.batch_size = 100
                for i, chunk in enumerate(all_chunks):
                    properties = {
                        "content": chunk['content'],
                        "filename": chunk['metadata']['filename'],
                        "cell_type": chunk['metadata']['cell_type'],
                        "cell_index": chunk['metadata']['cell_index'],
                        "notebook_path": chunk['metadata']['notebook_path']
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name="JadeNotebooks",
                        vector=embeddings[i]
                    )
            
            logger.info(f"Successfully ingested {len(all_chunks)} chunks")
            return {"count": len(all_chunks), "message": "Successfully ingested notebooks"}
            
        except Exception as e:
            logger.error(f"Error ingesting notebooks: {e}")
            raise
    
    async def query(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if not self.client:
                raise Exception("Weaviate client not initialized")
            
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode([question]).tolist()[0]
            
            # Search for relevant documents in Weaviate
            results = (
                self.client.query
                .get("JadeNotebooks", ["content", "filename", "cell_type", "cell_index", "notebook_path"])
                .with_near_vector({"vector": question_embedding})
                .with_limit(max_results)
                .with_additional(["certainty", "distance"])
                .do()
            )
            
            if not results.get('data', {}).get('Get', {}).get('JadeNotebooks'):
                return {
                    "answer": "I couldn't find relevant information in the course materials.",
                    "sources": [],
                    "question": question
                }
            
            # Prepare context from retrieved documents
            context_docs = []
            for item in results['data']['Get']['JadeNotebooks']:
                # Calculate confidence score from certainty (0-1 scale)
                certainty = item.get('_additional', {}).get('certainty', 0)
                distance = item.get('_additional', {}).get('distance', 1)
                
                # Convert certainty to percentage and round to 1 decimal place
                confidence = round(certainty * 100, 1) if certainty else 0
                
                context_docs.append({
                    'content': item['content'],
                    'confidence': confidence,
                    'certainty': certainty,
                    'distance': distance,
                    'metadata': {
                        'filename': item['filename'],
                        'cell_type': item['cell_type'],
                        'cell_index': item['cell_index'],
                        'notebook_path': item['notebook_path']
                    }
                })
            
            # Create context for the LLM
            context = "\n\n".join([doc['content'] for doc in context_docs])
            
            # Generate prompt
            prompt = f"""Based on the following course materials, please answer the question: {question}

Course Materials:
{context}

Please provide a clear, helpful answer based on the course content. If the information is not sufficient, please say so."""

            # Query Ollama
            response = await self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            answer = response['response'].strip()
            
            return {
                "answer": answer,
                "sources": context_docs,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "question": question
            }
