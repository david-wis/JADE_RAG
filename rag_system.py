import os
import asyncio
import json
from typing import List, Dict, Any, Optional
import weaviate
import nbformat
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
import logging
from config import (
    WEAVIATE_URL,
    OLLAMA_HOST,
    OLLAMA_PORT,
    MODEL_NAME,
    NOTEBOOKS_DIR,
    AI_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    TEXT_SPLITTER_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    ENABLE_RERANKING,
    RERANKING_MODEL,
    INITIAL_RETRIEVAL_COUNT,
    FINAL_RETRIEVAL_COUNT,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
    LANGSMITH_TRACING,
    ENABLE_LANGSMITH_TRACING,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangSmith for tracing
from langsmith import Client
from langsmith import traceable


class TextSplitter:
    """Configurable text splitter for Jupyter notebook content using LangChain"""
    
    def __init__(self, strategy: str = "cell_based", chunk_size: int = 1000, 
                 chunk_overlap: int = 200, min_chunk_size: int = 100):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize LangChain text splitter for non-cell-based strategies
        if strategy != "cell_based":
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],  # Python-friendly separators
                is_separator_regex=False,
            )
        
    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text based on the configured strategy"""
        if self.strategy == "cell_based":
            return self._cell_based_split(text, metadata)
        else:
            return self._langchain_split(text, metadata)
    
    def _cell_based_split(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split by cell boundaries (original behavior)"""
        return [{"content": text, "metadata": metadata}]
    
    def _langchain_split(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text using LangChain's RecursiveCharacterTextSplitter"""
        # Use LangChain to split the text
        text_chunks = self.langchain_splitter.split_text(text)
        
        # Convert to our format with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Filter out chunks that are too small
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "splitter_strategy": self.strategy
                })
                chunks.append({"content": chunk_text.strip(), "metadata": chunk_metadata})
        
        return chunks


class RAGSystem:
    def __init__(self):
        self.weaviate_url = WEAVIATE_URL
        self.ai_provider = AI_PROVIDER

        # Ollama configuration
        self.ollama_host = OLLAMA_HOST
        self.ollama_port = OLLAMA_PORT
        self.model_name = MODEL_NAME

        # OpenAI configuration
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.openai_base_url = OPENAI_BASE_URL

        # Text splitter configuration
        self.text_splitter = TextSplitter(
            strategy=TEXT_SPLITTER_STRATEGY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_size=MIN_CHUNK_SIZE
        )

        # Reranking configuration
        self.enable_reranking = ENABLE_RERANKING
        self.reranking_model_name = RERANKING_MODEL
        self.initial_retrieval_count = INITIAL_RETRIEVAL_COUNT
        self.final_retrieval_count = FINAL_RETRIEVAL_COUNT

        self.client = None
        self.embedding_model = None
        self.reranker = None
        self.llm = None
        
        # LangSmith configuration
        self.enable_langsmith_tracing = ENABLE_LANGSMITH_TRACING
        self.langsmith_client = None

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
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            # self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

            # Initialize reranker if enabled
            if self.enable_reranking:
                logger.info(f"Initializing CrossEncoder reranker: {self.reranking_model_name}")
                self.reranker = CrossEncoder(self.reranking_model_name)
                logger.info("CrossEncoder reranker initialized successfully")
            else:
                logger.info("Reranking disabled")

            # Initialize LLM based on provider
            if self.ai_provider == "openai":
                if not self.openai_api_key:
                    raise Exception(
                        "OpenAI API key is required when using OpenAI provider"
                    )

                self.llm = ChatOpenAI(
                    model=self.openai_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.7
                )
                logger.info(f"Initialized OpenAI LLM with model: {self.openai_model}")
            else:
                # Initialize Ollama LLM
                self.llm = OllamaLLM(
                    model=self.model_name,
                    temperature=0.7,
                    base_url=f"http://{self.ollama_host}:{self.ollama_port}"
                )
                logger.info(f"Initialized Ollama LLM with model: {self.model_name}")

            # LangSmith is configured via environment variables in langsmith_config.py
            if os.environ.get("LANGSMITH_TRACING") == "true":
                logger.info(f"LangSmith tracing enabled for project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
            else:
                logger.info("LangSmith tracing disabled")

            logger.info(
                f"RAG system initialized successfully with {self.ai_provider} provider"
            )

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
                    "description": "The content of the notebook cell",
                },
                {
                    "name": "filename",
                    "dataType": ["string"],
                    "description": "The name of the notebook file",
                },
                {
                    "name": "cell_type",
                    "dataType": ["string"],
                    "description": "The type of cell (markdown, code)",
                },
                {
                    "name": "cell_index",
                    "dataType": ["int"],
                    "description": "The index of the cell in the notebook",
                },
                {
                    "name": "notebook_path",
                    "dataType": ["string"],
                    "description": "The full path to the notebook file",
                },
            ],
        }

        self.client.schema.create_class(collection_schema)
        logger.info(f"Created collection {collection_name}")

    def extract_notebook_content(self, notebook_path: str) -> List[Dict[str, Any]]:
        """Extract content from a Jupyter notebook using the configured text splitter"""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            chunks = []
            filename = os.path.basename(notebook_path)

            for cell_idx, cell in enumerate(notebook.cells):
                if cell.cell_type == "markdown":
                    content = cell.source.strip()
                    if content:
                        metadata = {
                            "filename": filename,
                            "cell_type": "markdown",
                            "cell_index": cell_idx,
                            "notebook_path": notebook_path,
                        }
                        # Use text splitter to potentially split the content
                        cell_chunks = self.text_splitter.split_text(content, metadata)
                        chunks.extend(cell_chunks)
                        
                elif cell.cell_type == "code":
                    # Extract code and comments
                    code_content = cell.source.strip()
                    if code_content:
                        formatted_content = f"Code:\n{code_content}"
                        metadata = {
                            "filename": filename,
                            "cell_type": "code",
                            "cell_index": cell_idx,
                            "notebook_path": notebook_path,
                        }
                        # Use text splitter to potentially split the content
                        cell_chunks = self.text_splitter.split_text(formatted_content, metadata)
                        chunks.extend(cell_chunks)

            logger.info(f"Extracted {len(chunks)} chunks from {filename} using {self.text_splitter.strategy} strategy")
            return chunks

        except Exception as e:
            logger.error(f"Error processing notebook {notebook_path}: {e}")
            return []

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using CrossEncoder"""
        if not self.reranker or not documents:
            return documents
        
        try:
            # Prepare query-document pairs for CrossEncoder
            pairs = []
            for doc in documents:
                pairs.append([query, doc["content"]])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add rerank scores to documents and sort
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(rerank_scores[i])
            
            # Sort by rerank score (higher is better)
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents using CrossEncoder")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents

    @traceable(name="ingest_notebooks")
    async def ingest_notebooks(self) -> Dict[str, Any]:
        """Ingest all notebooks from the Clases directory"""
        try:
            notebooks_dir = NOTEBOOKS_DIR
            if not os.path.exists(notebooks_dir):
                notebooks_dir = "./Clases"

            if not os.path.exists(notebooks_dir):
                raise FileNotFoundError(
                    f"Notebooks directory not found: {notebooks_dir}"
                )

            all_chunks = []
            notebook_files = [
                f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")
            ]

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
            documents = [chunk["content"] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(documents).tolist()

            # Add to Weaviate
            logger.info("Adding to Weaviate...")
            with self.client.batch as batch:
                batch.batch_size = 100
                for i, chunk in enumerate(all_chunks):
                    properties = {
                        "content": chunk["content"],
                        "filename": chunk["metadata"]["filename"],
                        "cell_type": chunk["metadata"]["cell_type"],
                        "cell_index": chunk["metadata"]["cell_index"],
                        "notebook_path": chunk["metadata"]["notebook_path"],
                    }

                    batch.add_data_object(
                        data_object=properties,
                        class_name="JadeNotebooks",
                        vector=embeddings[i],
                    )

            logger.info(f"Successfully ingested {len(all_chunks)} chunks")
            return {
                "count": len(all_chunks),
                "message": "Successfully ingested notebooks",
            }

        except Exception as e:
            logger.error(f"Error ingesting notebooks: {e}")
            raise

    @traceable(name="rag_query")
    async def query(self, question: str, max_results: int = None) -> Dict[str, Any]:
        """Query the RAG system with optional reranking"""
        try:
            if not self.client:
                raise Exception("Weaviate client not initialized")

            # Determine retrieval count based on reranking settings
            if max_results is None:
                if self.enable_reranking:
                    retrieval_count = self.initial_retrieval_count
                    final_count = self.final_retrieval_count
                else:
                    retrieval_count = 5  # Default
                    final_count = 5
            else:
                if self.enable_reranking:
                    retrieval_count = max(max_results * 3, self.initial_retrieval_count)  # Retrieve more for reranking
                    final_count = max_results
                else:
                    retrieval_count = max_results
                    final_count = max_results

            # Generate embedding for the question
            question_embedding = self.embedding_model.encode([question]).tolist()[0]

            # Search for relevant documents in Weaviate
            results = (
                self.client.query.get(
                    "JadeNotebooks",
                    ["content", "filename", "cell_type", "cell_index", "notebook_path"],
                )
                .with_near_vector({"vector": question_embedding})
                .with_limit(retrieval_count)
                .with_additional(["certainty", "distance"])
                .do()
            )

            if not results.get("data", {}).get("Get", {}).get("JadeNotebooks"):
                return {
                    "answer": "No pude encontrar información relevante en los materiales del curso.",
                    "sources": [],
                    "question": question,
                }

            # Prepare context from retrieved documents
            context_docs = []
            for item in results["data"]["Get"]["JadeNotebooks"]:
                # Calculate confidence score from certainty (0-1 scale)
                certainty = item.get("_additional", {}).get("certainty", 0)
                distance = item.get("_additional", {}).get("distance", 1)

                # Convert certainty to percentage and round to 1 decimal place
                confidence = round(certainty * 100, 1) if certainty else 0

                context_docs.append(
                    {
                        "content": item["content"],
                        "confidence": confidence,
                        "certainty": certainty,
                        "distance": distance,
                        "metadata": {
                            "filename": item["filename"],
                            "cell_type": item["cell_type"],
                            "cell_index": item["cell_index"],
                            "notebook_path": item["notebook_path"],
                        },
                    }
                )

            # Apply reranking if enabled
            if self.enable_reranking and len(context_docs) > final_count:
                logger.info(f"Reranking {len(context_docs)} documents to get top {final_count}")
                context_docs = self.rerank_documents(question, context_docs)
                context_docs = context_docs[:final_count]  # Take top N after reranking
            elif len(context_docs) > final_count:
                context_docs = context_docs[:final_count]  # Take top N without reranking

            # Create context for the LLM
            context = "\n\n".join([doc["content"] for doc in context_docs])

            # Generate prompt
            prompt = f"""Basándote en los siguientes materiales del curso, por favor responde la pregunta: {question}

Materiales del Curso:
{context}

Por favor proporciona una respuesta clara y útil basada en el contenido del curso. Si la información no es suficiente, por favor indícalo."""

            # Query LLM using LangChain
            messages = [
                SystemMessage(content="Eres un asistente útil que responde preguntas basándose en los materiales del curso."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle different response types (OpenAI has .content, Ollama returns string directly)
            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()

            return {"answer": answer, "sources": context_docs, "question": question}

        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "answer": f"Encontré un error mientras procesaba tu pregunta: {str(e)}",
                "sources": [],
                "question": question,
            }
