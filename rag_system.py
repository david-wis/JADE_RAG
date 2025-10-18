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
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangSmith for tracing
from langsmith import Client
from langsmith import traceable


class TextSplitter:
    """Configurable text splitter for Jupyter notebook content using LangChain"""
    
    def __init__(self, strategy: str = "cell_based", chunk_size: int = 1000, 
                 chunk_overlap: int = 200, min_chunk_size: int = 100, debug: bool = False):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.debug = debug
        
        # Initialize LangChain text splitter for non-cell-based strategies
        if strategy != "cell_based":
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=[
                    "\n# ",        
                    "\n## ",       
                    "\n### ",      
                    "```python",   # Check if metrics improve with this
                    "```",         # Check if metrics improve with this
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ],
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
        
        # Debug output
        if self.debug:
            print(f"\n=== DEBUG: Text Splitting for {metadata.get('filename', 'unknown')} ===")
            print(f"Original text length: {len(text)}")
            print(f"Number of chunks generated: {len(text_chunks)}")
            print(f"Chunk size limit: {self.chunk_size}")
            print(f"Chunk overlap: {self.chunk_overlap}")
            print(f"Min chunk size: {self.min_chunk_size}")
            print("\n--- Chunks ---")
            for i, chunk in enumerate(text_chunks):
                print(f"Chunk {i+1} (length: {len(chunk)}):")
                print(f"'{chunk[:100]}{'...' if len(chunk) > 100 else ''}'")
                print("-" * 50)
        
        # Convert to our format with metadata and apply stricter filtering
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Filter out chunks that are too small
            # if len(chunk_text.strip()) >= self.min_chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "splitter_strategy": self.strategy
            })
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        
        if self.debug:
            print(f"Final chunks after filtering and merging (min size {self.min_chunk_size}): {len(chunks)}")
            print("=" * 60)
        
        return chunks
    
    def visualize_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Visualize how text would be split into chunks without actually splitting"""
        if metadata is None:
            metadata = {"filename": "test", "cell_type": "test", "cell_index": 0, "notebook_path": "test"}
        
        print(f"\n=== CHUNK VISUALIZATION ===")
        print(f"Text length: {len(text)} characters")
        print(f"Strategy: {self.strategy}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Chunk overlap: {self.chunk_overlap}")
        print(f"Min chunk size: {self.min_chunk_size}")
        print("=" * 50)
        
        if self.strategy == "cell_based":
            print("Cell-based strategy: No splitting performed")
            print(f"Content: '{text[:200]}{'...' if len(text) > 200 else ''}'")
        else:
            # Use LangChain to split the text
            text_chunks = self.langchain_splitter.split_text(text)
            
            print(f"Number of chunks generated: {len(text_chunks)}")
            print("\n--- Chunk Details ---")
            
            for i, chunk in enumerate(text_chunks):
                print(f"\nChunk {i+1}:")
                print(f"  Length: {len(chunk)} characters")
                print(f"  Content preview: '{chunk[:150]}{'...' if len(chunk) > 150 else ''}'")
                print(f"  Would be included: {'Yes' if len(chunk.strip()) >= self.min_chunk_size else 'No (too small)'}")
                print("-" * 40)
        
        print("=" * 50)


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
            min_chunk_size=MIN_CHUNK_SIZE,
            debug=False  # Can be enabled for debugging
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
    
    def enable_chunk_debug(self, enabled: bool = True):
        """Enable or disable debug mode for text splitting"""
        self.text_splitter.debug = enabled
        if enabled:
            logger.info("Chunk debug mode enabled - you'll see detailed splitting information")
        else:
            logger.info("Chunk debug mode disabled")
    
    def visualize_text_splitting(self, text: str, filename: str = "test"):
        """Visualize how a text would be split into chunks"""
        metadata = {
            "filename": filename,
            "cell_type": "test",
            "cell_index": 0,
            "notebook_path": f"test/{filename}"
        }
        self.text_splitter.visualize_chunks(text, metadata)
    
    def export_chunks_to_file(self, notebook_path: str, output_file: str = None):
        """Extract and export chunks from a notebook to a file for analysis"""
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(notebook_path))[0]
            output_file = f"chunks_{base_name}.txt"
        
        chunks = self.extract_notebook_content(notebook_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"CHUNKS ANALYSIS FOR: {notebook_path}\n")
            f.write(f"Strategy: {self.text_splitter.strategy}\n")
            f.write(f"Chunk size: {self.text_splitter.chunk_size}\n")
            f.write(f"Chunk overlap: {self.text_splitter.chunk_overlap}\n")
            f.write(f"Min chunk size: {self.text_splitter.min_chunk_size}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1}:\n")
                f.write(f"Filename: {chunk['metadata']['filename']}\n")
                f.write(f"Cell type: {chunk['metadata']['cell_type']}\n")
                f.write(f"Cell index: {chunk['metadata']['cell_index']}\n")
                f.write(f"Length: {len(chunk['content'])} characters\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['content'])
                f.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"Chunks exported to: {output_file}")
        return output_file

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
        """Extract content from a Jupyter notebook as a single unit and split it consistently"""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Combine all content into a single text
            combined_content = []
            filename = os.path.basename(notebook_path)

            for cell_idx, cell in enumerate(notebook.cells):
                if cell.cell_type == "markdown":
                    content = cell.source.strip()
                    if content:
                        combined_content.append(content)
                        
                elif cell.cell_type == "code":
                    code_content = cell.source.strip()
                    if code_content:
                        combined_content.append(code_content)

            if not combined_content:
                return []

            # Join all content with double newlines
            full_content = "\n\n".join(combined_content)
            
            # Create metadata for the combined content
            metadata = {
                "filename": filename,
                "cell_type": "mixed",
                "cell_index": 0,
                "notebook_path": notebook_path,
                "total_cells": len(notebook.cells)
            }
            
            # Use text splitter to split the combined content
            chunks = self.text_splitter.split_text(full_content, metadata)

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
            
            # Show chunks being processed for embeddings
            print(f"\n=== CHUNKS PARA EMBEDDINGS ({len(documents)} total) ===")
            for i, doc in enumerate(documents):
                print(f"Chunk {i+1} (length: {len(doc)}):")
                print(f"'{doc[:150]}{'...' if len(doc) > 150 else ''}'")
                print("-" * 50)
            
            # Show histogram of chunk lengths
            chunk_lengths = [len(doc) for doc in documents]
            print(f"\n=== HISTOGRAMA DE LONGITUDES DE CHUNKS ===")
            print(f"Total chunks: {len(chunk_lengths)}")
            print(f"Longitud promedio: {sum(chunk_lengths) / len(chunk_lengths):.1f}")
            print(f"Longitud mínima: {min(chunk_lengths)}")
            print(f"Longitud máxima: {max(chunk_lengths)}")
            
            # Create histogram
            length_counts = Counter(chunk_lengths)
            max_count = max(length_counts.values())
            
            print(f"\nDistribución de longitudes:")
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                bar_length = int((count / max_count) * 50)  # Scale to 50 chars max
                bar = "█" * bar_length
                print(f"{length:4d} chars: {bar} ({count})")
            
            # Create matplotlib plot
            try:
                
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                plt.subplot(1, 2, 1)
                plt.hist(chunk_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(sum(chunk_lengths) / len(chunk_lengths), color='red', linestyle='--', 
                           label=f'Promedio: {sum(chunk_lengths) / len(chunk_lengths):.1f}')
                plt.xlabel('Longitud del Chunk (caracteres)')
                plt.ylabel('Frecuencia')
                plt.title('Distribución de Longitudes de Chunks')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Create box plot
                plt.subplot(1, 2, 2)
                plt.boxplot(chunk_lengths, vert=True, patch_artist=True, 
                           boxprops=dict(facecolor='lightgreen', alpha=0.7))
                plt.ylabel('Longitud del Chunk (caracteres)')
                plt.title('Box Plot de Longitudes de Chunks')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('chunk_lengths_analysis.png', dpi=150, bbox_inches='tight')
                print(f"\n📊 Gráfico guardado como: chunk_lengths_analysis.png")
                plt.show()
                
            except Exception as e:
                print(f"\n⚠️  Error creando gráfico: {e}")
            
            print("=" * 60)
            
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

    @traceable(name="retrieve_documents")
    async def retrieve_documents(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database without LLM processing"""
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

            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Search for relevant documents in Weaviate
            results = (
                self.client.query.get(
                    "JadeNotebooks",
                    ["content", "filename", "cell_type", "cell_index", "notebook_path"],
                )
                .with_near_vector({"vector": query_embedding})
                .with_limit(retrieval_count)
                .with_additional(["certainty", "distance"])
                .do()
            )

            if not results.get("data", {}).get("Get", {}).get("JadeNotebooks"):
                return []

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
                context_docs = self.rerank_documents(query, context_docs)
                context_docs = context_docs[:final_count]  # Take top N after reranking
            elif len(context_docs) > final_count:
                context_docs = context_docs[:final_count]  # Take top N without reranking

            return context_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

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
