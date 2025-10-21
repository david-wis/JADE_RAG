import os
import asyncio
import json
import re
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
    PYTHON_NOTEBOOKS_DIR,
    HASKELL_NOTEBOOKS_DIR,
    AI_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    TEXT_SPLITTER_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    DEBUG_MODE,
    ENABLE_RERANKING,
    RERANKING_MODEL,
    INITIAL_RETRIEVAL_COUNT,
    FINAL_RETRIEVAL_COUNT,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
    LANGSMITH_TRACING,
    ENABLE_LANGSMITH_TRACING,
    TEMPERATURE_EXAMPLE_GENERATION,
    TEMPERATURE_THEORY_CORRECTION,
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


def extract_class_number(filename: str) -> int:
    """
    Extract the class number from notebook filename.
    
    Examples:
    - "00 - Variables.ipynb" -> 0
    - "01 - Operadores aritméticos.ipynb" -> 1
    - "03_Funciones_utiles_y_Errores.ipynb" -> 3
    - "04.2-ciclos-while.ipynb" -> 4
    - "09.1-for.ipynb" -> 9
    - "09.2-intro-listas.ipynb" -> 9
    
    Args:
        filename: The notebook filename
        
    Returns:
        The class number as an integer, or 999 if no number is found
    """
    # Remove .ipynb extension
    name_without_ext = filename.replace('.ipynb', '')
    
    # Try to match number at the beginning of the filename
    # This handles patterns like "00", "01", "03", "04.2", "09.1", etc.
    match = re.match(r'^(\d+)', name_without_ext)
    
    if match:
        return int(match.group(1))
    else:
        # If no number is found, assign a high number to put it at the end
        logger.warning(f"No class number found in filename: {filename}, assigning 999")
        return 999


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
                    # "\n# ",        # Leads to ambiguity with python comments
                    "\n## ",       
                    "\n### ",      
                    "```python",   # Python code blocks
                    "```haskell",  # Haskell code blocks
                    "```",         # Generic code block endings
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
            metadata = {"filename": "test", "notebook_path": "test"}
        
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

    def _debug_chunk_analysis(self, documents: List[str]) -> None:
        """Display chunk analysis and create histogram visualization in debug mode"""
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
            debug=DEBUG_MODE  # Controlled by environment variable
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
        
        # Temperature configurations
        self.temperature_example_generation = TEMPERATURE_EXAMPLE_GENERATION
        self.temperature_theory_correction = TEMPERATURE_THEORY_CORRECTION

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
    
    def _create_llm_with_temperature(self, temperature: float):
        """Create an LLM instance with a specific temperature"""
        if self.ai_provider == "openai":
            if not self.openai_api_key:
                raise Exception("OpenAI API key is required when using OpenAI provider")
            
            return ChatOpenAI(
                model=self.openai_model,
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                temperature=temperature
            )
        else:
            # Initialize Ollama LLM
            return OllamaLLM(
                model=self.model_name,
                temperature=temperature,
                base_url=f"http://{self.ollama_host}:{self.ollama_port}"
            )
    
    def get_llm_for_example_generation(self):
        """Get LLM instance configured for creative example generation (higher temperature)"""
        return self._create_llm_with_temperature(self.temperature_example_generation)
    
    def get_llm_for_theory_correction(self):
        """Get LLM instance configured for precise theory-based correction (lower temperature)"""
        return self._create_llm_with_temperature(self.temperature_theory_correction)
    
    def _create_collection(self, dataset: str = "python"):
        """Create or get the notebooks collection in Weaviate for a specific dataset"""
        collection_name = f"JadeNotebooks_{dataset.title()}"
        
        # Check if collection exists
        if self.client.schema.exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return

        # Create collection schema
        collection_schema = {
            "class": collection_name,
            "description": f"JADE {dataset} course notebook content",
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
                    "name": "notebook_path",
                    "dataType": ["string"],
                    "description": "The full path to the notebook file",
                },
                {
                    "name": "class_number",
                    "dataType": ["int"],
                    "description": "The class number extracted from the notebook filename",
                },
                {
                    "name": "dataset",
                    "dataType": ["string"],
                    "description": "The dataset type (python or haskell)",
                },
            ],
        }

        self.client.schema.create_class(collection_schema)
        logger.info(f"Created collection {collection_name}")

    def extract_notebook_content(self, notebook_path: str, dataset: str = "python") -> List[Dict[str, Any]]:
        """Extract content from a Jupyter notebook as a single unit and split it consistently"""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Combine all content into a single text with proper formatting
            combined_content = []
            filename = os.path.basename(notebook_path)

            for cell in notebook.cells:
                cell_text = cell.source.strip()
                if cell_text:
                    # Add language-specific code block markers for code cells
                    if cell.cell_type == "code":
                        # Determine the language based on dataset
                        language = "python" if dataset == "python" else "haskell"
                        formatted_cell = f"```{language}\n{cell_text}\n```"
                        combined_content.append(formatted_cell)
                    else:
                        # For markdown cells, just add the text
                        combined_content.append(cell_text)

            if not combined_content:
                return []

            # Join all content with double newlines
            full_content = "\n\n".join(combined_content)

            # Create metadata for the combined content
            class_number = extract_class_number(filename)
            metadata = {
                "filename": filename,
                "notebook_path": notebook_path,
                "total_cells": len(notebook.cells),
                "class_number": class_number,
                "dataset": dataset
            }
            
            # Use text splitter to split the combined content
            chunks = self.text_splitter.split_text(full_content, metadata)

            # Debug: Write chunks to file if debug mode is enabled
            if self.text_splitter.debug:
                debug_filename = f"debug_chunks_{dataset}_{filename.replace('.ipynb', '')}.txt"
                debug_path = os.path.join("debug_output", debug_filename)
                os.makedirs("debug_output", exist_ok=True)
                
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== DEBUG: Chunks for {filename} ({dataset} dataset) ===\n")
                    f.write(f"Strategy: {self.text_splitter.strategy}\n")
                    f.write(f"Total chunks: {len(chunks)}\n")
                    f.write(f"Chunk size limit: {self.text_splitter.chunk_size}\n")
                    f.write(f"Chunk overlap: {self.text_splitter.chunk_overlap}\n")
                    f.write(f"Min chunk size: {self.text_splitter.min_chunk_size}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for i, chunk in enumerate(chunks):
                        f.write(f"--- CHUNK {i+1} ---\n")
                        f.write(f"Length: {len(chunk['content'])} characters\n")
                        f.write(f"Metadata: {chunk['metadata']}\n")
                        f.write(f"Content:\n{chunk['content']}\n")
                        f.write("-" * 80 + "\n\n")
                
                logger.info(f"Debug chunks written to: {debug_path}")

            logger.info(f"Extracted {len(chunks)} chunks from {filename} using {self.text_splitter.strategy} strategy")
            return chunks

        except Exception as e:
            logger.error(f"Error processing notebook {notebook_path}: {e}")
            return []

    @traceable(name="rerank_documents")
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]):
        """Rerank documents using CrossEncoder, returns before/after comparison for LangSmith trace"""
        if not self.reranker or not documents:
            logger.warning("Reranking disabled or no documents to rerank")
            return {
                "before_rerank": [],
                "after_rerank": [],
                "reranked_docs": documents
            }

        try:
            # Save before reranking snapshot for trace
            before_rerank = [
                {
                    "content": doc["content"],
                    "confidence": doc.get("confidence"),
                    "certainty": doc.get("certainty"),
                    "distance": doc.get("distance"),
                    "metadata": doc.get("metadata", {})
                }
                for doc in documents
            ]

            # Prepare query-document pairs for CrossEncoder
            pairs = [[query, doc["content"]] for doc in documents]
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            # Add rerank scores to documents and sort
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(rerank_scores[i])
            # Sort by rerank score (higher is better)
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

            # Save after reranking snapshot for trace
            after_rerank = [
                {
                    "content": doc["content"],
                    "rerank_score": doc.get("rerank_score"),
                    "confidence": doc.get("confidence"),
                    "certainty": doc.get("certainty"),
                    "distance": doc.get("distance"),
                    "metadata": doc.get("metadata", {})
                }
                for doc in reranked_docs
            ]

            return {
                "before_rerank": before_rerank,
                "after_rerank": after_rerank,
            }
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return {
                "before_rerank": before_rerank if 'before_rerank' in locals() else [],
                "after_rerank": [],
            }

    @traceable(name="ingest_notebooks")
    async def ingest_notebooks(self, dataset: str = "python") -> Dict[str, Any]:
        """Ingest all notebooks from the specified dataset directory"""
        try:
            # Determine the notebooks directory based on dataset
            if dataset == "python":
                notebooks_dir = PYTHON_NOTEBOOKS_DIR
            elif dataset == "haskell":
                notebooks_dir = HASKELL_NOTEBOOKS_DIR
            else:
                raise ValueError(f"Unsupported dataset: {dataset}. Must be 'python' or 'haskell'")

            if not os.path.exists(notebooks_dir):
                raise FileNotFoundError(
                    f"Notebooks directory not found: {notebooks_dir}"
                )

            all_chunks = []
            notebook_files = [
                f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")
            ]

            logger.info(f"Found {len(notebook_files)} notebook files in {dataset} dataset")

            for notebook_file in notebook_files:
                notebook_path = os.path.join(notebooks_dir, notebook_file)
                chunks = self.extract_notebook_content(notebook_path, dataset)
                all_chunks.extend(chunks)
                logger.info(f"Processed {notebook_file}: {len(chunks)} chunks")

            if not all_chunks:
                logger.warning(f"No content found in {dataset} notebooks")
                return {"count": 0, "message": f"No content found in {dataset} dataset"}

            # Clear existing collection for this dataset
            collection_name = f"JadeNotebooks_{dataset.title()}"
            try:
                self.client.schema.delete_class(collection_name)
                self._create_collection(dataset)
            except:
                pass

            # Generate embeddings
            logger.info("Generating embeddings...")
            documents = [chunk["content"] for chunk in all_chunks]
            
            # Show chunks being processed for embeddings only in debug mode
            if self.text_splitter.debug:
                self.text_splitter._debug_chunk_analysis(documents)
            
            embeddings = self.embedding_model.encode(documents).tolist()

            # Add to Weaviate
            logger.info("Adding to Weaviate...")
            with self.client.batch as batch:
                batch.batch_size = 100
                for i, chunk in enumerate(all_chunks):
                    properties = {
                        "content": chunk["content"],
                        "filename": chunk["metadata"]["filename"],
                        "notebook_path": chunk["metadata"]["notebook_path"],
                        "class_number": chunk["metadata"]["class_number"],
                        "dataset": chunk["metadata"]["dataset"],
                    }

                    batch.add_data_object(
                        data_object=properties,
                        class_name=collection_name,
                        vector=embeddings[i],
                    )

            # Debug: Write summary of all chunks if debug mode is enabled
            if self.text_splitter.debug:
                summary_filename = f"debug_summary_{dataset}_all_chunks.txt"
                summary_path = os.path.join("debug_output", summary_filename)
                os.makedirs("debug_output", exist_ok=True)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== DEBUG: Summary for {dataset} dataset ===\n")
                    f.write(f"Total notebooks processed: {len(notebook_files)}\n")
                    f.write(f"Total chunks generated: {len(all_chunks)}\n")
                    f.write(f"Strategy: {self.text_splitter.strategy}\n")
                    f.write(f"Chunk size limit: {self.text_splitter.chunk_size}\n")
                    f.write(f"Chunk overlap: {self.text_splitter.chunk_overlap}\n")
                    f.write(f"Min chunk size: {self.text_splitter.min_chunk_size}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Group chunks by notebook
                    chunks_by_notebook = {}
                    for chunk in all_chunks:
                        filename = chunk['metadata']['filename']
                        if filename not in chunks_by_notebook:
                            chunks_by_notebook[filename] = []
                        chunks_by_notebook[filename].append(chunk)
                    
                    for filename, notebook_chunks in chunks_by_notebook.items():
                        f.write(f"--- NOTEBOOK: {filename} ---\n")
                        f.write(f"Chunks: {len(notebook_chunks)}\n")
                        for i, chunk in enumerate(notebook_chunks):
                            f.write(f"  Chunk {i+1}: {len(chunk['content'])} chars\n")
                        f.write("\n")
                
                logger.info(f"Debug summary written to: {summary_path}")

            logger.info(f"Successfully ingested {len(all_chunks)} chunks for {dataset} dataset")
            return {
                "count": len(all_chunks),
                "message": f"Successfully ingested {dataset} notebooks",
                "dataset": dataset
            }

        except Exception as e:
            logger.error(f"Error ingesting {dataset} notebooks: {e}")
            raise

    @traceable(name="retrieve_documents")
    async def retrieve_documents(self, query: str, max_results: int, dataset: str = "python", max_class_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database without LLM processing"""
        try:
            if not self.client:
                raise Exception("Weaviate client not initialized")

            collection_name = f"JadeNotebooks_{dataset.title()}"

            if self.enable_reranking:
                retrieval_count = max(max_results * 3, self.initial_retrieval_count)  # Retrieve more for reranking
                final_count = max_results
            else:
                retrieval_count = max_results
                final_count = max_results

            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Build the query with optional class number filtering
            query_builder = (
                self.client.query.get(
                    collection_name,
                    ["content", "filename", "notebook_path", "class_number", "dataset"],
                )
                .with_near_vector({"vector": query_embedding})
                .with_limit(retrieval_count)
                .with_additional(["certainty", "distance"])
            )
            
            # Add class number filter if specified
            if max_class_number is not None:
                query_builder = query_builder.with_where({
                    "path": ["class_number"],
                    "operator": "LessThanEqual",
                    "valueInt": max_class_number
                })
            
            results = query_builder.do()

            if not results.get("data", {}).get("Get", {}).get(collection_name):
                return []

            # Prepare context from retrieved documents
            context_docs = []
            i = 1
            for item in results["data"]["Get"][collection_name]:
                # Calculate confidence score from certainty (0-1 scale)
                certainty = item.get("_additional", {}).get("certainty", 0)
                distance = item.get("_additional", {}).get("distance", 1)

                # Convert certainty to percentage and round to 1 decimal place
                confidence = round(certainty * 100, 1) if certainty else 0

                logger.debug(f"First 20 characters of content {i}: {item['content'][:20]}")
                i += 1

                context_docs.append(
                    {
                        "content": item["content"],
                        "confidence": confidence,
                        "certainty": certainty,
                        "distance": distance,
                        "metadata": {
                            "filename": item["filename"],
                            "notebook_path": item["notebook_path"],
                            "class_number": item["class_number"],
                            "dataset": item.get("dataset", dataset),
                        },
                    }
                )

            # Apply reranking if enabled
            if self.enable_reranking and len(context_docs) > final_count:
                logger.info(f"Reranking {len(context_docs)} documents to get top {final_count}")
                rerank_result = self.rerank_documents(query, context_docs)
                context_docs = rerank_result["reranked_docs"][:final_count]  # Take top N after reranking
            elif len(context_docs) > final_count:
                context_docs = context_docs[:final_count]  # Take top N without reranking

            return context_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    @traceable(name="rag_query")
    async def query(self, question: str, max_results: int, dataset: str = "python", max_class_number: Optional[int] = None) -> Dict[str, Any]:
        """Query the RAG system with optional reranking"""
        try:
            if not self.client:
                raise Exception("Weaviate client not initialized")

            collection_name = f"JadeNotebooks_{dataset.title()}"

            if self.enable_reranking:
                retrieval_count = max(max_results * 3, self.initial_retrieval_count)  # Retrieve more for reranking
                final_count = max_results
            else:
                retrieval_count = max_results
                final_count = max_results

            # Generate embedding for the question
            question_embedding = self.embedding_model.encode([question]).tolist()[0]

            # Build the query with optional class number filtering
            query_builder = (
                self.client.query.get(
                    collection_name,
                    ["content", "filename", "notebook_path", "class_number", "dataset"],
                )
                .with_near_vector({"vector": question_embedding})
                .with_limit(retrieval_count)
                .with_additional(["certainty", "distance"])
            )
            
            # Add class number filter if specified
            if max_class_number is not None:
                query_builder = query_builder.with_where({
                    "path": ["class_number"],
                    "operator": "LessThanEqual",
                    "valueInt": max_class_number
                })
            
            results = query_builder.do()

            if not results.get("data", {}).get("Get", {}).get(collection_name):
                return {
                    "answer": f"No pude encontrar información relevante en los materiales del curso de {dataset}.",
                    "sources": [],
                    "question": question,
                }

            # Prepare context from retrieved documents
            context_docs = []
            for item in results["data"]["Get"][collection_name]:
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
                            "notebook_path": item["notebook_path"],
                            "class_number": item["class_number"],
                            "dataset": item.get("dataset", dataset),
                        },
                    }
                )

            # Apply reranking if enabled
            if self.enable_reranking and len(context_docs) > final_count:
                logger.info(f"Reranking {len(context_docs)} documents to get top {final_count}")
                rerank_result = self.rerank_documents(question, context_docs)
                context_docs = rerank_result["reranked_docs"][:final_count]  # Take top N after reranking
            elif len(context_docs) > final_count:
                context_docs = context_docs[:final_count]  # Take top N without reranking

            # Create context for the LLM
            context = "\n\n".join([doc["content"] for doc in context_docs])

            # Generate prompt based on dataset
            if dataset == "haskell":
                prompt = f"""Basándote en los siguientes materiales del curso de Haskell, por favor responde la pregunta: {question}

Materiales del Curso de Haskell:
{context}

Por favor proporciona una respuesta clara y útil basada en el contenido del curso de Haskell. Si la información no es suficiente, por favor indícalo."""
                system_message = "Eres un asistente útil que responde preguntas basándose en los materiales del curso de Haskell."
            else:
                prompt = f"""Basándote en los siguientes materiales del curso de Python, por favor responde la pregunta: {question}

Materiales del Curso de Python:
{context}

Por favor proporciona una respuesta clara y útil basada en el contenido del curso de Python. Si la información no es suficiente, por favor indícalo."""
                system_message = "Eres un asistente útil que responde preguntas basándose en los materiales del curso de Python."

            # Query LLM using LangChain
            messages = [
                SystemMessage(content=system_message),
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
