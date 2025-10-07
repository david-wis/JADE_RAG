import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, chroma_host="chromadb", chroma_port="8000"):
        """Initialize RAG system with ChromaDB connection."""
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        self.collection_name = "jupyter_notebooks"
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Jupyter notebook embeddings"}
            )
    
    def add_documents(self, documents, metadatas=None, ids=None):
        """Add documents to the ChromaDB collection."""
        if metadatas is None:
            metadatas = [{"source": "notebook"} for _ in documents]
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text, n_results=5):
        """Query the RAG system."""
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def get_collection_stats(self):
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count()
        }
