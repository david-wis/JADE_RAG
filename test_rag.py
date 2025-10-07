#!/usr/bin/env python3
"""
Test script for the JADE RAG system.
This script processes notebooks and performs sample queries.
"""

import os
import time
from rag_system import RAGSystem
from notebook_processor import NotebookProcessor

def main():
    print("=" * 60)
    print("JADE RAG System - Test Script")
    print("=" * 60)
    
    # Initialize RAG system
    chroma_host = os.getenv('CHROMA_HOST', 'localhost')
    chroma_port = os.getenv('CHROMA_PORT', '8000')
    
    print(f"\n1. Connecting to ChromaDB at {chroma_host}:{chroma_port}...")
    try:
        rag = RAGSystem(chroma_host=chroma_host, chroma_port=chroma_port)
        print("   ✓ Connected successfully!")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        print("\nMake sure ChromaDB is running:")
        print("   docker run -p 8000:8000 chromadb/chroma:latest")
        return
    
    # Process notebooks
    notebooks_dir = './notebooks'
    if not os.path.exists(notebooks_dir):
        print(f"\n✗ Notebooks directory not found: {notebooks_dir}")
        return
    
    print(f"\n2. Processing notebooks from {notebooks_dir}...")
    documents, metadatas, ids = NotebookProcessor.process_directory(notebooks_dir)
    
    if not documents:
        print("   ✗ No notebooks found or no content extracted")
        return
    
    print(f"   ✓ Extracted {len(documents)} documents from notebooks")
    
    # Add to RAG system
    print("\n3. Adding documents to ChromaDB...")
    rag.add_documents(documents, metadatas, ids)
    print("   ✓ Documents added successfully!")
    
    # Get stats
    stats = rag.get_collection_stats()
    print(f"\n4. Collection stats:")
    print(f"   - Collection name: {stats['name']}")
    print(f"   - Document count: {stats['count']}")
    
    # Sample queries
    sample_queries = [
        "How do I create a DataFrame?",
        "What is linear regression?",
        "Show me data analysis examples",
    ]
    
    print("\n5. Running sample queries...")
    for i, query in enumerate(sample_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        results = rag.query(query, n_results=3)
        
        if results and results['documents']:
            print(f"   Found {len(results['documents'][0])} results:")
            for j, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0][:2],  # Show top 2
                results['metadatas'][0][:2],
                results['distances'][0][:2]
            ), 1):
                print(f"\n   Result {j}:")
                print(f"   - Source: {metadata.get('source', 'unknown')}")
                print(f"   - Cell type: {metadata.get('cell_type', 'unknown')}")
                print(f"   - Distance: {distance:.4f}")
                print(f"   - Content preview: {doc[:100]}...")
        else:
            print("   No results found")
        
        time.sleep(0.5)  # Brief pause between queries
    
    print("\n" + "=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Start the web UI: python app.py")
    print("2. Access it at: http://localhost:5000")
    print("=" * 60)

if __name__ == '__main__':
    main()
