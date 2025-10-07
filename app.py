from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from rag_system import RAGSystem
from notebook_processor import NotebookProcessor

app = Flask(__name__)
CORS(app)

# Initialize RAG system
chroma_host = os.getenv('CHROMA_HOST', 'localhost')
chroma_port = os.getenv('CHROMA_PORT', '8000')

try:
    rag = RAGSystem(chroma_host=chroma_host, chroma_port=chroma_port)
    print(f"Connected to ChromaDB at {chroma_host}:{chroma_port}")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    rag = None

@app.route('/')
def index():
    """Render the main Q&A interface."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle Q&A queries."""
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    data = request.json
    question = data.get('question', '')
    n_results = data.get('n_results', 5)
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        results = rag.query(question, n_results=n_results)
        
        # Format results
        formatted_results = []
        if results and results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'rank': i + 1,
                    'content': doc,
                    'source': metadata.get('source', 'unknown'),
                    'cell_type': metadata.get('cell_type', 'unknown'),
                    'cell_index': metadata.get('cell_index', -1),
                    'distance': distance
                })
        
        return jsonify({
            'question': question,
            'results': formatted_results,
            'count': len(formatted_results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get collection statistics."""
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        stats = rag.get_collection_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_notebooks():
    """Process notebooks from the notebooks directory."""
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    notebooks_dir = '/app/notebooks'
    
    if not os.path.exists(notebooks_dir):
        return jsonify({'error': 'Notebooks directory not found'}), 404
    
    try:
        documents, metadatas, ids = NotebookProcessor.process_directory(notebooks_dir)
        
        if not documents:
            return jsonify({'message': 'No notebooks found or no content extracted', 'count': 0})
        
        rag.add_documents(documents, metadatas, ids)
        
        return jsonify({
            'message': 'Notebooks processed successfully',
            'count': len(documents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
