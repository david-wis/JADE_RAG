import os
import json
import nbformat
from typing import List, Dict, Tuple

class NotebookProcessor:
    """Process Jupyter notebooks and extract content for embeddings."""
    
    @staticmethod
    def read_notebook(notebook_path: str) -> nbformat.NotebookNode:
        """Read a Jupyter notebook file."""
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        return notebook
    
    @staticmethod
    def extract_cells(notebook: nbformat.NotebookNode) -> List[Dict[str, str]]:
        """Extract cells from notebook."""
        cells = []
        for idx, cell in enumerate(notebook.cells):
            cell_data = {
                'cell_type': cell.cell_type,
                'content': cell.source,
                'index': idx
            }
            
            # Add output for code cells if available
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                outputs = []
                for output in cell.outputs:
                    if hasattr(output, 'text'):
                        outputs.append(output.text)
                    elif hasattr(output, 'data'):
                        if 'text/plain' in output.data:
                            outputs.append(output.data['text/plain'])
                
                if outputs:
                    cell_data['output'] = '\n'.join(outputs)
            
            cells.append(cell_data)
        
        return cells
    
    @staticmethod
    def process_notebook(notebook_path: str) -> Tuple[List[str], List[Dict]]:
        """
        Process a notebook and return documents and metadata for embedding.
        
        Returns:
            Tuple of (documents, metadatas)
        """
        notebook = NotebookProcessor.read_notebook(notebook_path)
        cells = NotebookProcessor.extract_cells(notebook)
        
        documents = []
        metadatas = []
        
        notebook_name = os.path.basename(notebook_path)
        
        for cell in cells:
            # Create document text
            content = cell['content']
            if content.strip():  # Only add non-empty cells
                doc_text = f"[{cell['cell_type']}] {content}"
                
                # Add output if available
                if 'output' in cell and cell['output'].strip():
                    doc_text += f"\n[Output] {cell['output']}"
                
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    'source': notebook_name,
                    'cell_type': cell['cell_type'],
                    'cell_index': cell['index']
                }
                metadatas.append(metadata)
        
        return documents, metadatas
    
    @staticmethod
    def process_directory(directory_path: str) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Process all notebooks in a directory.
        
        Returns:
            Tuple of (documents, metadatas, ids)
        """
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Find all .ipynb files
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.ipynb'):
                    notebook_path = os.path.join(root, file)
                    print(f"Processing: {notebook_path}")
                    
                    try:
                        documents, metadatas = NotebookProcessor.process_notebook(notebook_path)
                        
                        # Generate unique IDs
                        base_id = file.replace('.ipynb', '').replace(' ', '_')
                        ids = [f"{base_id}_cell_{meta['cell_index']}" for meta in metadatas]
                        
                        all_documents.extend(documents)
                        all_metadatas.extend(metadatas)
                        all_ids.extend(ids)
                        
                        print(f"  Extracted {len(documents)} cells")
                    except Exception as e:
                        print(f"  Error processing {notebook_path}: {e}")
        
        return all_documents, all_metadatas, all_ids
