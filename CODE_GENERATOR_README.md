# JADE Code Example Generator

This is a new feature added to the JADE RAG system that generates code examples based on rubric requirements and validates them using the theoretical content from the course notebooks.

## Features

- **Code Generation**: Takes a rubric requirement and generates N diverse code examples
- **RAG Integration**: Uses the existing RAG system to find relevant theoretical content from notebooks
- **Theory-Based Improvement**: Improves generated examples based on best practices from course materials
- **Web Interface**: Clean, modern web interface for easy interaction
- **Dual Navigation**: Easy switching between Q&A system and code generator

## How It Works

1. **Initial Generation**: The system generates initial code examples based on the requirement using AI
2. **Theory Retrieval**: Uses RAG to find relevant theoretical content from the course notebooks
3. **Improvement**: Improves the examples to align with course best practices and theoretical recommendations
4. **Display**: Shows both original and improved examples with explanations

## Usage

### Web Interface

1. Start the server: `python main.py`
2. Navigate to `http://localhost:8001/code-gen`
3. Enter a rubric requirement
4. Select the number of examples to generate
5. Click "Generate Examples"

### API Endpoint

```bash
curl -X POST "http://localhost:8001/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "requirement": "Create a function that takes a list of numbers and returns the sum of all even numbers",
    "num_examples": 3
  }'
```

### Testing

Run the test script to verify everything works:

```bash
python test_code_generator.py
```

## Example Output

For a requirement like "Create a function that takes a list of numbers and returns the sum of all even numbers", the system will:

1. Generate initial examples using different approaches (list comprehension, filter, loop, etc.)
2. Find relevant theory from notebooks about list operations, functions, and best practices
3. Improve examples to use recommended functions like `sum()` and `filter()` instead of manual loops
4. Show both original and improved versions with explanations

## Configuration

The code generator uses the same configuration as the main RAG system:

- **AI Provider**: OpenAI or Ollama (configurable in `config.py`)
- **Model Settings**: Uses the same model configuration as the Q&A system
- **RAG Settings**: Inherits all RAG configuration (reranking, retrieval counts, etc.)

## Architecture

- `code_generator.py`: Main code generation logic
- `main.py`: Extended with new endpoints and UI
- `rag_system.py`: Existing RAG system (unchanged)
- `config.py`: Configuration (unchanged)

## Navigation

The system now has two main interfaces:

- **Q&A System** (`/ui`): Original question-answering interface
- **Code Generator** (`/code-gen`): New code example generation interface

Both interfaces have navigation links to easily switch between them.
