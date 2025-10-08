import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:20b")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8001"))

# Notebooks Configuration
NOTEBOOKS_DIR = os.getenv("NOTEBOOKS_DIR", "/app/notebooks")
