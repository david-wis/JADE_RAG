#!/usr/bin/env python3
"""
Local runner for JADE RAG System
Run this script to start the RAG server locally without Docker
"""

import uvicorn
from config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    print("🚀 Starting JADE RAG System locally...")
    print(f"📡 Server will be available at: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"🌐 Web UI will be at: http://{SERVER_HOST}:{SERVER_PORT}/ui")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    uvicorn.run(
        "main:app",  # Import string instead of app object
        host=SERVER_HOST, 
        port=SERVER_PORT,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

