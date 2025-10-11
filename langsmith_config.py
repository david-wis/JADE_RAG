"""
LangSmith configuration utilities for JADE_RAG
"""

import os
import yaml
from dotenv import load_dotenv, dotenv_values

def load_langsmith_config():
    """Load LangSmith config from langsmith_config.yaml and .env"""
    config_path = os.path.join(os.path.dirname(__file__), "langsmith_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading langsmith config: {e}")
        config = {}
    
    load_dotenv(override=True)
    DOTENV = dotenv_values()
    langsmith = config.get("langsmith", {})
    
    if langsmith.get("enable"):
        print("🔧 LangSmith integration enabled")
        os.environ["LANGSMITH_TRACING"] = langsmith.get("tracing", "true")
        if langsmith.get("endpoint"):
            os.environ["LANGSMITH_ENDPOINT"] = langsmith["endpoint"]
        if DOTENV.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = DOTENV["LANGSMITH_API_KEY"]
        if DOTENV.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = DOTENV["LANGSMITH_PROJECT"]
        
        print(f"📁 Project: {os.environ.get('LANGSMITH_PROJECT', 'Not set')}")
        print(f"🔗 Endpoint: {os.environ.get('LANGSMITH_ENDPOINT', 'Not set')}")
        print(f"🔑 API Key: {'*' * 20}{DOTENV.get('LANGSMITH_API_KEY', '')[-4:] if DOTENV.get('LANGSMITH_API_KEY') else 'Not set'}")
    else:
        print("⚠️ LangSmith integration disabled")
