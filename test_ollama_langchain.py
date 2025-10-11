#!/usr/bin/env python3
"""
Script para probar Ollama con LangChain
"""

from langsmith_config import load_langsmith_config
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

# Cargar configuración de LangSmith
load_langsmith_config()

@traceable(name="test_ollama_langchain")
def test_ollama():
    """Función de prueba con Ollama LLM"""
    print("🧪 Probando Ollama con LangChain...")
    
    # Crear un LLM de Ollama
    llm = OllamaLLM(
        model="gpt-oss:20b", 
        temperature=0.7,
        base_url="http://localhost:11434"
    )
    
    # Crear mensajes
    messages = [
        SystemMessage(content="Eres un asistente útil."),
        HumanMessage(content="Di '¡Hola desde JADE_RAG con Ollama y LangChain!'")
    ]
    
    # Invocar el LLM
    response = llm.invoke(messages)
    
    # Handle different response types
    if hasattr(response, 'content'):
        result = response.content
    else:
        result = str(response)
    
    print(f"✅ Respuesta del LLM: {result}")
    return result

if __name__ == "__main__":
    print("🚀 Probando Ollama con LangChain...")
    test_ollama()
    print("🎉 ¡Prueba completada! Revisa tu dashboard de LangSmith.")
