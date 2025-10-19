"""
Module for initial code generation based on requirements.
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .shared import LLMFactory, XMLParser

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Handles initial code generation from requirements."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.llm = None
    
    async def initialize(self):
        """Initialize the LLM for code generation."""
        try:
            self.llm = self.llm_factory.create_llm()
            logger.info("Code generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {e}")
            raise
    
    @traceable(name="generate_initial_examples")
    async def generate_initial_examples(self, requirement: str, num_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate initial code examples based on the requirement."""
        try:
            prompt = f"""Eres un alumno de una materia de programación en Python. Dado el siguiente requerimiento, genera {num_examples} ejemplos de código diferentes que cumplan con este requerimiento.

Requerimiento: {requirement}

Para cada ejemplo, proporciona:
1. El código Python completo
2. Una breve explicación del enfoque utilizado

IMPORTANTE: 
- Usa la indentación correcta de Python (4 espacios)
- Asegúrate de que el código esté bien formateado y sea legible
- El código debe ser correcto, pero no es necesario que sea eficiente
- Debes asumir tus conocimientos son básicos (por ejemplo, no uses clases ni lambdas).
- El código debe ser lo más imperativo posible (evita usar funciones de Python que simplifiquen el código)
- El código debe ser muy simple y corto.

Formatea tu respuesta usando formato XML con la siguiente estructura:
<example>
<code>tu código python aquí</code>
<approach>explicación del enfoque</approach>
</example>

<example>
<code>otro código python aquí</code>
<approach>explicación del enfoque</approach>
</example>

Asegúrate de que los ejemplos sean diversos y demuestren diferentes formas de resolver el problema. Enfócate en código claro y legible que siga las mejores prácticas de Python."""

            # Query LLM using LangChain
            if not self.llm:
                raise Exception("LLM not initialized")
                
            messages = [
                SystemMessage(content="Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle different response types (OpenAI has .content, Ollama returns string directly)
            if hasattr(response, 'content') and getattr(response, 'content', None):
                response_text = getattr(response, 'content', '').strip()
            else:
                response_text = str(response).strip()
            
            # Parse XML response
            try:
                examples = self.xml_parser.parse_xml_response(response_text, num_examples)
                logger.info(f"Generated {len(examples)} initial examples using XML")
                return examples
                
            except Exception as e:
                logger.error(f"Failed to parse XML response: {e}")
                # Ultimate fallback
                return [{
                    "example_id": 1,
                    "description": "Generated example based on requirement",
                    "code": "# Example code\nprint('Hello, World!')",
                    "approach": "Basic implementation"
                }]
                
        except Exception as e:
            logger.error(f"Error generating initial examples: {e}")
            return []
