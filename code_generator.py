import os
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
import logging
from rag_system import RAGSystem
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from config import (
    AI_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    OLLAMA_HOST,
    OLLAMA_PORT,
    MODEL_NAME,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
    ENABLE_LANGSMITH_TRACING,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangSmith for tracing
from langsmith import traceable


class CodeExampleGenerator:
    """Generates code examples based on rubric requirements and validates them using RAG"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.ai_provider = AI_PROVIDER
        
        # OpenAI configuration
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.openai_base_url = OPENAI_BASE_URL
        
        # Ollama configuration
        self.ollama_host = OLLAMA_HOST
        self.ollama_port = OLLAMA_PORT
        self.model_name = MODEL_NAME
        
        # Initialize LLM
        self.llm = None
        
        # LangSmith configuration
        self.enable_langsmith_tracing = ENABLE_LANGSMITH_TRACING
        
    async def initialize(self):
        """Initialize the AI client based on provider"""
        try:
            if self.ai_provider == "openai":
                if not self.openai_api_key:
                    raise Exception("OpenAI API key is required when using OpenAI provider")
                
                self.llm = ChatOpenAI(
                    model=self.openai_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.8
                )
                logger.info(f"Initialized OpenAI LLM with model: {self.openai_model}")
            else:
                # Initialize Ollama LLM
                self.llm = OllamaLLM(
                    model=self.model_name,
                    temperature=0.8,
                    base_url=f"http://{self.ollama_host}:{self.ollama_port}"
                )
                logger.info(f"Initialized Ollama LLM with model: {self.model_name}")
                
            logger.info("Code example generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {e}")
            raise
    
    @traceable(name="generate_initial_examples")
    async def generate_initial_examples(self, requirement: str, num_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate initial code examples based on the requirement"""
        try:
            prompt = f"""Eres un instructor de programación en Python. Dado el siguiente requerimiento, genera {num_examples} ejemplos de código diferentes que cumplan con este requerimiento.

Requerimiento: {requirement}

Para cada ejemplo, proporciona:
1. El código Python completo
2. Una breve explicación del enfoque utilizado

IMPORTANTE: 
- Usa la indentación correcta de Python (4 espacios)
- Asegúrate de que el código esté bien formateado y sea legible

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
            messages = [
                SystemMessage(content="Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle different response types (OpenAI has .content, Ollama returns string directly)
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()
            
            # Parse XML response
            try:
                examples = self._parse_xml_response(response_text, num_examples)
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
    
    def _parse_xml_response(self, response_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Parse XML response to extract code examples"""
        try:
            examples = []
            
            # Clean the response text - remove any "json" prefix or other text
            cleaned_text = response_text.strip()
            if cleaned_text.lower().startswith('json'):
                # Find the first '<' or '[' after "json"
                for i, char in enumerate(cleaned_text):
                    if char in '<[':
                        cleaned_text = cleaned_text[i:]
                        break
            
            # Try to find XML-like structure
            # Look for patterns like <example>, <code>, <approach>, etc.
            example_pattern = r'<example[^>]*>(.*?)</example>'
            code_pattern = r'<code[^>]*>(.*?)</code>'
            approach_pattern = r'<approach[^>]*>(.*?)</approach>'
            description_pattern = r'<description[^>]*>(.*?)</description>'
            improvements_pattern = r'<improvements[^>]*>(.*?)</improvements>'
            theory_alignment_pattern = r'<theory_alignment[^>]*>(.*?)</theory_alignment>'
            
            # Find all example blocks
            example_matches = re.findall(example_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            
            if example_matches:
                for i, example_content in enumerate(example_matches[:num_examples]):
                    # Extract code
                    code_match = re.search(code_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    code = code_match.group(1).strip() if code_match else "# No code found"
                    
                    # Extract approach
                    approach_match = re.search(approach_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    approach = approach_match.group(1).strip() if approach_match else "No approach provided"
                    
                    # Extract description
                    desc_match = re.search(description_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    description = desc_match.group(1).strip() if desc_match else f"Example {i+1}"
                    
                    # Extract improvements (for improved examples)
                    improvements_match = re.search(improvements_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    improvements = improvements_match.group(1).strip() if improvements_match else None
                    
                    # Extract theory alignment (for improved examples)
                    theory_match = re.search(theory_alignment_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    theory_alignment = theory_match.group(1).strip() if theory_match else None
                    
                    example_dict = {
                        "example_id": i + 1,
                        "description": description,
                        "code": code,
                        "approach": approach
                    }
                    
                    # Add optional fields if they exist
                    if improvements:
                        example_dict["improvements"] = improvements.split('\n') if '\n' in improvements else [improvements]
                    if theory_alignment:
                        example_dict["theory_alignment"] = theory_alignment
                    
                    examples.append(example_dict)
            else:
                # If no XML structure found, try to extract code blocks from the text
                code_blocks = re.findall(r'```python\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                if not code_blocks:
                    code_blocks = re.findall(r'```\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                
                for i, code in enumerate(code_blocks[:num_examples]):
                    examples.append({
                        "example_id": i + 1,
                        "description": f"Example {i+1}",
                        "code": code.strip(),
                        "approach": "Extracted from response"
                    })
            
            # If still no examples, create a basic one
            if not examples:
                examples.append({
                    "example_id": 1,
                    "description": "Generated example based on requirement",
                    "code": "# Example code\nprint('Hello, World!')",
                    "approach": "Basic implementation"
                })
            
            logger.info(f"Parsed {len(examples)} examples using XML fallback")
            return examples
            
        except Exception as e:
            logger.error(f"Error in XML fallback parsing: {e}")
            # Ultimate fallback
            return [{
                "example_id": 1,
                "description": "Generated example based on requirement",
                "code": "# Example code\nprint('Hello, World!')",
                "approach": "Basic implementation"
            }]
    
    async def get_relevant_theory(self, requirement: str, examples: List[Dict[str, Any]] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant theoretical content from notebooks using RAG"""
        try:
            # Create a comprehensive query that includes both requirement and examples
            query_parts = [requirement]
            
            if examples:
                # Add code snippets from examples to the query
                for example in examples:
                    if "code" in example and example["code"]:
                        # Extract key concepts from code (remove comments and common keywords)
                        code_snippet = example["code"][:500]  # Limit length
                        query_parts.append(f"código ejemplo: {code_snippet}")
            
            # Combine all query parts
            combined_query = " ".join(query_parts)
            
            # Retrieve relevant documents from vector database (no LLM processing)
            theory_sources = await self.rag_system.retrieve_documents(combined_query, max_results=max_results)
            
            return theory_sources
                
        except Exception as e:
            logger.error(f"Error getting relevant theory: {e}")
            return []
    
    @traceable(name="improve_examples_with_theory")
    async def improve_examples_with_theory(self, examples: List[Dict[str, Any]], 
                                         requirement: str, 
                                         theory_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improve examples based on theoretical best practices from notebooks"""
        try:
            if not theory_sources:
                logger.warning("No theory sources available for improvement")
                return examples
            
            # Create context from theory sources with numbered separators
            theory_context_parts = []
            for i, source in enumerate(theory_sources, 1):
                source_content = f"MATERIAL {i}:\n"
                source_content += f"Archivo: {source['metadata']['filename']}\n"
                source_content += f"Contenido: {source['content']}\n"
                source_content += "-" * 50
                theory_context_parts.append(source_content)
            
            theory_context = "\n\n".join(theory_context_parts)
            
            improved_examples = []
            
            for example in examples:
                prompt = f"""Eres un instructor de programación en Python. Tengo un ejemplo de código que cumple con un requerimiento, y quiero que lo adaptes basándote en los materiales del curso.

Requerimiento Original: {requirement}

Teoría del Curso y Mejores Prácticas:
{theory_context}

Ejemplo Actual:
Descripción: {example['description']}
Código: {example['code']}
Enfoque: {example['approach']}

Por favor arma un nuevo ejemplo:
1. Siguiendo las mejores prácticas mencionadas en la teoría del curso
2. Usando las funciones, métodos o enfoques recomendados de la teoría (prioriza hacerlo con funciones, metodos o enfoques del material provisto por más que cambie el código original)
3. Haciendo que el código esté más alineado con las enseñanzas del curso
4. Respetando el requerimiento original ya que el material solo define el estilo y no la consigna

IMPORTANTE:
- Escribe todo el código y explicaciones en ESPAÑOL únicamente
- Usa la indentación correcta de Python (4 espacios)
- Asegúrate de que el código esté bien formateado y sea legible

Proporciona tu respuesta usando formato XML:
<example>
<code>código python mejorado aquí</code>
<approach>explicación de las mejoras</approach>
<improvements>lista de mejoras</improvements>
<theory_alignment>cómo se alinea con la teoría del curso</theory_alignment>
</example>"""

                # Query LLM using LangChain
                messages = [
                    SystemMessage(content="Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML y escribe todo en ESPAÑOL únicamente."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.llm.invoke(messages)
                
                # Handle different response types (OpenAI has .content, Ollama returns string directly)
                if hasattr(response, 'content'):
                    response_text = response.content.strip()
                else:
                    response_text = str(response).strip()
                
                try:
                    # Parse XML response for improved example
                    xml_parsed = self._parse_xml_response(response_text, 1)
                    if xml_parsed:
                        improved_example = xml_parsed[0]
                        improved_example["example_id"] = example["example_id"]
                        improved_example["original_code"] = example["code"]
                        improved_examples.append(improved_example)
                    else:
                        # Keep original example if parsing fails
                        example["improvements"] = ["Failed to parse improvements"]
                        example["theory_alignment"] = "Could not analyze theory alignment"
                        improved_examples.append(example)
                        
                except Exception as e:
                    logger.error(f"Failed to parse improved example XML: {e}")
                    # Keep original example if parsing fails
                    example["improvements"] = ["Failed to parse improvements"]
                    example["theory_alignment"] = "Could not analyze theory alignment"
                    improved_examples.append(example)
            
            logger.info(f"Improved {len(improved_examples)} examples with theory")
            return improved_examples
            
        except Exception as e:
            logger.error(f"Error improving examples with theory: {e}")
            return examples
    
    @traceable(name="generate_examples")
    async def generate_examples(self, requirement: str, num_examples: int = 3) -> Dict[str, Any]:
        """Main method to generate and improve code examples"""
        try:
            logger.info(f"Generating {num_examples} examples for requirement: {requirement}")
            
            # Step 1: Generate initial examples
            initial_examples = await self.generate_initial_examples(requirement, num_examples)
            
            if not initial_examples:
                return {
                    "requirement": requirement,
                    "examples": [],
                    "theory_sources": [],
                    "error": "Failed to generate initial examples"
                }
            
            # Step 2: Get relevant theory from notebooks using both requirement and examples
            theory_sources = await self.get_relevant_theory(requirement, initial_examples, max_results=5)
            
            # Step 3: Improve examples with theory
            if theory_sources:
                improved_examples = await self.improve_examples_with_theory(
                    initial_examples, requirement, theory_sources
                )
            else:
                logger.warning("No theory sources found, using initial examples")
                improved_examples = initial_examples
                for example in improved_examples:
                    example["improvements"] = ["No theory sources available for improvement"]
                    example["theory_alignment"] = "No course theory found for this requirement"
            
            return {
                "requirement": requirement,
                "examples": improved_examples,
                "theory_sources": theory_sources,
                "num_examples": len(improved_examples),
                "has_theory_improvement": len(theory_sources) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in generate_examples: {e}")
            return {
                "requirement": requirement,
                "examples": [],
                "theory_sources": [],
                "error": str(e)
            }
