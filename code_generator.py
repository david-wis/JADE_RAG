import os
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
import logging
from rag_system import RAGSystem
from config import (
    AI_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    OLLAMA_HOST,
    OLLAMA_PORT,
    MODEL_NAME,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenAI if using OpenAI provider
if AI_PROVIDER == "openai":
    from openai import AsyncOpenAI
else:
    import ollama


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
        
        # Initialize AI client
        self.openai_client = None
        self.ollama_client = None
        
    async def initialize(self):
        """Initialize the AI client based on provider"""
        try:
            if self.ai_provider == "openai":
                if not self.openai_api_key:
                    raise Exception("OpenAI API key is required when using OpenAI provider")
                
                self.openai_client = AsyncOpenAI(
                    api_key=self.openai_api_key, 
                    base_url=self.openai_base_url
                )
                logger.info(f"Initialized OpenAI client with model: {self.openai_model}")
            else:
                # Initialize Ollama client
                self.ollama_client = ollama.AsyncClient(
                    host=f"http://{self.ollama_host}:{self.ollama_port}"
                )
                logger.info(f"Initialized Ollama client with model: {self.model_name}")
                
            logger.info("Code example generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {e}")
            raise
    
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

            if self.ai_provider == "openai" and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                    max_tokens=2000,
                )
                response_text = (response.choices[0].message.content or "").strip()
            elif self.ollama_client:
                # Query Ollama
                response = await self.ollama_client.generate(
                    model=self.model_name, 
                    prompt=prompt, 
                    stream=False
                )
                # Handle Ollama response properly
                try:
                    if isinstance(response, dict):
                        response_text = (response.get("response") or "").strip()
                    else:
                        response_text = str(response).strip()
                except Exception:
                    response_text = str(response).strip()
            else:
                raise Exception("No AI client available")
            
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
    
    async def get_relevant_theory(self, requirement: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant theoretical content from notebooks using RAG"""
        try:
            # Query the RAG system for relevant theory
            rag_result = await self.rag_system.query(requirement, max_results=max_results)
            
            if rag_result and "sources" in rag_result:
                return rag_result["sources"]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting relevant theory: {e}")
            return []
    
    async def improve_examples_with_theory(self, examples: List[Dict[str, Any]], 
                                         requirement: str, 
                                         theory_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improve examples based on theoretical best practices from notebooks"""
        try:
            if not theory_sources:
                logger.warning("No theory sources available for improvement")
                return examples
            
            # Create context from theory sources
            theory_context = "\n\n".join([
                f"Source: {source['metadata']['filename']}\n{source['content']}"
                for source in theory_sources
            ])
            
            improved_examples = []
            
            for example in examples:
                prompt = f"""Eres un instructor de programación en Python. Tengo un ejemplo de código que cumple con un requerimiento, y quiero que lo mejores basándote en las mejores prácticas teóricas de los materiales del curso.

Requerimiento Original: {requirement}

Teoría del Curso y Mejores Prácticas:
{theory_context}

Ejemplo Actual:
Descripción: {example['description']}
Código: {example['code']}
Enfoque: {example['approach']}

Por favor mejora este ejemplo:
1. Siguiendo las mejores prácticas mencionadas en la teoría del curso
2. Usando las funciones, métodos o enfoques recomendados de la teoría
3. Haciendo que el código esté más alineado con las enseñanzas del curso
4. Manteniendo la misma funcionalidad central pero mejorando la implementación

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

                if self.ai_provider == "openai" and self.openai_client:
                    response = await self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML y escribe todo en ESPAÑOL únicamente."
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,
                        max_tokens=1500,
                    )
                    response_text = (response.choices[0].message.content or "").strip()
                elif self.ollama_client:
                    # Query Ollama
                    response = await self.ollama_client.generate(
                        model=self.model_name, 
                        prompt=prompt, 
                        stream=False
                    )
                    # Handle Ollama response properly
                    try:
                        if isinstance(response, dict):
                            response_text = (response.get("response") or "").strip()
                        else:
                            response_text = str(response).strip()
                    except Exception:
                        response_text = str(response).strip()
                else:
                    raise Exception("No AI client available")
                
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
            
            # Step 2: Get relevant theory from notebooks
            theory_sources = await self.get_relevant_theory(requirement, max_results=5)
            
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
