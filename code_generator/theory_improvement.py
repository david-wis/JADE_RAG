"""
Module for improving code examples based on theoretical content from notebooks.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .shared import LLMFactory, XMLParser
from config import TEMPERATURE_THEORY_CORRECTION

logger = logging.getLogger(__name__)


class TheoryImprover:
    """Handles improvement of code examples using theoretical content."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
    
    async def get_relevant_theory(self, rag_system, requirement: str, examples: Optional[List[Dict[str, Any]]] = None, max_results: int = 5, max_class_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get relevant theoretical content from notebooks using RAG."""
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
            theory_sources = await rag_system.retrieve_documents(combined_query, max_results=max_results, max_class_number=max_class_number)
            
            return theory_sources
                
        except Exception as e:
            logger.error(f"Error getting relevant theory: {e}")
            return []
    
    @traceable(name="improve_examples_with_theory")
    async def improve_examples_with_theory(self, examples: List[Dict[str, Any]], 
                                         requirement: str, 
                                         theory_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improve examples based on theoretical best practices from notebooks."""
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

El objetivo es crear otro programa EQUIVALENTE al original, pero siguiendo las recomendaciones de la teoría.

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
3. Respetando el requerimiento original ya que el material solo define el estilo y no la consigna
4. Manteniendo inputs, outputs y comportamiento del programa original.
5. Si el requerimiento no indica dar ejemplos de ejecucion o comentarios, no los agregues

IMPORTANTE:
- Usa la indentación correcta de Python (4 espacios)
- Asegúrate de que el código esté bien formateado y sea legible
- El código debe ser correcto, pero no es necesario que sea eficiente
- Debes asumir tus conocimientos son básicos (por ejemplo, no uses clases ni lambdas).
- El código debe ser lo más imperativo posible (evita usar funciones de Python que simplifiquen el código)
- El código debe ser muy simple y corto.

Ejemplo:
Supongamos que el requerimiento es "Crear una función que valide la existencia de un archivo txt" y hay un ejemplo de la teoria sobre como leer un csv y validar que tenga exactamente 10 filas.
- Si hay una explicación teoríca relevante (no la consigna de un ejercicio), deberías tenerla en cuenta.
- Deberías usar el ejemplo de la teoria para crear un nuevo ejemplo que valide la existencia de un archivo txt (por ejemplo usando open si el material lo indica).
- No deberías modificar el código para que trabaje con csvs.
- No deberías tener en cuenta la validación de cantidad de filas.

Antes de dar tu respuesta quiero que pienses si el ejemplo que estas proponiendo es verdaderamente equivalente al original y lo corrijas si no lo es.
Quiero que indiques por que son equivalentes dentro del apartado de approach.

Proporciona tu respuesta usando formato XML:
<example>
<code>código python mejorado aquí</code>
<approach>explicación de las mejoras y por que el código es equivalente al original</approach>
<improvements>lista de mejoras</improvements>
<theory_alignment>cómo se alinea con la teoría del curso</theory_alignment>
</example>"""

                # Create LLM instance with lower temperature for theory-based correction
                theory_llm = self.llm_factory.create_llm(temperature=TEMPERATURE_THEORY_CORRECTION)
                    
                messages = [
                    SystemMessage(content="Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML y escribe todo en ESPAÑOL únicamente."),
                    HumanMessage(content=prompt)
                ]
                
                response = theory_llm.invoke(messages)
                
                # Handle different response types (OpenAI has .content, Ollama returns string directly)
                if hasattr(response, 'content'):
                    response_text = getattr(response, 'content', '').strip()
                else:
                    response_text = str(response).strip()
                
                try:
                    # Parse XML response for improved example
                    xml_parsed = self.xml_parser.parse_xml_response(response_text, 1)
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
