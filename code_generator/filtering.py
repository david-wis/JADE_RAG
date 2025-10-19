"""
Module for filtering theory-specific elements from improved code examples.
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .shared import LLMFactory, XMLParser
from config import TEMPERATURE_FILTERING

logger = logging.getLogger(__name__)


class TheoryFilter:
    """Handles filtering of theory-specific elements from improved examples."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
    
    @traceable(name="filter_theory_specific_elements")
    async def filter_theory_specific_elements(self, examples: List[Dict[str, Any]], 
                                            requirement: str, 
                                            theory_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out theory-specific elements that are not relevant to the original requirement."""
        try:
            if not theory_sources:
                logger.info("No theory sources available for filtering")
                return examples
            
            # Create context from theory sources for filtering analysis
            theory_context_parts = []
            for i, source in enumerate(theory_sources, 1):
                source_content = f"MATERIAL {i}:\n"
                source_content += f"Archivo: {source['metadata']['filename']}\n"
                source_content += f"Contenido: {source['content']}\n"
                source_content += "-" * 50
                theory_context_parts.append(source_content)
            
            theory_context = "\n\n".join(theory_context_parts)
            
            filtered_examples = []
            
            for example in examples:
                # Get original code if available (from the initial examples before theory improvement)
                original_code = example.get('original_code', 'No original code available')
                
                prompt = f"""Eres un instructor de programación en Python experto en análisis de código. Tu tarea es revisar un ejemplo de código que fue mejorado basándose en material teórico y determinar si contiene elementos muy específicos del material que no son relevantes para el requerimiento original.

Requerimiento Original: {requirement}

Material Teórico Utilizado:
{theory_context}

COMPARACIÓN DE CÓDIGOS:

Código Original (antes de mejoras con teoría):
```python
{original_code}
```

Código Mejorado (después de aplicar teoría):
```python
{example.get('code', 'N/A')}
```

Información del Ejemplo:
Descripción: {example.get('description', 'N/A')}
Enfoque Original: {example.get('approach', 'N/A')}
Mejoras Aplicadas: {example.get('improvements', 'N/A')}
Alineación con Teoría: {example.get('theory_alignment', 'N/A')}

INSTRUCCIONES:
1. Compara el código original con el código mejorado para identificar qué elementos se agregaron del material teórico
2. Analiza si los elementos agregados son relevantes para cumplir el requerimiento original
3. Identifica si se agregaron funcionalidades, validaciones o lógica que van más allá de lo que pide la consigna
4. Si encuentras elementos irrelevantes o que eran parte de un ejercicio del material pero no del requerimiento, proporciona una versión filtrada del código
5. Si el código mejorado está bien y solo contiene elementos relevantes, devuelve el código mejorado sin cambios

CRITERIOS PARA FILTRAR:
- Validaciones específicas del material que no están en el requerimiento
- Funcionalidades adicionales que no se pidieron
- Lógica compleja que no es necesaria para la consigna básica
- Elementos de ejemplo del material que se incorporaron innecesariamente
- Controles de flujo del material que si bien aportan valor, no están explicitos en el requerimiento

CRITERIOS PARA MANTENER:
- Estructura y estilo recomendado por la teoría
- Funciones y métodos apropiados mencionados en el material
- Mejores prácticas de programación
- Código que cumple exactamente con el requerimiento

IMPORTANTE:
- En la mayoría de los casos NO deberías modificar el código mejorado
- Solo filtra cuando hay elementos claramente irrelevantes agregados del material teórico
- Mantén la funcionalidad del código original intacta
- Preserva inputs, outputs y comportamiento del programa original
- El objetivo es mantener solo las mejoras relevantes del material teórico

Proporciona tu respuesta usando formato XML:
<analysis>
<has_irrelevant_elements>true/false</has_irrelevant_elements>
<irrelevant_elements>descripción de elementos irrelevantes encontrados (si los hay)</irrelevant_elements>
<filtering_justification>justificación de por qué se filtraron o no se filtraron elementos</filtering_justification>
</analysis>

<filtered_example>
<code>código filtrado (o original si no hay cambios)</code>
<approach>explicación actualizada del enfoque</approach>
<filtering_summary>resumen de los cambios realizados (si los hubo)</filtering_summary>
</filtered_example>"""

                # Create LLM instance with low temperature for precise filtering
                filter_llm = self.llm_factory.create_llm(temperature=TEMPERATURE_FILTERING)
                    
                messages = [
                    SystemMessage(content="Eres un instructor de programación en Python muy útil. Siempre responde usando formato XML y escribe todo en ESPAÑOL únicamente."),
                    HumanMessage(content=prompt)
                ]
                
                response = filter_llm.invoke(messages)
                
                # Handle different response types (OpenAI has .content, Ollama returns string directly)
                if hasattr(response, 'content'):
                    response_text = getattr(response, 'content', '').strip()
                else:
                    response_text = str(response).strip()
                
                try:
                    # Parse XML response for filtered example
                    filtered_example = self.xml_parser.parse_filtered_xml_response(response_text, example)
                    filtered_examples.append(filtered_example)
                        
                except Exception as e:
                    logger.error(f"Failed to parse filtered example XML: {e}")
                    # Keep original example if parsing fails
                    example["filtering_error"] = "Failed to parse filtering response"
                    example["filtering_summary"] = "No filtering applied due to parsing error"
                    filtered_examples.append(example)
            
            logger.info(f"Filtered {len(filtered_examples)} examples for theory-specific elements")
            return filtered_examples
            
        except Exception as e:
            logger.error(f"Error filtering theory-specific elements: {e}")
            return examples
