"""
Module for filtering theory-specific elements from improved code examples.
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .shared import LLMFactory, XMLParser
from .prompt_templates import PromptTemplates, Language
from config import TEMPERATURE_FILTERING

logger = logging.getLogger(__name__)


class TheoryFilter:
    """Handles filtering of theory-specific elements from improved examples."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
    
    @traceable(name="filter_theory_specific_elements")
    async def filter_theory_specific_elements(self, examples: List[Dict[str, Any]], 
                                            requirement: str, 
                                            theory_sources: List[Dict[str, Any]],
                                            language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
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
                
                # Get language-specific prompt
                prompt = self.prompt_templates.format_template(
                    language,
                    "filtering",
                    requirement=requirement,
                    theory_context=theory_context,
                    original_code=original_code,
                    improved_code=example.get('code', 'N/A'),
                    description=example.get('description', 'N/A'),
                    approach=example.get('approach', 'N/A'),
                    improvements=example.get('improvements', 'N/A'),
                    theory_alignment=example.get('theory_alignment', 'N/A')
                )
                
                # Get language-specific system message
                system_message = self.prompt_templates.get_system_message(language, "filtering")

                # Create LLM instance with low temperature for precise filtering
                filter_llm = self.llm_factory.create_llm(temperature=TEMPERATURE_FILTERING)
                    
                messages = [
                    SystemMessage(content=system_message),
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
