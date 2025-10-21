"""
Module for improving code examples based on theoretical content from notebooks.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .shared import LLMFactory, XMLParser
from .prompt_templates import PromptTemplates, Language
from config import TEMPERATURE_THEORY_CORRECTION

logger = logging.getLogger(__name__)


class TheoryImprover:
    """Handles improvement of code examples using theoretical content."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
    
    async def get_relevant_theory(self, rag_system, requirement: str, examples: Optional[List[Dict[str, Any]]] = None, max_results: int = 5, max_class_number: Optional[int] = None, dataset: str = "python") -> List[Dict[str, Any]]:
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
            theory_sources = await rag_system.retrieve_documents(combined_query, max_results=max_results, max_class_number=max_class_number, dataset=dataset)
            
            return theory_sources
                
        except Exception as e:
            logger.error(f"Error getting relevant theory: {e}")
            return []
    
    @traceable(name="improve_examples_with_theory")
    async def improve_examples_with_theory(self, examples: List[Dict[str, Any]], 
                                         requirement: str, 
                                         theory_sources: List[Dict[str, Any]],
                                         language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
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
                # Get language-specific prompt
                prompt = self.prompt_templates.format_template(
                    language,
                    "theory_improvement",
                    requirement=requirement,
                    theory_context=theory_context,
                    description=example['description'],
                    code=example['code'],
                    approach=example['approach']
                )
                
                # Get language-specific system message
                system_message = self.prompt_templates.get_system_message(language, "theory_improvement")

                # Create LLM instance with lower temperature for theory-based correction
                theory_llm = self.llm_factory.create_llm(temperature=TEMPERATURE_THEORY_CORRECTION)
                    
                messages = [
                    SystemMessage(content=system_message),
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
