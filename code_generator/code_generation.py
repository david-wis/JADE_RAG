"""
Module for initial code generation based on requirements.
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .shared import LLMFactory, XMLParser
from .prompt_templates import PromptTemplates, Language

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Handles initial code generation from requirements."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
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
    async def generate_initial_examples(self, requirement: str, num_examples: int = 3, language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
        """Generate initial code examples based on the requirement."""
        try:
            # Get language-specific prompt
            prompt = self.prompt_templates.format_template(
                language, 
                "code_generation", 
                requirement=requirement, 
                num_examples=num_examples
            )
            
            # Get language-specific system message
            system_message = self.prompt_templates.get_system_message(language, "code_generation")

            # Query LLM using LangChain
            if not self.llm:
                raise Exception("LLM not initialized")
                
            messages = [
                SystemMessage(content=system_message),
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
