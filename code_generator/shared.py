"""
Shared dependencies and utilities for the code_generator module.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import SecretStr
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
    TEMPERATURE_EXAMPLE_GENERATION,
    TEMPERATURE_THEORY_CORRECTION,
    TEMPERATURE_FILTERING,
)

# Configure logging
logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory class for creating LLM instances with different configurations."""
    
    def __init__(self):
        self.ai_provider = AI_PROVIDER
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.openai_base_url = OPENAI_BASE_URL
        self.ollama_host = OLLAMA_HOST
        self.ollama_port = OLLAMA_PORT
        self.model_name = MODEL_NAME
    
    def create_llm(self, temperature: float = TEMPERATURE_EXAMPLE_GENERATION):
        """Create an LLM instance with a specific temperature."""
        if self.ai_provider == "openai":
            if not self.openai_api_key:
                raise Exception("OpenAI API key is required when using OpenAI provider")
            
            return ChatOpenAI(
                model=self.openai_model,
                api_key=SecretStr(self.openai_api_key) if self.openai_api_key else None,
                base_url=self.openai_base_url,
                temperature=temperature
            )
        else:
            # Initialize Ollama LLM
            return OllamaLLM(
                model=self.model_name,
                temperature=temperature,
                base_url=f"http://{self.ollama_host}:{self.ollama_port}"
            )
    
    def create_ragas_llm(self):
        """Create an LLM instance for Ragas metrics."""
        if self.ai_provider == "openai":
            from ragas.llms.base import llm_factory
            return llm_factory('gpt-4o-mini')
        else:
            from ragas.llms.base import llm_factory
            return llm_factory(self.model_name)


class XMLParser:
    """Utility class for parsing XML responses from LLMs."""
    
    @staticmethod
    def parse_xml_response(response_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Parse XML response to extract code examples."""
        import re
        
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
    
    @staticmethod
    def parse_filtered_xml_response(response_text: str, original_example: Dict[str, Any]) -> Dict[str, Any]:
        """Parse XML response to extract filtered example."""
        import re
        
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Look for analysis and filtered_example patterns
            analysis_pattern = r'<analysis[^>]*>(.*?)</analysis>'
            filtered_example_pattern = r'<filtered_example[^>]*>(.*?)</filtered_example>'
            
            # Extract analysis section
            analysis_match = re.search(analysis_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            analysis_content = analysis_match.group(1) if analysis_match else ""
            
            # Extract filtered example section
            filtered_match = re.search(filtered_example_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            filtered_content = filtered_match.group(1) if filtered_match else ""
            
            # Parse analysis elements
            has_irrelevant_pattern = r'<has_irrelevant_elements[^>]*>(.*?)</has_irrelevant_elements>'
            irrelevant_elements_pattern = r'<irrelevant_elements[^>]*>(.*?)</irrelevant_elements>'
            justification_pattern = r'<filtering_justification[^>]*>(.*?)</filtering_justification>'
            
            has_irrelevant = re.search(has_irrelevant_pattern, analysis_content, re.DOTALL | re.IGNORECASE)
            irrelevant_elements = re.search(irrelevant_elements_pattern, analysis_content, re.DOTALL | re.IGNORECASE)
            justification = re.search(justification_pattern, analysis_content, re.DOTALL | re.IGNORECASE)
            
            # Parse filtered example elements
            code_pattern = r'<code[^>]*>(.*?)</code>'
            approach_pattern = r'<approach[^>]*>(.*?)</approach>'
            summary_pattern = r'<filtering_summary[^>]*>(.*?)</filtering_summary>'
            
            filtered_code = re.search(code_pattern, filtered_content, re.DOTALL | re.IGNORECASE)
            filtered_approach = re.search(approach_pattern, filtered_content, re.DOTALL | re.IGNORECASE)
            filtering_summary = re.search(summary_pattern, filtered_content, re.DOTALL | re.IGNORECASE)
            
            # Create filtered example
            filtered_example = original_example.copy()
            
            # Update code if filtered version was provided
            if filtered_code and filtered_code.group(1).strip():
                filtered_example["code"] = filtered_code.group(1).strip()
                filtered_example["was_filtered"] = True
            else:
                filtered_example["was_filtered"] = False
            
            # Update approach if provided
            if filtered_approach and filtered_approach.group(1).strip():
                filtered_example["approach"] = filtered_approach.group(1).strip()
            
            # Add filtering metadata
            filtered_example["has_irrelevant_elements"] = has_irrelevant.group(1).strip().lower() == "true" if has_irrelevant else False
            filtered_example["irrelevant_elements"] = irrelevant_elements.group(1).strip() if irrelevant_elements else ""
            filtered_example["filtering_justification"] = justification.group(1).strip() if justification else ""
            filtered_example["filtering_summary"] = filtering_summary.group(1).strip() if filtering_summary else "No filtering applied"
            
            return filtered_example
            
        except Exception as e:
            logger.error(f"Error parsing filtered XML response: {e}")
            # Return original example with error metadata
            original_example["filtering_error"] = f"XML parsing error: {e}"
            original_example["was_filtered"] = False
            original_example["filtering_summary"] = "No filtering applied due to parsing error"
            return original_example
    
    @staticmethod
    def parse_rubrics_xml_response(response_text: str) -> List[str]:
        """Parse XML response to extract multiple rubric requirements."""
        import re
        from config import NUM_GENERATED_RUBRICS
        
        try:
            requirements = []
            
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Look for rubric patterns
            rubric_pattern = r'<rubric[^>]*>(.*?)</rubric>'
            requirement_pattern = r'<requirement[^>]*>(.*?)</requirement>'
            
            # Find all rubric blocks
            rubric_matches = re.findall(rubric_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            
            if rubric_matches:
                for rubric_content in rubric_matches:
                    # Extract requirement from each rubric
                    requirement_match = re.search(requirement_pattern, rubric_content, re.DOTALL | re.IGNORECASE)
                    if requirement_match:
                        requirement = requirement_match.group(1).strip()
                        requirements.append(requirement)
            else:
                # Fallback: try to extract from plain text
                lines = cleaned_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('<') and not line.startswith('```'):
                        requirements.append(line)
                        if len(requirements) >= NUM_GENERATED_RUBRICS:
                            break
            
            # Ensure we have the right number of requirements
            while len(requirements) < NUM_GENERATED_RUBRICS:
                requirements.append(f"Requerimiento {len(requirements) + 1} inferido del código")
            
            # Limit to the configured number
            requirements = requirements[:NUM_GENERATED_RUBRICS]
            
            logger.info(f"Parsed {len(requirements)} requirements from XML response")
            return requirements
            
        except Exception as e:
            logger.error(f"Error parsing rubrics XML response: {e}")
            return [f"Error al inferir el requerimiento {i+1}" for i in range(NUM_GENERATED_RUBRICS)]
