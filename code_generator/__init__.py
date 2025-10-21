"""
Code Generator Module

This module provides functionality for generating, improving, and evaluating code examples
based on requirements and theoretical content from notebooks.

Main Components:
- CodeGenerator: Generates initial code examples from requirements
- TheoryImprover: Improves examples using theoretical content
- TheoryFilter: Filters theory-specific elements (optional)
- MetricsCalculator: Calculates various metrics for examples
"""

import logging
from typing import List, Dict, Any, Optional
from langsmith import traceable

from .shared import LLMFactory
from .code_generation import CodeGenerator
from .theory_improvement import TheoryImprover
from .filtering import TheoryFilter
from .metrics import MetricsCalculator
from .prompt_templates import Language

logger = logging.getLogger(__name__)


class CodeExampleGenerator:
    """Main class that orchestrates the code generation, improvement, and evaluation process."""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.llm_factory = LLMFactory()
        
        # Initialize components
        self.code_generator = CodeGenerator(self.llm_factory)
        self.theory_improver = TheoryImprover(self.llm_factory)
        self.theory_filter = TheoryFilter(self.llm_factory)
        self.metrics_calculator = MetricsCalculator(self.llm_factory)
        
        # Configuration
        from config import ENABLE_FILTERING
        self.enable_filtering = ENABLE_FILTERING
        
    async def initialize(self):
        """Initialize all components."""
        try:
            await self.code_generator.initialize()
            await self.metrics_calculator.initialize()
            logger.info("Code example generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {e}")
            raise
    
    @traceable(name="generate_examples")
    async def generate_examples(self, requirement: str, num_examples: int = 3, ground_truth: Optional[str] = None, max_class_number: Optional[int] = None, dataset: str = "python", language: Language = Language.PYTHON) -> Dict[str, Any]:
        """Main method to generate and improve code examples."""
        try:
            logger.info(f"Generating {num_examples} examples for requirement: {requirement} using {dataset} dataset")
            
            # Step 1: Generate initial examples
            initial_examples = await self.code_generator.generate_initial_examples(requirement, num_examples, language)
            
            if not initial_examples:
                return {
                    "requirement": requirement,
                    "examples": [],
                    "theory_sources": [],
                    "error": "Failed to generate initial examples"
                }
            
            # Step 2: Get relevant theory from notebooks using both requirement and examples
            theory_sources = await self.theory_improver.get_relevant_theory(
                self.rag_system, requirement, initial_examples, max_results=5, max_class_number=max_class_number, dataset=dataset
            )
            
            # Step 3: Improve examples with theory
            if theory_sources:
                improved_examples = await self.theory_improver.improve_examples_with_theory(
                    initial_examples, requirement, theory_sources, language
                )
            else:
                logger.warning(f"No theory sources found for {dataset} dataset, using initial examples")
                improved_examples = initial_examples
                for example in improved_examples:
                    example["improvements"] = [f"No theory sources available for improvement in {dataset} dataset"]
                    example["theory_alignment"] = f"No course theory found for this requirement in {dataset} dataset"
            
            # Step 4: Filter theory-specific elements (if enabled)
            if self.enable_filtering and theory_sources:
                filtered_examples = await self.theory_filter.filter_theory_specific_elements(
                    improved_examples, requirement, theory_sources, language
                )
            else:
                logger.info("Filtering disabled or no theory sources available")
                filtered_examples = improved_examples
                for example in filtered_examples:
                    example["was_filtered"] = False
                    example["filtering_summary"] = "Filtering disabled or no theory sources available"
            
            # Step 5: Calculate context precision scores if ground truth is provided
            if ground_truth and theory_sources:
                filtered_examples = await self.metrics_calculator.calculate_context_precision(
                    requirement, ground_truth, filtered_examples, theory_sources
                )
            
            # Step 6: Calculate answer relevancy scores for all examples
            filtered_examples = await self.metrics_calculator.calculate_answer_relevancy(requirement, filtered_examples, language)
            
            return {
                "requirement": requirement,
                "examples": filtered_examples,
                "theory_sources": theory_sources,
                "num_examples": len(filtered_examples),
                "has_theory_improvement": len(theory_sources) > 0,
                "has_theory_filtering": self.enable_filtering and len(theory_sources) > 0,
                "has_context_precision": ground_truth is not None and len(theory_sources) > 0,
                "has_answer_relevancy": True,
                "dataset": dataset,
                "language": language.value
            }
            
        except Exception as e:
            logger.error(f"Error in generate_examples: {e}")
            return {
                "requirement": requirement,
                "examples": [],
                "theory_sources": [],
                "error": str(e)
            }


# Export main class and components for direct access if needed
__all__ = [
    'CodeExampleGenerator',
    'CodeGenerator', 
    'TheoryImprover',
    'TheoryFilter',
    'MetricsCalculator',
    'LLMFactory',
    'Language'
]
