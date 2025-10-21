"""
Module for calculating various metrics on generated code examples.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Ragas imports
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference, numeric_metric
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .shared import LLMFactory, XMLParser
from .prompt_templates import PromptTemplates, Language
from config import NUM_GENERATED_RUBRICS

logger = logging.getLogger(__name__)


@numeric_metric(name="answer_relevancy", allowed_values=(0, 1))
def answer_relevancy_metric(original_requirement: str, generated_code: str, generated_rubrics: List[str]) -> float:
    """
    Calcula la relevancia de la respuesta usando la fórmula:
    answer relevancy = (1/N) * Σ[i=1 to N] cos(E_gi, E_o)
    
    Donde:
    - E_gi: Embedding de la i-ésima rubrica generada
    - E_o: Embedding del prompt original
    - N: Cantidad de rubricas generadas
    
    Args:
        original_requirement: El requerimiento original (E_o)
        generated_code: El código generado (no usado en la fórmula)
        generated_rubrics: Lista de rubricas generadas a partir del código (E_gi)
    
    Returns:
        float: Score de relevancia entre 0 y 1
    """
    try:
        if not generated_rubrics or len(generated_rubrics) == 0:
            logger.warning("No generated rubrics provided for answer relevancy calculation")
            return 0.0
        
        # Inicializar el modelo de embeddings (usando un modelo multilingüe)
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Calcular embedding del requerimiento original (E_o)
        original_embedding = model.encode([original_requirement])
        
        # Calcular embeddings de todas las rubricas generadas (E_gi)
        generated_embeddings = model.encode(generated_rubrics)
        
        # Calcular similitudes coseno entre cada rubrica generada y el requerimiento original
        similarities = []
        for i, generated_embedding in enumerate(generated_embeddings):
            similarity = cosine_similarity([generated_embedding], original_embedding)[0][0]
            similarities.append(similarity)
            logger.info(f"Cosine similarity for rubric {i+1}: {similarity:.3f}")
        
        # Aplicar la fórmula: (1/N) * Σ[i=1 to N] cos(E_gi, E_o)
        N = len(generated_rubrics)
        answer_relevancy = sum(similarities) / N
        
        # Asegurar que el score esté entre 0 y 1
        final_score = float(max(0, min(1, answer_relevancy)))
        
        logger.info(f"Answer relevancy calculated: {final_score:.3f} (average of {N} rubrics)")
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculating answer relevancy: {e}")
        return 0.0


class MetricsCalculator:
    """Handles calculation of various metrics for code examples."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
        self.context_precision_metric = None
        self.ragas_llm = None
    
    async def initialize(self):
        """Initialize the metrics calculator with Ragas components."""
        try:
            # Initialize Ragas LLM for metrics
            self.ragas_llm = self.llm_factory.create_ragas_llm()
            
            # Initialize Ragas context precision metric
            if self.ragas_llm:
                self.context_precision_metric = LLMContextPrecisionWithReference(llm=self.ragas_llm)
                logger.info("Ragas context precision metric initialized successfully")
                
            logger.info("Metrics calculator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics calculator: {e}")
            raise
    
    async def calculate_context_precision(self, requirement: str, ground_truth: str, 
                                        examples: List[Dict[str, Any]], 
                                        theory_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate context precision scores for each example using Ragas."""
        if not self.context_precision_metric or not ground_truth:
            logger.info("Context precision calculation skipped - no metric or ground truth provided")
            return examples
        
        # Prepare retrieved contexts from theory sources
        retrieved_contexts = []
        for source in theory_sources:
            retrieved_contexts.append(source['content'])
        
        if not retrieved_contexts:
            logger.warning("No retrieved contexts available for context precision calculation")
            return examples
        
        enhanced_examples = []
        
        for example in examples:
            # Create a sample for Ragas evaluation
            sample = SingleTurnSample(
                user_input=requirement,
                reference=ground_truth,
                retrieved_contexts=retrieved_contexts,
                rubrics={
                    "score0_description": "The context is completely irrelevant to the requirement",
                    "score1_description": "The context is fully relevant to the requirement because it provides best practices or tips that can be applied to the requirement",
                }
            )
            
            # Calculate context precision score
            score = await self.context_precision_metric.single_turn_ascore(sample)
            
            # Add the score to the example
            enhanced_example = example.copy()
            enhanced_example["context_precision_score"] = float(score)
            enhanced_examples.append(enhanced_example)
            
            logger.info(f"Context precision score for example {example.get('example_id', 'unknown')}: {score:.3f}")
        
        return enhanced_examples
    
    @traceable(name="infer_requirements_from_code")
    async def infer_requirements_from_code(self, code: str, language: Language = Language.PYTHON) -> List[str]:
        """Infer multiple possible requirements that could have generated the given code."""
        try:
            # Get language-specific prompt
            prompt = self.prompt_templates.format_template(
                language,
                "rubrics_inference",
                code=code,
                num_rubrics=NUM_GENERATED_RUBRICS
            )
            
            # Get language-specific system message
            system_message = self.prompt_templates.get_system_message(language, "rubrics_inference")

            # Create LLM instance for inference
            inference_llm = self.llm_factory.create_llm()
                
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = inference_llm.invoke(messages)
            
            # Handle different response types (OpenAI has .content, Ollama returns string directly)
            if hasattr(response, 'content') and getattr(response, 'content', None):
                response_text = getattr(response, 'content', '').strip()
            else:
                response_text = str(response).strip()
            
            # Parse XML response to extract multiple requirements
            requirements = self.xml_parser.parse_rubrics_xml_response(response_text)
            
            logger.info(f"Generated {len(requirements)} requirements from code")
            return requirements
            
        except Exception as e:
            logger.error(f"Error inferring requirements from code: {e}")
            return [f"Error al inferir el requerimiento {i+1}" for i in range(NUM_GENERATED_RUBRICS)]
    
    async def calculate_answer_relevancy(self, requirement: str, examples: List[Dict[str, Any]], language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
        """Calculate answer relevancy scores for each example using the custom metric."""
        try:
            enhanced_examples = []
            
            for example in examples:
                if "code" not in example or not example["code"]:
                    logger.warning(f"No code found for example {example.get('example_id', 'unknown')}")
                    enhanced_example = example.copy()
                    enhanced_example["answer_relevancy_score"] = 0.0
                    enhanced_example["generated_rubrics"] = []
                    enhanced_examples.append(enhanced_example)
                    continue
                
                # Infer multiple requirements from the generated code
                generated_rubrics = await self.infer_requirements_from_code(example["code"], language)
                
                # Calculate answer relevancy score using the custom metric with the formula
                relevancy_score = answer_relevancy_metric(
                    original_requirement=requirement,
                    generated_code=example["code"],
                    generated_rubrics=generated_rubrics
                )
                
                # Add the score and generated rubrics to the example
                enhanced_example = example.copy()
                enhanced_example["answer_relevancy_score"] = float(relevancy_score) if relevancy_score is not None else 0.0
                enhanced_example["generated_rubrics"] = generated_rubrics
                enhanced_examples.append(enhanced_example)
                
                logger.info(f"Answer relevancy score for example {example.get('example_id', 'unknown')}: {relevancy_score:.3f} (based on {len(generated_rubrics)} rubrics)")
            
            return enhanced_examples
            
        except Exception as e:
            logger.error(f"Error calculating answer relevancy: {e}")
            return examples
