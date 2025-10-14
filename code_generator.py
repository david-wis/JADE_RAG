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

# Ragas imports
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference, numeric_metric
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
    NUM_GENERATED_RUBRICS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangSmith for tracing
from langsmith import traceable


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
        
        # Ragas configuration
        self.context_precision_metric = None
        
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
            
            # Initialize Ragas context precision metric
            if self.llm:
                self.context_precision_metric = LLMContextPrecisionWithReference(llm=self.llm)
                logger.info("Ragas context precision metric initialized successfully")
                
            logger.info("Code example generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {e}")
            raise
    
    @traceable(name="generate_initial_examples")
    async def generate_initial_examples(self, requirement: str, num_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate initial code examples based on the requirement"""
        try:
            prompt = f"""Eres un alumno de una materia de programación en Python. Dado el siguiente requerimiento, genera {num_examples} ejemplos de código diferentes que cumplan con este requerimiento.

Requerimiento: {requirement}

Para cada ejemplo, proporciona:
1. El código Python completo
2. Una breve explicación del enfoque utilizado

IMPORTANTE: 
- Usa la indentación correcta de Python (4 espacios)
- Asegúrate de que el código esté bien formateado y sea legible
- El código debe ser correcto, pero no es necesario que sea eficiente
- Debes asumir tus conocimientos son básicos (por ejemplo, no uses clases ni lambdas).
- El código debe ser lo más imperativo posible (evita usar funciones de Python que simplifiquen el código)

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
- Usa la indentación correcta de Python (4 espacios)
- Asegúrate de que el código esté bien formateado y sea legible
- El código debe ser correcto, pero no es necesario que sea eficiente
- Debes asumir tus conocimientos son básicos (por ejemplo, no uses clases ni lambdas).
- El código debe ser lo más imperativo posible (evita usar funciones de Python que simplifiquen el código)

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
    
    async def calculate_context_precision(self, requirement: str, ground_truth: str, 
                                        examples: List[Dict[str, Any]], 
                                        theory_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate context precision scores for each example using Ragas"""
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
    async def infer_requirements_from_code(self, code: str) -> List[str]:
        """Infer multiple possible requirements that could have generated the given code"""
        try:
            prompt = f"""Eres un instructor de programación en Python. Dado el siguiente código Python, infiere {NUM_GENERATED_RUBRICS} posibles requerimientos o consignas diferentes que podrían haber llevado a un estudiante a escribir este código.

EJEMPLO DE REFERENCIA:

Código de ejemplo:
```python
def sumar_pares(lista):
    suma = 0
    for numero in lista:
        if numero % 2 == 0:
            suma += numero
    return suma
```

Rubricas generadas:
<rubric>
<requirement>La función debe tomar una lista de números y devolver la suma de todos los números pares</requirement>
</rubric>

<rubric>
<requirement>`sumar_pares` debe recorrer una lista y sumar únicamente los elementos que son divisibles por 2</requirement>
</rubric>

<rubric>
<requirement>Debe calcular la suma total de los números pares presentes en una lista de enteros</requirement>
</rubric>

---

AHORA ANALIZA ESTE CÓDIGO:

Código:
```python
{code}
```

Por favor, proporciona {NUM_GENERATED_RUBRICS} requerimientos diferentes siguiendo el mismo formato del ejemplo. Cada requerimiento debe:
1. Ser claro y específico
2. Explicar qué funcionalidad se esperaba
3. Estar escrito en español
4. Ser conciso pero completo
5. Representar diferentes interpretaciones posibles del código
6. Usar diferentes palabras pero describir la misma funcionalidad básica

Formatea tu respuesta usando formato XML con la siguiente estructura:
<rubric>
<requirement>primer requerimiento aquí</requirement>
</rubric>

<rubric>
<requirement>segundo requerimiento aquí</requirement>
</rubric>

<rubric>
<requirement>tercer requerimiento aquí</requirement>
</rubric>

"""

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
            
            # Parse XML response to extract multiple requirements
            requirements = self._parse_rubrics_xml_response(response_text)
            
            logger.info(f"Generated {len(requirements)} requirements from code")
            return requirements
            
        except Exception as e:
            logger.error(f"Error inferring requirements from code: {e}")
            return [f"Error al inferir el requerimiento {i+1}" for i in range(NUM_GENERATED_RUBRICS)]
    
    def _parse_rubrics_xml_response(self, response_text: str) -> List[str]:
        """Parse XML response to extract multiple rubric requirements"""
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
    
    async def calculate_answer_relevancy(self, requirement: str, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate answer relevancy scores for each example using the custom metric"""
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
                generated_rubrics = await self.infer_requirements_from_code(example["code"])
                
                # Calculate answer relevancy score using the custom metric with the formula
                relevancy_score = answer_relevancy_metric(
                    original_requirement=requirement,
                    generated_code=example["code"],
                    generated_rubrics=generated_rubrics
                )
                
                # Add the score and generated rubrics to the example
                enhanced_example = example.copy()
                enhanced_example["answer_relevancy_score"] = float(relevancy_score)
                enhanced_example["generated_rubrics"] = generated_rubrics
                enhanced_examples.append(enhanced_example)
                
                logger.info(f"Answer relevancy score for example {example.get('example_id', 'unknown')}: {relevancy_score:.3f} (based on {len(generated_rubrics)} rubrics)")
            
            return enhanced_examples
            
        except Exception as e:
            logger.error(f"Error calculating answer relevancy: {e}")
            return examples
    
    @traceable(name="generate_examples")
    async def generate_examples(self, requirement: str, num_examples: int = 3, ground_truth: Optional[str] = None) -> Dict[str, Any]:
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
            
            # Step 4: Calculate context precision scores if ground truth is provided
            if ground_truth and theory_sources:
                improved_examples = await self.calculate_context_precision(
                    requirement, ground_truth, improved_examples, theory_sources
                )
            
            # Step 5: Calculate answer relevancy scores for all examples
            improved_examples = await self.calculate_answer_relevancy(requirement, improved_examples)
            
            return {
                "requirement": requirement,
                "examples": improved_examples,
                "theory_sources": theory_sources,
                "num_examples": len(improved_examples),
                "has_theory_improvement": len(theory_sources) > 0,
                "has_context_precision": ground_truth is not None and len(theory_sources) > 0,
                "has_answer_relevancy": True
            }
            
        except Exception as e:
            logger.error(f"Error in generate_examples: {e}")
            return {
                "requirement": requirement,
                "examples": [],
                "theory_sources": [],
                "error": str(e)
            }
