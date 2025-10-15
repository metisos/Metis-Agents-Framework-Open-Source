"""
Response Evaluator Component for Metis Agentic Orchestration System

This module provides comprehensive evaluation of task responses against original queries
to ensure quality, relevance, and completeness before final output to users.
"""

import json
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class EvaluationLevel(Enum):
    """Evaluation rigor levels"""
    SKIP = "skip"
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

class TaskType(Enum):
    """Task type classifications for specialized evaluation"""
    QUESTION = "question"
    CODE_GENERATION = "code_generation"
    CONTENT_CREATION = "content_creation"
    RESEARCH = "research"
    EMAIL_MANAGEMENT = "email_management"
    CALCULATION = "calculation"
    WEB_SCRAPING = "web_scraping"
    GENERAL_TASK = "general_task"

@dataclass
class EvaluationCriteria:
    """Evaluation criteria configuration"""
    name: str
    weight: float
    threshold: float
    prompt: str
    task_specific: bool = False

@dataclass
class EvaluationResult:
    """Evaluation result container"""
    overall_score: float
    dimension_scores: Dict[str, float]
    passes_threshold: bool
    confidence: float
    improvement_suggestions: List[str]
    regeneration_recommended: bool
    evaluation_time: float
    metadata: Dict[str, Any]

class EvaluatorConfig:
    """Configuration for the response evaluator"""
    
    def __init__(self):
        # Master switch to enable/disable evaluation
        self.enable_evaluation = False  # Set to False to completely disable evaluation
        
        # Quality thresholds by context
        self.quality_thresholds = {
            "critical": 8.5,    # High-stakes business queries
            "standard": 7.0,    # Normal queries
            "testing": 5.5,     # Development/testing
            "experimental": 4.0  # Experimental features
        }
        
        # Performance settings
        self.max_improvement_attempts = 2
        self.evaluation_timeout = 15  # seconds
        self.cache_expiry_hours = 24
        self.enable_async_evaluation = True
        
        # Evaluation levels by query characteristics
        self.evaluation_rules = {
            "skip_patterns": [
                r"^\s*(hello|hi|hey|greetings|howdy|what's up|how are you).*$",
                r"^\s*(thanks?|thank you).*$",
                r"^\s*(bye|goodbye|exit|quit).*$",
                r"^\s*(yes|no|ok|okay).*$"
            ],
            "basic_patterns": [
                r"simple calculation",
                r"what is \d+",
                r"define \w+"
            ],
            "comprehensive_patterns": [
                r"research paper",
                r"business plan",
                r"code.*implementation",
                r"analysis.*report"
            ]
        }

class ResponseEvaluator:
    """
    Main evaluator class that assesses response quality against original queries
    """
    
    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """Initialize the response evaluator"""
        from components.llm_interface import get_llm
        
        self.llm = get_llm()
        self.config = config or EvaluatorConfig()
        
        # Cache for evaluations and improvements
        self.evaluation_cache = {}
        self.improvement_cache = {}
        
        # Evaluation criteria definitions
        self.base_criteria = self._define_base_criteria()
        self.task_specific_criteria = self._define_task_specific_criteria()
        
        print("✅ ResponseEvaluator initialized with comprehensive evaluation framework")
    
    def _define_base_criteria(self) -> Dict[str, EvaluationCriteria]:
        """Define base evaluation criteria applicable to all responses"""
        return {
            "relevance": EvaluationCriteria(
                name="relevance",
                weight=0.30,
                threshold=7.0,
                prompt="How well does this response directly address the specific query and its intent?"
            ),
            "completeness": EvaluationCriteria(
                name="completeness", 
                weight=0.25,
                threshold=7.0,
                prompt="Does this response fully answer all aspects and components of the original query?"
            ),
            "accuracy": EvaluationCriteria(
                name="accuracy",
                weight=0.25,
                threshold=8.0,
                prompt="Is the information provided factually correct and reliable?"
            ),
            "clarity": EvaluationCriteria(
                name="clarity",
                weight=0.20,
                threshold=7.0,
                prompt="Is the response well-structured, clear, and easy to understand?"
            )
        }
    
    def _define_task_specific_criteria(self) -> Dict[TaskType, Dict[str, EvaluationCriteria]]:
        """Define specialized criteria for different task types"""
        return {
            TaskType.CODE_GENERATION: {
                "syntax_correctness": EvaluationCriteria(
                    name="syntax_correctness",
                    weight=0.25,
                    threshold=9.0,
                    prompt="Is the generated code syntactically correct and free of errors?",
                    task_specific=True
                ),
                "functionality": EvaluationCriteria(
                    name="functionality", 
                    weight=0.30,
                    threshold=8.0,
                    prompt="Does the code solve the specified problem and meet requirements?",
                    task_specific=True
                ),
                "best_practices": EvaluationCriteria(
                    name="best_practices",
                    weight=0.20,
                    threshold=7.0,
                    prompt="Does the code follow language best practices and conventions?",
                    task_specific=True
                )
            },
            TaskType.EMAIL_MANAGEMENT: {
                "task_completion": EvaluationCriteria(
                    name="task_completion",
                    weight=0.40,
                    threshold=8.0,
                    prompt="Were all requested email operations completed successfully?",
                    task_specific=True
                ),
                "formatting": EvaluationCriteria(
                    name="formatting",
                    weight=0.30,
                    threshold=7.0,
                    prompt="Is the email summary well-formatted and readable?",
                    task_specific=True
                )
            },
            TaskType.RESEARCH: {
                "source_quality": EvaluationCriteria(
                    name="source_quality",
                    weight=0.30,
                    threshold=7.0,
                    prompt="Are credible, authoritative sources referenced?",
                    task_specific=True
                ),
                "depth": EvaluationCriteria(
                    name="depth",
                    weight=0.25,
                    threshold=7.0,
                    prompt="Is the analysis sufficiently thorough and comprehensive?",
                    task_specific=True
                ),
                "balance": EvaluationCriteria(
                    name="balance",
                    weight=0.20,
                    threshold=6.0,
                    prompt="Does the response present balanced viewpoints on the topic?",
                    task_specific=True
                )
            },
            TaskType.CONTENT_CREATION: {
                "creativity": EvaluationCriteria(
                    name="creativity",
                    weight=0.25,
                    threshold=7.0,
                    prompt="Is the content original and creative?",
                    task_specific=True
                ),
                "engagement": EvaluationCriteria(
                    name="engagement",
                    weight=0.25,
                    threshold=7.0,
                    prompt="Is the content engaging and compelling?",
                    task_specific=True
                )
            },
            TaskType.QUESTION: {
                "conciseness": EvaluationCriteria(
                    name="conciseness",
                    weight=0.30,
                    threshold=7.0,
                    prompt="Is the answer direct and to the point without unnecessary elaboration?",
                    task_specific=True
                )
            }
        }
    
    def evaluate_response(self, original_query: str, response: Any, task_type_hint: str = None) -> EvaluationResult:
        """
        Main evaluation method that assesses response quality
        
        Args:
            original_query: The user's original query
            response: The generated response (dict or string)
            task_type_hint: Optional hint about query type ("question" or "task")
            
        Returns:
            EvaluationResult object with comprehensive assessment
        """
        start_time = time.time()
        
        # Check if evaluation is disabled globally
        if not self.config.enable_evaluation:
            print("⏩ Response evaluation is disabled. Skipping evaluation.")
            # Standardize response format for the skip evaluation
            if isinstance(response, str):
                response_dict = {"content": response}
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"content": str(response)}
                
            return self._create_skip_evaluation(original_query, response_dict, start_time)
        
        # Standardize response format
        if isinstance(response, str):
            response_content = response
            response_dict = {"content": response}
        elif isinstance(response, dict):
            response_dict = response
            response_content = response.get("content", str(response))
        else:
            response_content = str(response)
            response_dict = {"content": response_content}
            
        # Skip evaluation for very short responses (less than 50 characters)
        # This is a quick optimization for simple greetings and short answers
        if len(response_content.strip()) < 50:
            print(f"DEBUG - Skipping evaluation for short response: '{response_content.strip()}' (length: {len(response_content.strip())})")
            return self._create_skip_evaluation(original_query, response_dict, start_time)
        
        # Create cache key
        cache_key = self._create_cache_key(original_query, response_content)
        
        # Check cache for existing evaluation
        cached_result = self.evaluation_cache.get(cache_key)
        if cached_result:
            # Check if cache is still valid
            if time.time() - cached_result.metadata.get("timestamp", 0) < self.config.cache_expiry_hours * 3600:
                print(f"🔄 Using cached evaluation for query: {original_query[:50]}...")
                return cached_result
        
        # Determine evaluation level based on query
        eval_level = self._determine_evaluation_level(original_query)
        
        # Skip evaluation for simple queries if configured
        if eval_level == EvaluationLevel.SKIP:
            return self._create_skip_evaluation(original_query, response_dict, start_time)
        
        # Identify task type for specialized evaluation
        task_type = self.classify_task_type(original_query, task_type_hint)
        
        # Get applicable criteria
        criteria = self._get_applicable_criteria(task_type, eval_level)
        
        # Evaluate each dimension
        dimension_scores = {}
        
        for criterion_name, criterion in criteria.items():
            # Evaluate this dimension
            score, _ = self._evaluate_dimension(
                original_query, 
                response_content,
                criterion,
                task_type
            )
            
            dimension_scores[criterion_name] = score
        
        # Use patched improvement suggestion generator for better error handling
        improvement_suggestions = self._generate_improvement_suggestions_patched(
            original_query,
            response_content,
            dimension_scores,
            criteria
        )
        
        # Calculate overall score based on weighted dimensions
        overall_score = self._calculate_overall_score(dimension_scores, criteria)
        
        # Calculate evaluation confidence
        confidence = self._calculate_confidence(dimension_scores)
        
        # Determine if response passes quality threshold
        passes_threshold = overall_score >= self.config.quality_thresholds["standard"]
        
        # Determine if regeneration is recommended
        regeneration_recommended = overall_score < (self.config.quality_thresholds["standard"] - 1.0)
        
        # Create evaluation result
        result = EvaluationResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            passes_threshold=passes_threshold,
            confidence=confidence,
            improvement_suggestions=improvement_suggestions,
            regeneration_recommended=regeneration_recommended,
            evaluation_time=time.time() - start_time,
            metadata={
                "task_type": task_type.value,
                "evaluation_level": eval_level.value,
                "timestamp": time.time(),
                "query_length": len(original_query),
                "response_length": len(response_content)
            }
        )
        
        # Cache the result
        self.evaluation_cache[cache_key] = result
        
        return result
    
    def improve_response(self, original_query: str, response: Any, evaluation: EvaluationResult) -> Optional[Dict[str, Any]]:
        """
        Attempt to improve a response based on evaluation results
        
        Args:
            original_query: The original user query
            response: The response to improve
            evaluation: The evaluation result
            
        Returns:
            Improved response or None if improvement failed
        """
        # Only attempt improvement if there are suggestions
        if not evaluation.improvement_suggestions:
            print("⚠️ No improvement suggestions available")
            return None
        
        # Standardize response format
        if isinstance(response, str):
            response_content = response
            response_dict = {"content": response}
        else:
            response_dict = response if isinstance(response, dict) else {"content": str(response)}
            response_content = response_dict.get("content", str(response_dict))
        
        # Create improvement prompt
        prompt = f"""
        Task: Improve the following response based on specific feedback.
        
        Original Query: "{original_query}"
        
        Original Response:
        ```
        {response_content}
        ```
        
        Improvement Suggestions:
        {chr(10).join([f"- {suggestion}" for suggestion in evaluation.improvement_suggestions])}
        
        Overall Score: {evaluation.overall_score:.1f}/10
        
        Instructions:
        1. Address all improvement suggestions while maintaining the original content's structure
        2. Focus specifically on areas scored below threshold
        3. Return the full improved response
        4. Maintain the same format as the original response
        5. Do not add explanations about the improvements made
        """
        
        try:
            print(f"🔧 Attempting to improve response with score {evaluation.overall_score:.1f}...")
            
            # Generate improved response
            improved_content = self.llm.complete(prompt)
            
            # Create improved response dictionary
            improved_response = response_dict.copy()
            improved_response["content"] = improved_content
            improved_response["improved"] = True
            
            return improved_response
            
        except Exception as e:
            print(f"❌ Error improving response: {str(e)}")
            return None
    
    def classify_task_type(self, query: str, task_type_hint: str = None) -> TaskType:
        """Classify the task type for specialized evaluation"""
        if task_type_hint == "question":
            return TaskType.QUESTION
        
        query_lower = query.lower()
        
        # Pattern-based classification
        if any(keyword in query_lower for keyword in ["code", "function", "program", "script", "implement"]):
            return TaskType.CODE_GENERATION
        
        if any(keyword in query_lower for keyword in ["research", "analyze", "investigate", "study", "examine"]):
            return TaskType.RESEARCH
        
        if any(keyword in query_lower for keyword in ["email", "gmail", "inbox", "message"]):
            return TaskType.EMAIL_MANAGEMENT
        
        if any(keyword in query_lower for keyword in ["calculate", "compute", "math", "equation", "formula"]):
            return TaskType.CALCULATION
        
        if any(keyword in query_lower for keyword in ["write", "create", "draft", "compose", "generate content"]):
            return TaskType.CONTENT_CREATION
        
        if any(keyword in query_lower for keyword in ["scrape", "extract", "fetch", "website", "web page"]):
            return TaskType.WEB_SCRAPING
        
        # Use LLM for more complex classification if pattern matching is inconclusive
        prompt = f"""
        Classify the following query into exactly one of these task types:
        - QUESTION (simple information request)
        - CODE_GENERATION (creating or modifying code)
        - CONTENT_CREATION (writing articles, emails, creative content)
        - RESEARCH (gathering and synthesizing information)
        - EMAIL_MANAGEMENT (email related tasks)
        - CALCULATION (mathematical operations)
        - WEB_SCRAPING (extracting web content)
        - GENERAL_TASK (other task types)

        Query: "{query}"
        
        Task Type:
        """
        
        try:
            result = self.llm.complete(prompt).strip().upper()
            
            # Try to match with known task types
            for task_type in TaskType:
                if task_type.name in result:
                    return task_type
            
            # Default to general task if no match
            return TaskType.GENERAL_TASK
            
        except Exception as e:
            print(f"⚠️ Error classifying task: {str(e)}")
            return TaskType.GENERAL_TASK
    
    def _determine_evaluation_level(self, query: str) -> EvaluationLevel:
        """Determine the appropriate evaluation level based on query"""
        query_lower = query.lower().strip()
        
        # Skip evaluation for very short queries (less than 20 characters)
        if len(query_lower) < 20:
            print(f"DEBUG - Skipping evaluation for short query: '{query_lower}' (length: {len(query_lower)})")
            return EvaluationLevel.SKIP
        
        # Check for skip patterns (simple greetings, thanks, etc.)
        for pattern in self.config.evaluation_rules["skip_patterns"]:
            if re.match(pattern, query_lower):
                print(f"DEBUG - Skipping evaluation based on pattern match: '{query_lower}'")
                return EvaluationLevel.SKIP
        
        # Check for basic patterns (simple definitions, etc.)
        for pattern in self.config.evaluation_rules["basic_patterns"]:
            if re.search(pattern, query_lower):
                return EvaluationLevel.BASIC
                
        # Check for comprehensive patterns (complex tasks)
        for pattern in self.config.evaluation_rules["comprehensive_patterns"]:
            if re.search(pattern, query_lower):
                return EvaluationLevel.COMPREHENSIVE
        
        # Default to standard evaluation
        return EvaluationLevel.STANDARD
    
    def _create_skip_evaluation(self, query: str, response: Dict[str, Any], start_time: float) -> EvaluationResult:
        """Create evaluation result for skipped evaluations"""
        return EvaluationResult(
            overall_score=10.0,
            dimension_scores={"skipped": 10.0},
            passes_threshold=True,
            confidence=1.0,
            improvement_suggestions=[],
            regeneration_recommended=False,
            evaluation_time=time.time() - start_time,
            metadata={
                "evaluation_level": EvaluationLevel.SKIP.value,
                "reason": "Simple query pattern detected",
                "timestamp": time.time()
            }
        )
    
    def _get_applicable_criteria(self, task_type: TaskType, eval_level: EvaluationLevel) -> Dict[str, EvaluationCriteria]:
        """Get applicable criteria based on task type and evaluation level"""
        # Start with base criteria
        criteria = self.base_criteria.copy()
        
        # For basic evaluation, use only base criteria
        if eval_level == EvaluationLevel.BASIC:
            return criteria
        
        # Add task-specific criteria for standard and comprehensive evaluation
        if task_type in self.task_specific_criteria:
            criteria.update(self.task_specific_criteria[task_type])
            
        return criteria
    
    def _evaluate_dimension(self, query: str, response: str, criterion: EvaluationCriteria, task_type: TaskType) -> Tuple[float, Optional[str]]:
        """Evaluate a specific dimension of the response with improved error handling"""
        
        # Simplified, more reliable prompt
        evaluation_prompt = f"""
        Rate this response for {criterion.name} on a scale of 1-10.
        
        Evaluation criteria: {criterion.prompt}
        
        Query: {query}
        Response: {response[:1500]}{"..." if len(response) > 1500 else ""}
        
        Rating scale:
        1-3: Poor (major issues)
        4-6: Average (meets basic needs)
        7-8: Good (exceeds expectations)
        9-10: Excellent (exceptional quality)
        
        Respond with ONLY a number from 1 to 10. No additional text.
        """
        
        try:
            result = self.llm.complete(evaluation_prompt).strip()
            
            # Robust number extraction
            import re
            
            # Try multiple parsing strategies
            patterns = [
                r'\b([1-9]|10)\b',           # Whole numbers 1-10
                r'\b([1-9]\.\d+|10\.0)\b',   # Decimals like 7.5, 10.0
                r'([1-9]|10)\s*/\s*10',      # Format like "8/10"
                r'score:?\s*([1-9]|10)',     # Format like "Score: 8"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 1.0 <= score <= 10.0:
                            return score, None
                    except (ValueError, IndexError):
                        continue
            
            # Fallback: look for any valid number
            numbers = re.findall(r'\d+(?:\.\d+)?', result)
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if 1.0 <= num <= 10.0:
                        return num, None
                except ValueError:
                    continue
            
            print(f"⚠️ Could not parse score from: '{result}'. Using default 5.0")
            return 5.0, f"Improve {criterion.name} - parsing failed"
            
        except Exception as e:
            print(f"❌ Error in evaluation: {str(e)}")
            return 5.0, f"Improve {criterion.name} - evaluation failed"
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float], criteria: Dict[str, EvaluationCriteria]) -> float:
        """Calculate overall weighted score from dimension scores"""
        if not dimension_scores:
            return 5.0  # Default score if no dimensions evaluated
            
        # Calculate weighted sum
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for dimension, score in dimension_scores.items():
            if dimension in criteria:
                weight = criteria[dimension].weight
                weighted_sum += score * weight
                weight_sum += weight
        
        # If no weights applied, use simple average
        if weight_sum == 0:
            return sum(dimension_scores.values()) / len(dimension_scores)
            
        # Return weighted average
        return weighted_sum / weight_sum
    
    def _calculate_confidence(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate confidence based on score consistency"""
        if not dimension_scores:
            return 0.0
        
        scores = list(dimension_scores.values())
        
        # Higher confidence when more dimensions are evaluated
        dimension_factor = min(1.0, len(scores) / 5.0)
        
        # Higher confidence when scores are consistent
        if len(scores) > 1:
            import statistics
            variance = statistics.variance(scores)
            consistency_factor = 1.0 - min(1.0, variance / 10.0)
        else:
            consistency_factor = 0.5  # Medium confidence for single dimension
        
        # Overall confidence
        confidence = (dimension_factor + consistency_factor) / 2.0
        
        return confidence
    
    def _create_cache_key(self, query: str, response_content: str) -> str:
        """Create a unique cache key for the query-response pair"""
        combined = f"{query}||{response_content}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _generate_improvement_suggestions_patched(self, query: str, response: str, scores: Dict[str, float], criteria) -> List[str]:
        """
        PATCHED VERSION: Generate improvement suggestions with better error handling
        """
        suggestions = []
        
        # Identify low-scoring dimensions
        problem_areas = []
        for criterion_name, score in scores.items():
            if criterion_name in criteria and score < criteria[criterion_name].threshold:
                problem_areas.append((criterion_name, score))
        
        if not problem_areas:
            return suggestions
        
        # Create simple, focused prompt
        problem_list = ", ".join([f"{area} (scored {score:.1f})" for area, score in problem_areas[:3]])
        
        improvement_prompt = f"""
        This response has low scores in: {problem_list}
        
        Query: {query}
        Response: {response[:800]}{"..." if len(response) > 800 else ""}
        
        Give 2-3 specific ways to improve this response.
        Make each suggestion one clear sentence.
        
        Improvements:"""
        
        try:
            result = self.llm.complete(improvement_prompt)
            
            # Simple parsing - split by lines and clean up
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering, bullets, etc.
                line = re.sub(r'^[\d\.\-\•\*\s]+', '', line)
                if len(line) > 20 and any(word in line.lower() for word in ['improve', 'add', 'include', 'ensure', 'provide', 'make']):
                    suggestions.append(line)
            
            # Fallback suggestions if parsing fails
            if not suggestions:
                suggestions = self._get_fallback_suggestions(problem_areas)
                
            return suggestions[:3]  # Limit to 3 suggestions
            
        except Exception as e:
            print(f"⚠️ Error generating suggestions: {str(e)}")
            return self._get_fallback_suggestions(problem_areas)
    
    def _get_fallback_suggestions(self, problem_areas: List[tuple]) -> List[str]:
        """
        Simple fallback suggestions when LLM fails
        """
        fallback_map = {
            "relevance": "Make the response more directly address the user's specific question",
            "completeness": "Add more comprehensive information to fully answer the query", 
            "accuracy": "Verify and improve the factual correctness of the information",
            "clarity": "Improve the structure and readability of the response",
            "syntax_correctness": "Fix any syntax errors in the generated code",
            "functionality": "Ensure the code properly solves the specified problem",
            "task_completion": "Complete all requested operations successfully",
            "source_quality": "Include more credible sources and references",
            "depth": "Provide more detailed analysis and examples"
        }
        
        suggestions = []
        for area, score in problem_areas[:3]:
            if area in fallback_map:
                suggestions.append(fallback_map[area])
        
        return suggestions if suggestions else ["Consider regenerating with more specific instructions"]
