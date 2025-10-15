"""
Output Formatter for Single Agent Web Interface

This module provides content aggregation and formatting for the agent's output,
specifically designed to handle multi-tool workflows like research papers and creative writing.

It offers standardization of output format across different response types.
"""

import json
import re
import os
import sys
from typing import Dict, List, Any, Union, Optional, Tuple

# Add parent directory to path to import Single Agent modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now we can import from components
from components.llm_interface import get_llm

class OutputFormatter:
    """Formats agent output for web display using a standardized response format."""
    
    # Initialize LLM for task classification
    _llm = get_llm()
    
    @staticmethod
    def format_agent_response(agent_response: Any) -> Dict[str, Any]:
        """
        Format the agent's response using a standardized format.
        
        Standardized response format:
        {
            "type": "question_answer" | "task_result" | etc.,
            "content": "The actual content string, always in this field",
            "content_type": "markdown" | "code" | etc.,
            "task_type": "research" | "code" | "analysis" | etc.,
            "tools_used": [],
            "task_plan": [],
            "intent": "question" | "task"
        }
        
        Args:
            agent_response: The raw result from the agent
            
        Returns:
            A dictionary with formatted output ready for web display
        """
        print(f"DEBUG - Raw result type: {type(agent_response)}")
        result_preview = str(agent_response)[:100] + "..." if len(str(agent_response)) > 100 else str(agent_response)
        print(f"DEBUG - Raw result preview: {result_preview}")
        
        # Initialize the standardized response format
        formatted = {
            "type": "task_result",  # Default type
            "content": "",  # Will contain the main content string
            "content_type": "markdown",  # Default content type
            "task_type": "unknown",  # Task classification
            "tools_used": [],  # Tools used in processing
            "task_plan": [],  # Steps in the task plan
            "intent": "task"  # Default intent classification
        }
        
        # Handle clarification requests specifically
        if isinstance(agent_response, dict) and agent_response.get("type") == "clarification_request":
            formatted["type"] = "clarification_request"
            formatted["content"] = agent_response.get("content", agent_response.get("message", "I need more information to complete this task."))
            formatted["content_type"] = "clarification_request"
            formatted["clarification_options"] = agent_response.get("options", [])
            formatted["intent"] = "question"
            formatted["original_query"] = agent_response.get("original_query", "")
            return formatted
        
        # STANDARDIZED CONTENT EXTRACTION
        # This is the core of our simplified approach
        content = OutputFormatter._extract_content_standardized(agent_response)
        
        # Always set the content field
        formatted["content"] = content
        
        if content:
            print(f"DEBUG - Successfully extracted content: {len(content)} chars")
            print(f"DEBUG - Content begins with: {content[:100]}...")
        else:
            print("WARNING - No content was extracted from the agent response!")
            # Set a default message if no content was extracted
            formatted["content"] = "I've processed your request, but no specific content was generated."
        
        # Extract metadata fields from the response if available
        if isinstance(agent_response, dict):
            # Extract type field
            if "type" in agent_response:
                formatted["type"] = agent_response["type"]
            # For question_answer type, use 'question' intent
            if agent_response.get("type") == "question_answer":
                formatted["intent"] = "question"
            
            # Extract task_type if available
            if "task_type" in agent_response:
                formatted["task_type"] = agent_response["task_type"]
            
            # Extract tools used
            if "tools_used" in agent_response:
                formatted["tools_used"] = agent_response["tools_used"]
            
            # Extract task plan
            if "task_plan" in agent_response:
                formatted["task_plan"] = agent_response["task_plan"]
            
            # Extract intent
            if "intent" in agent_response:
                formatted["intent"] = agent_response["intent"]
            # Check for intent in metadata
            elif "metadata" in agent_response and isinstance(agent_response["metadata"], dict):
                if "intent" in agent_response["metadata"]:
                    formatted["intent"] = agent_response["metadata"]["intent"]
        
        # Determine the best task_type classification
        task_type = OutputFormatter._classify_task_type(agent_response, content)
        if task_type:
            formatted["task_type"] = task_type
        
        # Copy over any '_titans_memory' metadata
        if isinstance(agent_response, dict) and "_titans_memory" in agent_response:
            formatted["_titans_memory"] = agent_response["_titans_memory"]
        
        print(f"DEBUG - Final formatted response type: {formatted['type']}, intent: {formatted['intent']}")
        print(f"DEBUG - Final content length: {len(formatted['content'])}")
        
        return formatted
    
    @staticmethod
    def _extract_content_standardized(result: Any) -> str:
        """Extract content from the response in a standardized way.
        
        This is the core extraction method that handles all response types
        and returns a single string of content.
        """
        # Handle None/empty case
        if result is None:
            return ""
        
        # Handle string results directly
        if isinstance(result, str):
            return result
        
        # Handle dictionary results
        if isinstance(result, dict):
            # 1. First check for the standardized structure
            if "type" in result and result["type"] == "question_answer" and isinstance(result.get("data"), dict):
                data = result.get("data", {})
                
                # For question_answer type
                if "answer" in data and data["answer"]:
                    print(f"DEBUG - Found answer in data.answer: {data['answer'][:50]}...")
                    return data["answer"]
            
            # Handle task_result type with data
            if "type" in result and result["type"] == "task_result" and isinstance(result.get("data"), dict):
                data = result.get("data", {})
                
                # For task_result type - always look for content field first
                if "content" in data and data["content"]:
                    print(f"DEBUG - Found content in data.content: {data['content'][:50]}...")
                    return data["content"]
                
                # Check common result fields in data
                for field in ["result", "paper", "code", "analysis", "summary"]:
                    if field in data and data[field]:
                        print(f"DEBUG - Found content in data.{field}")
                        return data[field]
            
            # 2. Direct content fields at the top level
            if "content" in result and result["content"]:
                print(f"DEBUG - Found content at top level: {result['content'][:50]}...")
                return result["content"]
            
            # 3. Check other common content fields at top level
            for field in ["answer", "result", "response", "message", "output"]:
                if field in result and result[field]:
                    content = result[field]
                    if isinstance(content, str):
                        print(f"DEBUG - Found content in top-level {field}")
                        return content
                    elif isinstance(content, dict) and "content" in content:
                        print(f"DEBUG - Found content in {field}.content")
                        return content["content"]
            
            # 4. Special case for nested data structures
            if isinstance(result.get("data"), dict):
                data = result.get("data")
                for field in ["answer", "result", "content", "response", "message"]:
                    if field in data and data[field]:
                        print(f"DEBUG - Found content in nested data.{field}")
                        return data[field]
        
        # Fallback to empty string instead of serializing the whole object
        print(f"DEBUG - No content field found, keys available: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        return ""
    
    @classmethod
    def _classify_task_type(cls, result: Any, content: str) -> str:
        """Determine the type of task based on the response and content."""
        # First check for explicit task type in the response
        if isinstance(result, dict):
            # Check direct task_type field
            if "task_type" in result:
                return result["task_type"]
            
            # Check in metadata
            if isinstance(result.get("metadata"), dict) and "task_type" in result["metadata"]:
                return result["metadata"]["task_type"]
            
            # Check for tools used - if CodeGeneration is explicitly mentioned, prioritize this signal
            if "tools_used" in result and isinstance(result["tools_used"], list):
                tools = [t.lower() if isinstance(t, str) else str(t).lower() for t in result["tools_used"]]
                if any("code" in tool for tool in tools):
                    print(f"DEBUG - OutputFormatter: Detected code generation tool in tools_used")
                    return "code_generation"
        
        # If enough content to analyze, use LLM for classification
        sample_size = min(1000, len(content)) if content else 0
        if sample_size > 100:
            try:
                # Use chat-based approach like in Planner
                prompt = [
                    {"role": "system", "content": """
You are an AI Task Classifier for the Metis Agentic Orchestration System.

Your job is to classify content into ONE of these task types:
- code_generation: Content primarily contains code, programming examples, or code explanations
- research_paper: Academic or structured research with sections like abstract, introduction, methods
- creative_writing: Narrative content, stories, articles, essays without academic structure
- data_analysis: Statistical reports, data summaries, analytics results
- question_answering: Simple Q&A style responses to queries
- general: Default for content that doesn't clearly fit other categories

GUIDELINES:
1. Content with significant code blocks (especially with syntax highlighting markers) should be classified as code_generation
2. Game development, web development, and application development are all code_generation
3. Respond with ONLY the task type, no other text or explanation
4. If content has both code and explanations but is primarily focused on presenting functioning code, classify as code_generation
5. Python, JavaScript, HTML, CSS, Java and other programming language code should be classified as code_generation
6. Snake games, Tetris, web apps, etc. are all code_generation tasks
                    """}, 
                    {"role": "user", "content": f"Content sample: {content[:sample_size]}..."}
                ]
                
                # Get classification from LLM
                classification = cls._llm.chat_with_functions(prompt).strip().lower()
                print(f"DEBUG - OutputFormatter: LLM classification: {classification}")
                
                # Validate and use LLM classification if it's one of our known types
                valid_types = ["code_generation", "research_paper", "creative_writing", 
                              "data_analysis", "question_answering", "general"]
                
                for valid_type in valid_types:
                    if valid_type in classification:
                        return valid_type
                        
            except Exception as e:
                print(f"DEBUG - OutputFormatter: LLM classification error: {str(e)}")
                # Continue with rule-based classification if LLM fails
                
        # Rule-based fallback classification
        if content:
            # Research paper indicators
            if ("# Abstract" in content or "## Introduction" in content or
                "ABSTRACT" in content.upper() or "INTRODUCTION" in content.upper()):
                return "research_paper"
            
            # Code indicators - enhanced detection
            code_indicators = [
                "```python", "```javascript", "```java", "```c", "```cpp", "```html", "```css",
                "def ", "function ", "class ", "import ", "from ", "#include", "using namespace",
                "public static void", "const ", "var ", "let ", "game", "pygame"
            ]
            
            if any(indicator in content for indicator in code_indicators):
                return "code_generation"
            
            # Creative writing indicators
            if content.count("\n") > 10 and len(content) > 500 and "# " not in content:
                return "creative_writing"
        
        # Check for type-specific structures as last resort
        if isinstance(result, dict) and result.get("type") == "question_answer":
            return "question_answering"
            
        # Default
        return "general"


def format_response_for_frontend(data):
    """
    Format the agent's response for the frontend.
    
    Args:
        data: The data to format
        
    Returns:
        A dictionary with the formatted response
    """
    print(f"DEBUG - Using OutputFormatter to parse result")
    
    # Check if data is a string or other primitive
    if not isinstance(data, dict):
        data = {"content": str(data)}
    
    # Log the input data structure
    print(f"DEBUG - Input data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
    if isinstance(data, dict) and 'content' in data:
        print(f"DEBUG - Input data has content of length: {len(data['content'])}")
    
    # Use the OutputFormatter to standardize the response
    formatted = OutputFormatter.format_agent_response(data)
    
    # Recursively ensure all string values in the response are properly encoded
    def ensure_utf8(obj):
        if isinstance(obj, str):
            try:
                # Test if the content can be encoded as UTF-8
                obj.encode('utf-8')
                return obj
            except UnicodeEncodeError:
                # If encoding fails, replace problematic characters
                return obj.encode('utf-8', errors='replace').decode('utf-8')
        elif isinstance(obj, dict):
            return {k: ensure_utf8(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ensure_utf8(item) for item in obj]
        else:
            return obj
    
    # Apply encoding fix to the formatted result
    formatted = ensure_utf8(formatted)
    
    # Ensure content is present - for backward compatibility
    if not formatted.get("content") and data.get("content"):
        print("DEBUG - Using original content from data as formatted content was empty")
        formatted["content"] = data["content"]
    
    # For question_answer type responses from the agent
    if isinstance(data, dict) and data.get("type") == "question_answer" and isinstance(data.get("data"), dict):
        if data["data"].get("answer") and not formatted.get("content"):
            print("DEBUG - Using answer from data.data.answer as formatted content was empty")
            formatted["content"] = data["data"]["answer"]
            formatted["intent"] = "question"
    
    # Extract the key fields needed by the frontend
    result = {
        "content": formatted.get("content", ""),
        "content_type": formatted.get("content_type", "markdown"),
        "tools_used": formatted.get("tools_used", []),
        "task_plan": formatted.get("task_plan", []),
        "intent": formatted.get("intent", "task"),
        "type": formatted.get("type", "task_result"),
        "task_type": formatted.get("task_type", "general")
    }
    
    # Copy any special fields like _titans_memory
    if formatted.get("_titans_memory"):
        result["_titans_memory"] = formatted["_titans_memory"]
    
    # Copy clarification request fields if present
    if formatted.get("type") == "clarification_request":
        result["original_query"] = formatted.get("original_query", "")
        result["clarification_options"] = formatted.get("clarification_options", [])
    
    # Final check to ensure the content is properly encoded
    if "content" in result and isinstance(result["content"], str):
        try:
            result["content"].encode('utf-8')
        except UnicodeEncodeError:
            result["content"] = result["content"].encode('utf-8', errors='replace').decode('utf-8')
    
    # Debug the final result
    print(f"DEBUG - Final result keys: {result.keys()}")
    print(f"DEBUG - Final content length: {len(result.get('content', ''))}")
    if result.get('content'):
        print(f"DEBUG - Final content preview: {result['content'][:100]}...")
    else:
        print("WARNING - Final content is empty or None!")
            
    return result

