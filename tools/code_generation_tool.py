"""
Code Generation Tool for Metis Agentic Orchestration System

This tool specializes in generating high-quality code across various programming languages,
frameworks, and patterns. It serves as a dedicated component for code generation tasks
within the Metis system.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("code_generation_tool")


class CodeGenerationTool:
    def __init__(self):
        try:
            from components.llm_interface import get_llm
            self.llm = get_llm()
        except ImportError:
            logger.warning("LLM interface not available, using mock interface")
            self.llm = self._create_mock_llm()

        self.supported_skills = ["code_generation"]
        logger.info("Code Generation Tool initialized")

    def _create_mock_llm(self):
        class MockLLM:
            def complete(self, prompt: str) -> str:
                return """```python
def hello_world():
    print("Hello, world!")
```"""
        return MockLLM()

    def can_handle(self, task: str) -> bool:
        if not task or not isinstance(task, str):
            return False
        task_lower = task.lower()
        keywords = ['code', 'script', 'function', 'app', 'api', 'generate', 'build']
        return any(word in task_lower for word in keywords)

    def score_for_context(self, context: dict) -> float:
        task = context.get("task", "").lower()
        if any(keyword in task for keyword in ['python', 'script', 'api', 'function']):
            return 0.9
        return 0.2

    def execute(self, task: str, original_query: str = None) -> str:
        logger.info(f"Executing task: {task[:100]}...")
        
        try:
            # Log if we have an original query to help with debugging
            if original_query:
                logger.info(f"Original query provided: {original_query[:100]}...")
            
            # Detect language and framework from the most specific request available
            primary_task = original_query if original_query else task
            language, framework = self._detect_language_and_framework(primary_task)
            
            # Run code generation with both task description and original query
            result = self.run(
                language=language,
                task_description=task,
                framework=framework,
                code_style="best_practice",
                include_comments=True,
                original_query=original_query
            )
            
            if not result:
                return "Error: No result returned from code generation."
                
            return self._format_result(result)
        except Exception as e:
            logger.exception(f"Unexpected error in execute(): {str(e)}")
            return f"I encountered an error while generating code: {str(e)}. Please try rephrasing your request or provide more specific details."

    def _detect_language_and_framework(self, task: str) -> tuple[str, Optional[str]]:
        task_lower = task.lower()
        lang_map = {
            "python": ["python", "django", "flask", "fastapi"],
            "javascript": ["javascript", "react", "vue", "node", "nextjs"],
            "typescript": ["typescript"],
        }
        fw_map = {
            "react": ["react"],
            "django": ["django"],
            "flask": ["flask"],
            "fastapi": ["fastapi"],
        }

        detected_lang = next((k for k, v in lang_map.items() if any(word in task_lower for word in v)), "python")
        detected_fw = next((k for k, v in fw_map.items() if any(word in task_lower for word in v)), None)
        return detected_lang, detected_fw

    def generate_code(self, language: str, task_description: str, framework: Optional[str] = None,
                      code_style: str = "best_practice", include_comments: bool = True) -> Dict[str, Any]:
        # Fix indentation and enhance validation
        if not task_description or not isinstance(task_description, str) or not task_description.strip():
            logger.error(f"Invalid task description received: '{task_description}'")
            return {
                "success": False,
                "error": "Task description is empty or invalid. Please provide a specific task description.",
                "code_blocks": [],
                "raw_completion": ""
            }

        logger.info(f"Generating {language} code for: {task_description[:100]}...")
        
        # Also validate language
        if not language or not isinstance(language, str) or not language.strip():
            logger.error(f"Invalid language received: '{language}'")
            return {
                "success": False, 
                "error": "Programming language is empty or invalid. Please specify a valid language.",
                "code_blocks": [],
                "raw_completion": ""
            }

        framework_part = f" using the {framework} framework" if framework else ""
        comment_note = "Include detailed comments and docstrings." if include_comments else "Keep comments minimal."
        
        # Clean up task description - look for clear keywords that indicate what to build
        task_lower = task_description.lower()
        specific_project = None
        
        # Check for specific project types
        if "snake game" in task_lower or "snake" in task_lower and "game" in task_lower:
            specific_project = "snake game"
        elif "tetris" in task_lower:
            specific_project = "Tetris game"
        elif "todo" in task_lower and ("app" in task_lower or "application" in task_lower):
            specific_project = "TODO app"
        
        # Create a very specific prompt with enhanced formatting instructions
        prompt = f"""You are a professional software engineer with expertise in {language}.
Your task is to write {language} code{framework_part} that EXACTLY implements the following request:

### TASK DESCRIPTION:
{task_description}

{f'### IMPORTANT: YOU MUST IMPLEMENT A {specific_project.upper()} as requested!' if specific_project else ''}

### REQUIREMENTS:
- Implement EXACTLY what was requested in the task description
- Follow {code_style} coding style and best practices for {language}
- {comment_note}
- The code must be complete, functional, and ready to run without requiring additional code
- Use modern {language} syntax and libraries
- Include proper error handling

### CODE FORMATTING REQUIREMENTS:
- ALWAYS use proper markdown code blocks with triple backticks AND language identifier
- Format your response exactly like this example:

```{language}
# Your code here
# More code
```

- DO NOT write code blocks without the proper language identifier
- DO NOT include 'python' or any other language name as plain text before code blocks
- DO NOT include setup instructions unless explicitly requested

### FOCUS ON RELEVANCE:
- If the user asks for a game, create THAT SPECIFIC GAME, not a data fetching application or generic template
- If the user asks for a specific application, create EXACTLY that application
- Do not substitute the requested implementation with something else

Begin your response with the properly formatted code block.
"""

        try:
            completion = self.llm.complete(prompt).strip()
            code_blocks = self.extract_code_blocks(completion)
            if not code_blocks:
                logger.warning("No code blocks detected, treating entire response as code")
                code_blocks = [{
                    "language": language,
                    "code": completion
                }]
            return {
                "success": True,
                "code_blocks": code_blocks,
                "raw_completion": completion,
                "metadata": {
                    "language": language,
                    "framework": framework,
                    "task_description": task_description
                }
            }
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "code_blocks": [],
                "raw_completion": ""
            }

    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown-formatted text, with special handling for common formatting errors."""
        # Handle standard markdown code blocks
        pattern = r"```([\w]*)\s*\n([\s\S]*?)```"
        matches = re.findall(pattern, text)
        code_blocks = []
        
        # Process standard code blocks found
        for lang, code in matches:
            # Clean up the language identifier
            lang = lang.strip().lower()
            if not lang:
                # Try to infer language from the first line if not specified
                first_line = code.strip().split('\n')[0].lower()
                if 'python' in first_line:
                    lang = 'python'
                elif 'javascript' in first_line:
                    lang = 'javascript'
                else:
                    lang = 'text'
                
            # Clean up the code content
            code = code.strip()
            
            # Add to our list if we have valid code
            if code:
                code_blocks.append({
                    "language": lang,
                    "code": code
                })
        
        # Special handling for common mistakes
        if not code_blocks:
            # Case 1: Text contains a line starting with 'python' or similar but no proper code block
            text_lines = text.strip().split('\n')
            potential_code_section = False
            code_lines = []
            language = 'text'
            
            for i, line in enumerate(text_lines):
                line_lower = line.lower().strip()
                
                # Check if this line indicates the start of a code section
                if not potential_code_section and (line_lower == 'python' or line_lower.startswith('# python')):
                    potential_code_section = True
                    language = 'python'
                    continue  # Skip the language identifier line
                
                # Collect code lines once we're in a code section
                if potential_code_section:
                    code_lines.append(line)
            
            # If we found a code section, add it
            if potential_code_section and code_lines:
                code_blocks.append({
                    "language": language,
                    "code": '\n'.join(code_lines).strip()
                })
        
        # Return what we found
        return code_blocks

    def _format_result(self, result: Optional[Dict[str, Any]]) -> str:
        if not result:
            return "Error: No result was returned."

        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error occurred')}"

        code_blocks = result.get("code_blocks", [])
        raw_output = result.get("raw_completion", "")

        if code_blocks:
            formatted_output = "Here’s the generated code:\n\n"
            for block in code_blocks:
                lang = block.get("language", "text")
                code = block.get("code", "").strip()
                if code:
                    formatted_output += f"```{lang}\n{code}\n```\n\n"

            explanations = re.split(r"```[\w]*\n[\s\S]*?\n```", raw_output)
            explanation = "\n".join(
                part.strip()
                for part in explanations
                if len(part.strip()) > 40 and "here’s a sample" not in part.lower()
            )
            if explanation:
                formatted_output += f"**Explanation:**\n{explanation}\n"

            return formatted_output.strip()

        return raw_output or "No code was generated. Please try rephrasing your request."

    def run(self, **kwargs) -> Dict[str, Any]:
        required = ["language", "task_description"]
        missing = [param for param in required if not kwargs.get(param)]
        if missing:
            return {
                "success": False,
                "error": f"Missing required parameters: {', '.join(missing)}",
                "code_blocks": [],
                "raw_completion": ""
            }
            
        # Critical: Check for original_query to prioritize specific user requests
        task_description = kwargs["task_description"]
        original_query = kwargs.get("original_query", "")
        
        if not task_description or task_description.strip() == "":
            logger.warning("Empty task description received")
            if original_query and len(original_query.strip()) > 0:
                logger.info(f"Using original query instead of empty task: {original_query[:100]}")
                task_description = original_query
            else:
                return {
                    "success": False,
                    "error": "Task description is empty and no original query provided.",
                    "code_blocks": [],
                    "raw_completion": ""
                }
        
        # Extract the real query if it's embedded in a template or has prefixes
        # This handles cases like "Highly relevant previous context: 1. create a snake game..."
        if original_query:
            # Common patterns where the actual query might be embedded
            patterns = [
                r"create a ([\w\s]+)", 
                r"make a ([\w\s]+)", 
                r"build a ([\w\s]+)",
                r"\d+\.\s*([^\n]+)"  # Numbered lists like "1. create a snake game"
            ]
            
            # Try to extract the core query from patterns
            for pattern in patterns:
                matches = re.findall(pattern, original_query.lower())
                if matches:
                    logger.info(f"Extracted specific request from original query: '{matches[0]}'")
                    # Don't replace the whole query, just note that we found a match
                    break
                    
            # If the original query contains specific programming terms while task doesn't
            programming_terms = ["python", "javascript", "java", "game", "function", "class", "app"]
            
            # Check if task is a generic instruction while original query is specific
            generic_instructions = [
                "provide instructions", "setup", "explain", "generate code", "write code",
                "setup", "create a", "implement", "build a", "develop"
            ]
            
            task_is_generic = any(instr in task_description.lower() for instr in generic_instructions)
            has_programming_terms = any(term in original_query.lower() for term in programming_terms)
            
            # Special handling for game-related requests - these often get misclassified
            game_keywords = ["game", "snake", "tetris", "pacman", "chess", "puzzle"]
            has_game_keywords = any(keyword in original_query.lower() for keyword in game_keywords)
            
            # Prioritization logic
            should_use_original = False
            
            # Case 1: Original query mentions games but task doesn't
            if has_game_keywords and not any(keyword in task_description.lower() for keyword in game_keywords):
                should_use_original = True
                logger.info("Original query contains game keywords not in task description")
                
            # Case 2: Task is generic but original query has programming details
            elif task_is_generic and has_programming_terms:
                should_use_original = True
                logger.info("Task is generic but original query contains programming terms")
                
            # Case 3: Original query is much more detailed than task description
            elif task_is_generic and len(original_query) > len(task_description) * 1.5:
                should_use_original = True
                logger.info("Original query is significantly more detailed than task description")
                
            if should_use_original:
                logger.info(f"Using original query: '{original_query[:100]}...' instead of task: '{task_description[:50]}...'")
                task_description = original_query
        
        # Now pass the potentially updated task_description to generate_code
        return self.generate_code(
            language=kwargs["language"],
            task_description=task_description,  # This might now be the original_query
            framework=kwargs.get("framework"),
            code_style=kwargs.get("code_style", "best_practice"),
            include_comments=kwargs.get("include_comments", True)
        )

    def get_description(self) -> str:
        return "Generates high-quality code in various programming languages and frameworks."

    def get_parameters(self) -> Dict[str, Dict]:
        return {
            "language": {"type": "string", "required": True},
            "framework": {"type": "string", "required": False},
            "task_description": {"type": "string", "required": True},
            "code_style": {"type": "string", "default": "best_practice"},
            "include_comments": {"type": "boolean", "default": True},
            "original_query": {"type": "string", "required": False, "description": "The original user query to prioritize over generic task descriptions"}
        }


# Run standalone test
if __name__ == "__main__":
    tool = CodeGenerationTool()
    test_task = "Create a Python function that returns the nth Fibonacci number."
    print(tool.execute(test_task))
