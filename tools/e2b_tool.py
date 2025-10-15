"""
E2B Code Execution Tool for Metis System

This tool provides a sandboxed environment for safely executing code
using the E2B platform.
"""

import os
import time
import logging
import re
from typing import Dict, Any, Optional, List
# Import dotenv for environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    # Create a simple implementation of load_dotenv if the package is not available
    def load_dotenv(dotenv_path=None, override=False):
        """Simple implementation of dotenv loading function"""
        if dotenv_path and os.path.exists(dotenv_path):
            try:
                with open(dotenv_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, value = line.split('=', 1)
                        value = value.strip('\'"')
                        if override or key not in os.environ:
                            os.environ[key] = value
            except Exception as e:
                logging.error(f"Error loading .env file: {e}")
        return True
from e2b_code_interpreter import Sandbox

# Initialize logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("e2b_tool")

# Load environment variables
load_dotenv()
load_dotenv('.env.local', override=True)

# Get API key - E2B expects E2B_API_KEY
E2B_API_KEY = os.getenv("E2B_API_KEY") or os.getenv("E2B_ACCESS_TOKEN")
if not E2B_API_KEY:
    logger.warning("E2B_API_KEY not found in environment variables. Code execution will not be available.")


class E2BTool:
    """
    Tool for executing code in a sandboxed environment using E2B.
    """
    
    def __init__(self, timeout: int = 300):
        """
        Initialize the E2B tool.
        
        :param timeout: Sandbox lifetime in seconds (default: 300)
        """
        self.timeout = timeout
        self.api_key = E2B_API_KEY
        self.is_available = self.api_key is not None
        
        # Check availability and log result
        if self.is_available:
            logger.info(f"E2B tool initialized with API key, sandbox timeout: {timeout}s")
            # Set API key to environment for E2B SDK
            os.environ["E2B_API_KEY"] = self.api_key
        else:
            logger.warning("E2B tool initialized but API key is not available")
        
    def get_description(self) -> str:
        """Return the tool description."""
        return "Executes code safely in a sandboxed environment"
    
    def get_parameters(self) -> Dict[str, Dict]:
        """Return parameter specifications."""
        return {
            "code": {
                "type": "string",
                "description": "Code to execute in the sandbox"
            },
            "language": {
                "type": "string",
                "description": "Programming language of the code",
                "enum": ["python", "javascript", "typescript", "shell"]
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds"
            }
        }
    
    def execute_code(self, code: str, language: str = "python", 
                   timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.
        
        :param code: Code to execute
        :param language: Programming language (default: python)
        :param timeout: Execution timeout in seconds (overrides default)
        :return: Execution result dictionary
        """
        if not self.is_available:
            logger.error("Cannot execute code: E2B API key not available")
            return {
                "success": False,
                "error": "E2B_ACCESS_TOKEN not set. Cannot execute code.",
                "stdout": "",
                "stderr": "",
                "text": ""
            }
            
        execution_timeout = timeout or self.timeout
        logger.info(f"Executing {language} code in E2B sandbox (timeout: {execution_timeout}s)")
        
        try:
            # Log the exact E2B API key being used (first 5 chars only for security)
            api_key_prefix = self.api_key[:5] + "..." if self.api_key else "None"
            logger.info(f"Using E2B API key: {api_key_prefix}")
            
            # Create sandbox with specified timeout and API key
            start_time = time.time()
            logger.info(f"Creating E2B sandbox with timeout: {execution_timeout}s")
            
            with Sandbox(api_key=self.api_key, timeout=execution_timeout) as sandbox:
                logger.info(f"E2B sandbox created successfully in {time.time() - start_time:.2f}s")
                
                # Execute code based on language
                if language.lower() == "python":
                    logger.info("Running Python code in sandbox")
                    execution = sandbox.run_code(code)
                elif language.lower() in ["javascript", "typescript"]:
                    logger.info("Running JavaScript/TypeScript code in sandbox")
                    execution = sandbox.run_javascript(code)
                elif language.lower() == "shell":
                    logger.info("Running shell commands in sandbox")
                    execution = sandbox.run_shell(code)
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported language: {language}",
                        "stdout": "",
                        "stderr": "",
                        "text": ""
                    }
                
                # Process execution result
                # Get all potential output attributes from the execution object
                attrs = dir(execution)
                
                # Check for error first
                error = getattr(execution, 'error', None) if 'error' in attrs else None
                
                # Get stdout from logs if available
                stdout = ""
                logs = getattr(execution, 'logs', None) if 'logs' in attrs else None
                if logs and hasattr(logs, 'stdout'):
                    stdout = '\n'.join(logs.stdout) if logs.stdout else ""
                
                # Get stderr from logs if available
                stderr = ""
                if logs and hasattr(logs, 'stderr'):
                    stderr = '\n'.join(logs.stderr) if logs.stderr else ""
                
                # Check for direct success attribute
                success = not error
                
                # Combine text output from various sources
                text = getattr(execution, 'text', '') if 'text' in attrs else ''
                
                logger.info(f"E2B execution completed: success={success}")
                if error:
                    logger.error(f"E2B execution error: {error}")
                
                return {
                    "success": success,
                    "stdout": stdout,
                    "stderr": stderr,
                    "text": text,
                    "logs": logs,
                    "error": error
                }
                
        except Exception as e:
            logger.error(f"E2B execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": f"Error: {str(e)}",
                "text": f"Error: {str(e)}"
            }
    
    def execute_multiple(self, code_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple code blocks sequentially in the same sandbox.
        
        :param code_blocks: List of code blocks with 'code', 'language', and optional 'timeout'
        :return: List of execution results
        """
        if not self.is_available:
            logger.error("Cannot execute code: E2B API key not available")
            return [{
                "success": False,
                "error": "E2B_ACCESS_TOKEN not set. Cannot execute code.",
                "stdout": "",
                "stderr": "",
                "text": ""
            } for _ in code_blocks]
        
        results = []
        try:
            # Create a single sandbox for all code blocks
            with Sandbox(api_key=self.api_key, timeout=self.timeout) as sandbox:
                for block in code_blocks:
                    code = block.get("code", "")
                    language = block.get("language", "python")
                    timeout = block.get("timeout", self.timeout)
                    
                    try:
                        # Execute code based on language
                        if language.lower() == "python":
                            execution = sandbox.run_code(code)
                        elif language.lower() in ["javascript", "typescript"]:
                            execution = sandbox.run_javascript(code)
                        elif language.lower() == "shell":
                            execution = sandbox.run_shell(code)
                        else:
                            results.append({
                                "success": False,
                                "error": f"Unsupported language: {language}",
                                "stdout": "",
                                "stderr": "",
                                "text": ""
                            })
                            continue
                        
                        # Process execution result
                        error = getattr(execution, 'error', None)
                        success = not error
                        
                        # Get stdout from logs if available
                        stdout = ""
                        logs = getattr(execution, 'logs', None)
                        if logs and hasattr(logs, 'stdout'):
                            stdout = '\n'.join(logs.stdout) if logs.stdout else ""
                        
                        # Get stderr from logs if available
                        stderr = ""
                        if logs and hasattr(logs, 'stderr'):
                            stderr = '\n'.join(logs.stderr) if logs.stderr else ""
                        
                        results.append({
                            "success": success,
                            "stdout": stdout,
                            "stderr": stderr,
                            "text": getattr(execution, 'text', ''),
                            "error": error
                        })
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "stdout": "",
                            "stderr": f"Error: {str(e)}",
                            "text": f"Error: {str(e)}"
                        })
        except Exception as e:
            logger.error(f"E2B sandbox creation error: {str(e)}")
            # Return error for all blocks
            results = [{
                "success": False,
                "error": f"Sandbox creation failed: {str(e)}",
                "stdout": "",
                "stderr": f"Error: {str(e)}",
                "text": f"Error: {str(e)}"
            } for _ in code_blocks]
        
        return results
    
    def can_handle(self, task: str) -> bool:
        """
        Determine if this tool can handle code execution tasks.
        """
        # Check for code execution related keywords
        code_keywords = [
            'execute code', 'run code', 'execute script', 'run script',
            'code sandbox', 'test code', 'evaluate code', 'python code',
            'javascript code', 'typescript code', 'shell command', 'execute',
            'run this', 'try this code', 'execute following'
        ]
        
        task_lower = task.lower()
        
        # Check for keywords
        if any(keyword in task_lower for keyword in code_keywords):
            return True
            
        # Look for code blocks indicators
        code_patterns = [
            r'```(?:python|javascript|typescript|js|ts|shell|bash).*?```',
            r'```.*?```',
            r'def\s+\w+\s*\(.*?\):',
            r'function\s+\w+\s*\(.*?\)\s*{',
            r'const\s+\w+\s*=\s*function\s*\(.*?\)\s*{',
            r'import\s+[\w\s,{}]+\s+from\s+[\'"]',
            r'print\s*\(',
            r'console\.log\s*\('
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, task, re.DOTALL):
                return True
        
        return False
        
    def execute(self, task: str) -> str:
        """
        Execute code based on the task description.
        """
        # Extract code and language from the task
        code_info = self._extract_code(task)
        
        if not code_info or not code_info[0]:
            return "No valid code found to execute. Please provide code in a code block format or clearly indicate what code to run."
        
        code, language = code_info
        
        # Map language to appropriate E2B runtime
        runtime = "python3" if language in ["python", "py"] else language
        if runtime not in ["python3", "javascript", "typescript", "bash", "shell"]:
            runtime = "python3"  # Default to Python if unsure
            
        # Start time for tracking execution duration
        start_time = time.time()
        
        try:
            # Initialize the sandbox if not already done
            if not self.sandbox or not self.is_active():
                self.initialize()
                
            # Execute the code
            result = ""
            if runtime in ["bash", "shell"]:
                response = self.execute_shell(code)
                result = response.get('stdout', '') + response.get('stderr', '')
            else:
                response = self.execute_code(code, runtime)
                result = response.get('output', '')
                
            # Format the output
            execution_time = time.time() - start_time
            output = f"Executed code in {execution_time:.2f} seconds.\n\n"
            output += f"Result:\n```\n{result}\n```"
            
            return output
            
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def _extract_code(self, task: str) -> Optional[tuple]:
        """
        Extract code and language from the task description.
        """
        # Look for markdown code blocks with language specification
        code_block_pattern = r'```(\w*)\n([\s\S]*?)\n```'
        matches = re.findall(code_block_pattern, task)
        
        if matches:
            language, code = matches[0]
            language = language.lower().strip()
            
            # Default to python if language not specified
            if not language or language not in ["python", "py", "javascript", "js", "typescript", "ts", "bash", "shell"]:
                language = "python"
                
            return code.strip(), language
            
        # Check for inline code patterns if no code blocks found
        code_patterns = [
            (r'def\s+\w+\s*\(.*?\):[\s\S]*?(?=\n\n|$)', "python"),
            (r'function\s+\w+\s*\(.*?\)\s*{[\s\S]*?}', "javascript"),
            (r'const\s+\w+\s*=\s*function\s*\(.*?\)\s*{[\s\S]*?}', "javascript"),
            (r'print\s*\([\s\S]*?\)', "python"),
            (r'console\.log\s*\([\s\S]*?\)', "javascript")
        ]
        
        for pattern, lang in code_patterns:
            match = re.search(pattern, task, re.DOTALL)
            if match:
                return match.group(0).strip(), lang
                
        # If no code found, return None
        return None
        
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Call the tool with parameters.
        
        :param kwargs: Parameters for code execution
        :return: Execution result
        """
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")
        timeout = kwargs.get("timeout", self.timeout)
        
        return self.execute_code(code, language, timeout)
    

