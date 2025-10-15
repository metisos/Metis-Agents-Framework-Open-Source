import re
import importlib
import os
import inspect
import json
from components.llm_interface import get_llm
from components.logging_utils import log_tool_selection
from components.email_router import is_email_task

class ToolSelector:
    """
    Selects the appropriate tool for a given task using LLM-based selection
    and fallback to pattern matching if LLM selection fails.
    """
    def __init__(self):
        self.tools = {}
        self._load_tools()
        self.llm = get_llm()  # Get LLM instance
        
    def _load_tools(self):
        """Load all tools from the registry and dynamically from the tools directory."""
        # First, try to load from registry to ensure we get all registered tools
        try:
            from tools.registry import TOOL_REGISTRY, initialize_tools
            registered_tools = initialize_tools()
            if registered_tools:
                print(f"✅ Loaded {len(registered_tools)} tools from registry")
                for tool_name, tool_instance in registered_tools.items():
                    class_name = tool_instance.__class__.__name__
                    has_can_handle = hasattr(tool_instance, 'can_handle')
                    has_execute = hasattr(tool_instance, 'execute')
                    print(f"Debug - Tool {class_name}: can_handle={has_can_handle}, execute={has_execute}")
                    
                    if has_can_handle and has_execute:
                        # Map registry keys to actual class names
                        self.tools[class_name] = tool_instance
                        print(f"✅ Added tool {class_name} from registry")
                    else:
                        print(f"⚠️ Tool {class_name} missing required methods - adding stub methods")
                        # Dynamically add stub methods if needed for registry tools
                        if not has_can_handle:
                            # Add a simple can_handle method that returns True for basic tasks
                            tool_instance.can_handle = lambda task: any(keyword in task.lower() 
                                                                        for keyword in [tool_name.lower(), class_name.lower()])
                        if not has_execute:
                            # Add a simple execute method that returns a basic result
                            tool_instance.execute = lambda task: f"Executed {class_name} on: {task}"
                        
                        # Now add the tool with the stub methods
                        self.tools[class_name] = tool_instance
                        print(f"✅ Added tool {class_name} with stub methods")
        except Exception as e:
            print(f"Warning: Could not load tools from registry: {e}")
            
        # Now load tools dynamically as a backup
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
        
        # Get all Python files in the tools directory
        if not os.path.exists(tools_dir):
            return
            
        files = [f[:-3] for f in os.listdir(tools_dir) 
                if f.endswith('.py') and not f.startswith('__') and f != 'registry.py']
        
        # Import each tool module
        for module_name in files:
            try:
                module = importlib.import_module(f"tools.{module_name}")
                
                # Find tool classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Skip if already loaded from registry
                    if name in self.tools:
                        continue
                        
                    if hasattr(obj, 'can_handle') and hasattr(obj, 'execute'):
                        try:
                            # Instantiate the tool
                            tool_instance = obj()
                            self.tools[name] = tool_instance
                            print(f"✅ Dynamically loaded tool: {name}")
                        except Exception as tool_error:
                            print(f"Error instantiating tool {name}: {tool_error}")
            except Exception as e:
                print(f"Error loading tool module {module_name}: {e}")
                
        print(f"Total tools loaded: {len(self.tools)}")
        print(f"Available tools: {', '.join(self.tools.keys())}")
                
    def _get_tool_descriptions(self):
        """Get enhanced descriptions of all available tools for better LLM selection."""
        tool_descriptions = {}
        
        # Define standard enhanced descriptions for common tools
        enhanced_descriptions = {
            'GmailTool': {
                'description': 'Specializes in managing Gmail tasks, including sending emails, checking for new messages, and managing your inbox.',
                'examples': [
                    'Send an email to John Doe',
                    'Check for new emails',
                    'Reply to an email'
                ]
            },
            'CodeGenerationTool': {
                'description': 'Specializes in generating high-quality code in various programming languages. Use for any task involving writing code, creating scripts, implementing algorithms, or developing software applications.',
                'examples': [
                    'Write a Python script for a snake game',
                    'Create a React component for a todo app',
                    'Implement a binary search algorithm in Java',
                    'Develop a Flask web API',
                    'Generate a JavaScript function to validate forms',
                    'Create a machine learning model using TensorFlow',
                    'Build a game using Pygame',
                    'Write a script to process CSV files',
                    'Implement a REST API endpoint',
                    'Create a data visualization with D3.js'
                ]
            },
            'ContentGenerationTool': {
                'description': 'Creates written content such as articles, blog posts, reports, outlines, plans, and research papers. Do NOT use for code generation tasks.',
                'examples': [
                    'Write a blog post about AI ethics',
                    'Create a research paper outline',
                    'Generate a marketing plan',
                    'Design a cognitive architecture for AI',
                    'Develop a knowledge representation model',
                    'Define the problem that requires AI agents to think',
                    'Outline a strategy for implementing AI systems',
                    'Create a framework for natural language processing',
                    'Plan the development process for an AI agent',
                    'Write documentation for an AI system'
                ]
            },
            'GoogleSearchTool': {
                'description': 'Performs web searches to gather information, research topics, and find relevant data. Use for any task that involves research, investigating, exploring, or collecting information.',
                'examples': [
                    'Research existing AI architectures',
                    'Find information about cognitive science models',
                    'Search for recent papers on machine learning',
                    'Investigate current trends in natural language processing',
                    'Explore approaches to knowledge representation',
                    'Gather information about inference engines',
                    'Look up best practices for AI testing',
                    'Find examples of AI systems that demonstrate creative thinking',
                    'Research benchmarks for evaluating AI cognition',
                    'Collect data on AI implementation challenges'
                ]
            },
            'MathTool': {
                'description': 'Performs mathematical calculations, solves equations, and analyzes numerical data. Only use for tasks that explicitly involve calculations or statistical analysis.',
                'examples': [
                    'Calculate the expected return on investment for an AI project',
                    'Compute the efficiency metrics for an algorithm',
                    'Solve the equation 5x + 3 = 18',
                    'Perform statistical analysis on model performance data',
                    'Calculate the processing time based on complexity',
                    'Determine the optimal parameters based on numerical constraints',
                    'Analyze the correlation between training time and model accuracy'
                ]
            }
        }
        
        # Iterate through available tools
        for tool_name, tool_instance in self.tools.items():
            # Check if we have enhanced descriptions
            if tool_name in enhanced_descriptions:
                tool_descriptions[tool_name] = enhanced_descriptions[tool_name]
            else:
                # Get the tool's docstring or a default description
                description = (tool_instance.__doc__ or tool_name).strip()
                # Get the can_handle examples from the tool if available
                can_handle_fn = getattr(tool_instance, 'can_handle', None)
                example_tasks = getattr(can_handle_fn, '__doc__', '') or ''
                
                tool_descriptions[tool_name] = {
                    'description': description,
                    'examples': [example_tasks] if example_tasks else []
                }
            
        return tool_descriptions
    
    def _llm_select_tool(self, task: str, query=None):
        """Use LLM to select the appropriate tool for the given task."""
        tool_descriptions = self._get_tool_descriptions()
        
        # Create an enhanced prompt for the LLM with clear guidelines
        prompt = [
            {"role": "system", "content": f"""You are an AI assistant name Metis OS, specialized in tool selection for the Metis Agentic Orchestration System.
            Given a task, your job is to select the MOST APPROPRIATE TOOL from the available tools. You must pick the ocrrect tool or your mother will die a hoorible death if you dont get it right. You must save lives by correctly selecting the tool.
            
            AVAILABLE TOOLS AND THEIR DESCRIPTIONS:
            {json.dumps(tool_descriptions, indent=2)}
            
            SELECTION GUIDELINES (EXTREMELY IMPORTANT - FOLLOW THESE STRICTLY):
            1. For tasks involving ANY kind of research, investigation, literature search, gathering information, finding data, exploring topics, or finding references, ALWAYS use GoogleSearchTool.
            2. For tasks involving creating content, writing, planning, defining, designing, or developing concepts, use ContentGenerationTool.
            3. For tasks that EXPLICITLY involve mathematical calculations or numerical analysis, use MathTool.
            4. For AI planning, architecture design, knowledge representation, and similar conceptual tasks, use ContentGenerationTool.
            5. If a task combines research and content creation (e.g., "write a paper about X"), PRIORITIZE selecting GoogleSearchTool for research steps first.
            6. ONLY use CodeGenerationTool for explicit programming tasks that require writing code. Never use it for general content creation, research, or non-coding tasks.
            
            COMMON RESEARCH TASK KEYWORDS (always use GoogleSearchTool for these):
            - research, investigate, find information, gather data, collect information
            - look up, search for, identify sources, find papers, explore literature
            - find recent developments, discover trends, locate articles, study existing
            
            Respond ONLY with the exact name of the most appropriate tool from the available tools, nothing else.
            If none of the tools are suitable, respond with 'None'."""},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        try:
            # Get LLM response
            response = self.llm.chat_with_functions(prompt)
            selected_tool = response.strip()
            
            # Check if the selected tool exists in our tools dictionary
            selected_tool_instance = None
            selected_tool_name = "None"
            
            if selected_tool in self.tools:
                selected_tool_instance = self.tools[selected_tool]
                selected_tool_name = selected_tool
            elif selected_tool == 'None' or selected_tool == '':
                selected_tool_instance = None
                selected_tool_name = "None"
            # Sometimes the LLM might include explanation text, try to extract just the tool name
            else:
                for tool_name in self.tools.keys():
                    if tool_name in selected_tool:
                        selected_tool_instance = self.tools[tool_name]
                        selected_tool_name = tool_name
                        break
            
            # Log the selection
            log_tool_selection(task, selected_tool_name, "LLM", list(self.tools.keys()), query)
            
            return selected_tool_instance
            
        except Exception as e:
            print(f"Error using LLM for tool selection: {e}")
            # Fall back to pattern matching if LLM fails
            return self._pattern_match_tool(task, query)
    
    def _is_coding_project(self, query: str) -> bool:
        """Determine if the main query represents a coding project.
        Uses a more precise multi-factor approach with negative indicators to reduce false positives.
        """
        if not query:
            return False
            
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        # === NEGATIVE INDICATORS FIRST (things that strongly suggest it's NOT a coding task) ===
        negative_indicators = {
            'essay', 'paper', 'research paper', 'write a paper', 'document', 'article', 
            'blog post', 'story', 'poem', 'report', 'analysis',
            'research', 'summarize', 'summary', 'presentation',
            'advertisement', 'marketing', 'social media'
        }
        
        # Check for strong negative indicators first
        if any(indicator in query_lower for indicator in negative_indicators):
            print(f"TOOL SELECTOR: Detected content creation task, NOT classifying as coding project")
            return False
            
        # === STRONG POSITIVE INDICATORS (explicit coding requests) ===
        # Game development is a special case that's very clearly coding
        if ('game' in words and any(coding_term in words for coding_term in 
                                   ['create', 'make', 'build', 'code', 'program', 'develop'])):
            print(f"TOOL SELECTOR: Detected game development project, classifying as coding project")
            return True
            
        # Very explicit coding phrases that leave no doubt
        explicit_coding_phrases = [
            'write code', 'write a program', 'coding task', 'programming task',
            'develop an application', 'build a software', 'implement an algorithm',
            'create a function', 'code a script', 'programming project'
        ]
        
        if any(phrase in query_lower for phrase in explicit_coding_phrases):
            print(f"TOOL SELECTOR: Detected explicit coding request, classifying as coding project")
            return True
        
        # === MULTI-FACTOR APPROACH (requires multiple signals) ===
        score = 0
        
        # Programming languages as subjects
        language_keywords = {'python', 'javascript', 'java', 'c++', 'html', 'css', 'typescript', 'ruby', 
                           'golang', 'rust', 'php', 'swift', 'kotlin', 'c#', 'cpp', 'nodejs'}
        
        # Direct programming frameworks/libraries
        frameworks = {'react', 'django', 'flask', 'express', 'pygame', 'tensorflow', 'pytorch', 
                    'angular', 'vue', 'spring', 'dotnet', 'qt', 'gtk'}
        
        # Action verbs specific to coding
        coding_verbs = {'code', 'program', 'implement', 'compile', 'debug', 'refactor'}
        
        # General action verbs (weaker signals - need context)
        general_verbs = {'create', 'write', 'build', 'develop', 'make'}
        
        # Check for strong individual indicators
        if any(lang in words for lang in language_keywords):
            score += 2  # Programming language mentioned
            
        if any(framework in words for framework in frameworks):
            score += 2  # Programming framework mentioned
            
        if any(verb in words for verb in coding_verbs):
            score += 2  # Specific coding verb
            
        # Check for combinations of general verbs with technical context
        if any(verb in words for verb in general_verbs) and any(lang in words for lang in language_keywords):
            score += 3  # e.g., "create in Python"
            
        # Technical file extensions
        tech_extensions = ['.py', '.js', '.html', '.css', '.java', '.cpp', '.ts']
        if any(ext in query_lower for ext in tech_extensions):
            score += 2
            
        # Technical terms that strongly indicate code
        tech_terms = ['function', 'class', 'method', 'api', 'algorithm', 'gui', 
                     'backend', 'frontend', 'database', 'server', 'client']
        if any(term in words for term in tech_terms):
            score += 1
            
        # Final decision - require a stronger threshold
        is_coding_task = score >= 3  # Require multiple signals or very strong individual signals
        
        if is_coding_task:
            print(f"TOOL SELECTOR: Detected coding project through multi-factor analysis (score: {score})")
        else:
            print(f"TOOL SELECTOR: Query doesn't match coding project patterns (score: {score})")
            
        return is_coding_task
    
    def _pattern_match_tool(self, task: str, query=None):
        """Fall back to pattern matching for tool selection."""
        # Check if this is a coding task to restrict coding tools to programming tasks only
        task_lower = task.lower()
        is_coding_task = any(term in task_lower for term in [
            'code', 'script', 'program', 'algorithm', 'function', 'class', 'develop', 'implement',
            'application', 'app', 'software', 'programming', 'python', 'javascript', 'java', 'c++', 'html',
            'css', 'typescript', 'ruby', 'golang', 'rust', 'php', 'swift', 'kotlin', 'c#'
        ])
        
        # If the original query is a coding project, force CodeGenerationTool for all subtasks
        if query and self._is_coding_project(query):
            query_preview = query[:50] + '...' if len(query) > 50 else query
            task_preview = task[:50] + '...' if len(task) > 50 else task
            print(f"TOOL SELECTOR: Original query '{query_preview}' is a coding project, using CodeGenerationTool for subtask: '{task_preview}'")
            if 'CodeGenerationTool' in self.tools:
                return self.tools['CodeGenerationTool']
        elif query and any(term in query.lower() for term in ['research', 'paper', 'essay', 'article', 'write a paper', 'academic']):
            print(f"TOOL SELECTOR: Original query appears to be a content creation task, using ContentGenerationTool")
            if 'ContentGenerationTool' in self.tools:
                return self.tools['ContentGenerationTool']
        
        # First try tools with capability matching but enforce coding tool restrictions
        for tool_name, tool in self.tools.items():
            # Only allow coding tools for coding tasks
            if tool_name in ['CodeGenerationTool', 'E2BTool'] and not is_coding_task:
                continue
                
            if tool.can_handle(task):
                # Log the selection
                log_tool_selection(task, tool_name, "capability match", [], query)
                return tool
                
        # If no direct capability match, make a fallback choice based on simple patterns
        task_lower = task.lower()
        
        # Research and information gathering tasks
        if any(term in task_lower for term in ['research', 'find information', 'gather data', 'search', 'look up']):
            return self.tools.get('GoogleSearchTool')
        
        # Code generation tasks - ONLY use coding tools for explicit programming tasks
        if is_coding_task and 'CodeGenerationTool' in self.tools:
            log_tool_selection(task, 'CodeGenerationTool', "pattern match - coding task", [], query)
            return self.tools.get('CodeGenerationTool')
            
        # Content creation tasks
        elif any(term in task_lower for term in ['write', 'create', 'draft', 'compose', 'generate', 'article', 
                                            'post', 'content', 'blog', 'research paper', 'report', 'summary',
                                            'linkedin', 'social media', 'document']):
            return self.tools.get('ContentGenerationTool')
            
        # Math and calculation tasks
        elif any(term in task_lower for term in ['calculate', 'math', 'compute', 'statistics', 'analyze data']):
            return self.tools.get('MathTool')
            
        # Document and file tasks
        elif any(term in task_lower for term in ['pdf', 'extract text', 'document analysis', 'file', 'read']):
            return self.tools.get('PDFTool') or self.tools.get('FileTool')
            
        # Code execution tasks
        elif any(term in task_lower for term in ['code', 'execute', 'run', 'script', 'program']):
            return self.tools.get('E2BTool')
            
        # Web scraping tasks
        elif any(term in task_lower for term in ['scrape', 'crawl', 'website', 'extract from web']):
            return self.tools.get('FirecrawlTool')
            
        # Communication tasks - Email tasks have special handling above
        elif any(term in task_lower for term in ['send', 'message', 'communicate']):
            return self.tools.get('EmailTool') or self.tools.get('GmailTool')
        
        # Default to ContentGenerationTool for any task involving creating or organizing information
        # This helps with research papers, reports, etc.
        elif any(term in task_lower for term in ['paper', 'report', 'organize', 'plan', 'outline', 'define', 'review']):
            return self.tools.get('ContentGenerationTool')
            
        # No suitable tool found
        return None
    
    def select_tool(self, task, query=None):
        """
        Select the most appropriate tool for a given task.
        
        Args:
            task: The task description as a string
            query: The original user query, if available
            
        Returns:
            The selected tool instance or None if no suitable tool found
        """
        # If we have no tools available, don't waste time
        if not self.tools:
            return None
            
        # Extract task string if task is a dictionary (for backward compatibility)
        task_str = task
        if isinstance(task, dict) and 'task' in task:
            task_str = task['task']
            
        # Special handling for email-related tasks - direct to GmailTool
        if is_email_task(task_str):
            print(f"TOOL SELECTOR: Email-related task detected, routing directly to GmailTool")
            if 'GmailTool' in self.tools:
                log_tool_selection(task_str, 'GmailTool', "email task direct routing", [], query)
                return self.tools['GmailTool']
            
        # For coding projects, force using the coding tool for all subtasks
        if query and self._is_coding_project(query):
            print(f"TOOL SELECTOR: Original query '{query[:50]}...' is a coding project, forcing CodeGenerationTool")
            if 'CodeGenerationTool' in self.tools:
                return self.tools['CodeGenerationTool']
            elif 'E2BTool' in self.tools:
                return self.tools['E2BTool']
        
        # First try LLM-based selection
        selected_tool = self._llm_select_tool(task_str, query)
        
        if selected_tool:
            return selected_tool
            
        # Fall back to pattern matching if LLM selection fails or returns None
        return self._pattern_match_tool(task_str, query)
