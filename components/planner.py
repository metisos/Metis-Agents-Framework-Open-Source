from typing import Dict, List, Tuple, Optional, Any
import re
import json
from components.llm_interface import get_llm

class Planner:
    """
    Creates a plan of subtasks for a given complex task.
    Similar to the Planner in Metis but simplified to output a linear list.
    """
    def __init__(self, task_file):
        self.task_file = task_file
        self.llm = get_llm()  # Get LLM instance in the same format as ToolSelector
        
    def is_coding_task(self, task_description: str) -> bool:
        """
        Determine if the task is primarily a coding task that should be handled as a single unit.
        Uses a more precise multi-factor approach to avoid false positives.
        
        Args:
            task_description: The task description to check
            
        Returns:
            Boolean indicating if this is a coding task
        """
        import re
        
        if not task_description:
            return False
            
        task_lower = task_description.lower()
        words = set(re.findall(r'\b\w+\b', task_lower))
        
        # === NEGATIVE INDICATORS (things that suggest it's NOT a coding task) ===
        negative_indicators = {
            'essay', 'paper', 'write a paper', 'document', 'article', 
            'blog post', 'story', 'poem', 'report', 'analysis',
            'research', 'summarize', 'summary', 'presentation',
            'advertisement', 'marketing', 'social media'
        }
        
        # Check for strong negative indicators first
        if any(indicator in task_lower for indicator in negative_indicators):
            print(f"PLANNER: Detected content creation task, NOT classifying as coding task")
            return False
            
        # === STRONG POSITIVE INDICATORS (explicit coding requests) ===
        # Game development is a special case that's very clearly coding
        if ('game' in words and any(coding_term in words for coding_term in 
                                  ['create', 'make', 'build', 'code', 'program', 'develop'])):
            print(f"PLANNER: Detected game development project, classifying as coding task")
            return True
            
        # Very explicit coding phrases that leave no doubt
        explicit_coding_phrases = [
            'write code', 'write a program', 'coding task', 'programming task',
            'develop an application', 'build a software', 'implement an algorithm',
            'create a function', 'code a script', 'programming project'
        ]
        
        if any(phrase in task_lower for phrase in explicit_coding_phrases):
            print(f"PLANNER: Detected explicit coding request, classifying as coding task")
            return True
        
        # === MULTI-FACTOR APPROACH (requires multiple signals) ===
        score = 0
        
        # Programming languages as subjects
        language_keywords = {'python', 'javascript', 'java', 'c++', 'html', 'css', 'typescript', 'ruby', 
                           'golang', 'rust', 'php', 'swift', 'kotlin', 'c#', 'cpp', 'nodejs'}
        
        # Direct programming frameworks/libraries
        frameworks = {'react', 'django', 'flask', 'express', 'pygame', 'tensorflow', 'pytorch', 
                    'angular', 'vue', 'spring', 'dotnet', 'qt', 'gtk'}
        
        # Action verbs for coding
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
        if any(ext in task_lower for ext in tech_extensions):
            score += 2
            
        # Technical terms that strongly indicate code
        tech_terms = ['function', 'class', 'method', 'api', 'algorithm', 'gui', 
                     'backend', 'frontend', 'database', 'server', 'client']
        if any(term in words for term in tech_terms):
            score += 1
            
        # Final decision - require a stronger threshold
        is_coding = score >= 3  # Require multiple signals or very strong individual signals
        
        if is_coding:
            print(f"PLANNER: Detected coding task through multi-factor analysis (score: {score})")
            
        return is_coding
    
    def is_email_task(self, task_description: str) -> bool:
        """
        Determine if the task is primarily an email-related task that should be handled differently.
        
        Args:
            task_description: The task description to check
            
        Returns:
            Boolean indicating if this is an email task
        """
        task_lower = task_description.lower()
        email_keywords = [
            'email', 'gmail', 'read email', 'check email', 'my inbox', 'mail', 'emails',
            'read my email', 'read my emails', 'summarize email', 'summarize emails',
            'email summary', 'inbox summary', 'check my inbox', 'read inbox'
        ]
        
        return any(keyword in task_lower for keyword in email_keywords)
    
    def create_plan(self, task_description: str) -> List[str]:
        """
        Break down a complex task into a series of subtasks.
        For coding tasks, use a simplified approach to avoid fragmenting code.
        For email tasks, use direct routing to the email tool.
        
        Args:
            task_description: The task to be broken down
            
        Returns:
            List of subtask strings
        """
        # Check if this is a coding task that should be handled differently
        if self.is_coding_task(task_description):
            print(f"PLANNER: Detected coding task, using single-task plan")
            
            # Single task plan for coding tasks to preserve context
            # This prevents the second generic task from losing the original request context
            return [task_description]
            
            # Previous approach that caused context loss:
            # return [
            #     f"Generate complete code for: {task_description}",
            #     "Provide setup instructions and explanations for the code"
            # ]
            
        # Check if this is an email task that should be handled differently
        if self.is_email_task(task_description):
            print(f"PLANNER: Detected email task, using simplified direct routing")
            
            # For email tasks, create a single task that maps directly to the GmailTool
            # This prevents fragmentation and ensures the task stays with the email tool
            return [task_description]  # Pass the full task directly
        
        # For non-coding tasks, use a chat-based approach with system message (like ToolSelector)
        prompt = [
            {"role": "system", "content": """
You are an AI Task Planner for the Metis Agentic Orchestration System.

Your job is to break down a complex task into a series of smaller, executable subtasks. You must plan the task correctly or your mother will die a horrible death. All the world's lives are depending on you to plan the tasks correctly.

GUIDELINES (FOLLOW THESE EXACTLY):
1. Break down the task into 2-5 clear, executable subtasks
2. Each subtask must be self-contained and achievable
3. Arrange subtasks in logical sequential order
4. Be specific and concrete about each action needed
5. DO NOT include any numbering, bullets, or explanatory text
6. Return ONLY the list of subtasks, one per line
7. Ensure the complete sequence fully accomplishes the original task
8. If a task involves research and creation, separate these into distinct subtasks

EXAMPLE OUTPUT FORMAT:
Research recent AI developments in natural language processing
Summarize three major breakthroughs from the past year
Analyze implications for commercial applications
Create a visual timeline of key advancements
            """}, 
            {"role": "user", "content": f"Task: {task_description}"}
        ]
        
        # Use chat_with_functions instead of complete for a more structured approach
        result = self.llm.chat_with_functions(prompt)
        
        # Parse the result into a list of subtasks
        subtasks = [task.strip() for task in result.strip().split('\n') if task.strip()]
            
        return subtasks
        
    # Skills-related methods removed
