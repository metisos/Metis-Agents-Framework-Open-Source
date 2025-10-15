# single_agent/tools/math_tool.py

import re
import math
from typing import Union, List, Dict, Any

class MathTool:
    """
    Tool for mathematical operations and text analysis.
    Adapted for the single agent architecture with task detection capabilities.
    
    Implements the data_analysis skill.
    """
    
    def __init__(self):
        """Initialize the Math tool."""
        self.supported_skills = ["data_analysis"]
    
    def count_chars(self, text: str, char: str = None) -> Union[int, Dict[str, int]]:
        """
        Count occurrences of a character in text, or count all characters.
        
        :param text: Text to analyze
        :param char: Specific character to count (if None, counts all)
        :return: Count of the character or dictionary of all character counts
        """
        if not text:
            return 0 if char else {}
            
        if char:
            return text.count(char)
        else:
            return {c: text.count(c) for c in set(text)}
    
    def word_count(self, text: str) -> int:
        """
        Count words in a text.
        
        :param text: Text to analyze
        :return: Number of words
        """
        if not text:
            return 0
            
        # Split by whitespace and count non-empty strings
        return len([w for w in re.split(r'\s+', text) if w])
    
    def calculate(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.
        
        :param expression: Mathematical expression as string
        :return: Result of the calculation
        :raises: ValueError for invalid or unsafe expressions
        """
        # Remove all whitespace
        expression = re.sub(r'\s+', '', expression)
        
        # Check if expression contains only allowed characters
        if not re.match(r'^[0-9+\-*/().]+$', expression):
            raise ValueError("Expression contains invalid characters")
        
        try:
            # Use eval with restricted globals and locals
            safe_globals = {"__builtins__": None}
            safe_locals = {
                "abs": abs, 
                "max": max, 
                "min": min,
                "round": round,
                "pow": pow,
                "math": math
            }
            
            # Replace common math functions with explicit calls
            expression = expression.replace("sqrt", "math.sqrt")
            expression = expression.replace("sin", "math.sin")
            expression = expression.replace("cos", "math.cos")
            expression = expression.replace("tan", "math.tan")
            expression = expression.replace("log", "math.log")
            expression = expression.replace("exp", "math.exp")
            
            result = eval(expression, safe_globals, safe_locals)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {str(e)}")
    
    def statistics(self, numbers: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of numbers.
        
        :param numbers: List of numeric values
        :return: Dictionary with statistics (mean, median, etc.)
        """
        if not numbers:
            return {"error": "Empty list provided"}
            
        try:
            # Convert all values to float
            nums = [float(n) for n in numbers]
            
            # Sort for percentiles and median
            sorted_nums = sorted(nums)
            
            return {
                "count": len(nums),
                "sum": sum(nums),
                "mean": sum(nums) / len(nums),
                "median": sorted_nums[len(sorted_nums) // 2] if len(sorted_nums) % 2 == 1 else 
                          (sorted_nums[len(sorted_nums) // 2 - 1] + sorted_nums[len(sorted_nums) // 2]) / 2,
                "min": min(nums),
                "max": max(nums),
                "range": max(nums) - min(nums),
                "variance": sum((x - (sum(nums) / len(nums))) ** 2 for x in nums) / len(nums),
            }
        except Exception as e:
            return {"error": f"Failed to calculate statistics: {str(e)}"}
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract all numbers from a text string.
        
        :param text: Text to extract numbers from
        :return: List of extracted numbers
        """
        if not text:
            return []
            
        # Find all numeric patterns in the text
        number_pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, text)
        
        # Convert to float
        return [float(match) for match in matches]
        
    def can_handle(self, task: str) -> bool:
        """
        Determine if this tool can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            True if this tool can handle the task, False otherwise
        """
        # Check for mathematical keywords and patterns
        math_keywords = ['calculate', 'compute', 'solve', 'math', 'arithmetic', 'equation',
                         'statistics', 'average', 'mean', 'median', 'count', 'numbers']
        math_operators = ['+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos', '%']
        
        # Look for math keywords
        if any(keyword in task.lower() for keyword in math_keywords):
            return True
            
        # Look for math operators
        if any(op in task for op in math_operators):
            return True
            
        # Look for number patterns (e.g., "what is 5 + 3?")
        number_pattern = r'\d+\s*[\+\-\*\/\^]\s*\d+'
        return bool(re.search(number_pattern, task))
    
    def score_for_context(self, context: dict) -> float:
        """
        Score how well this tool matches the given context.
        Used by the skills system to select the best tool for a task.
        
        Args:
            context: Dictionary containing task details and other context
            
        Returns:
            A score between 0 and 1, with higher values indicating better match
        """
        task = context.get("task", "")
        query = context.get("query", "")
        
        if not task:
            return 0.0
            
        task_lower = task.lower()
        
        # High-priority math tasks
        high_priority_keywords = [
            "calculate", "compute", "solve equation", "mathematical", 
            "statistics", "analyze data", "data analysis", "formula",
            "average", "mean", "median", "standard deviation"
        ]
        
        # Check for explicit math indicators
        for keyword in high_priority_keywords:
            if keyword in task_lower:
                return 0.9  # Very high score for explicit math tasks
        
        # Check for math operators
        math_operators = ['+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos', '%']
        if any(op in task for op in math_operators):
            return 0.85  # High score for math operators
        
        # Check for number patterns
        number_pattern = r'\d+\s*[\+\-\*\/\^]\s*\d+'
        if re.search(number_pattern, task):
            return 0.8  # High score for number patterns
        
        # Financial analysis terms
        financial_terms = ["ROI", "return on investment", "profit margin", "interest rate", "compound interest"]
        for term in financial_terms:
            if term.lower() in task_lower:
                return 0.75  # Good score for financial terms
        
        # Check for counting/analysis operations
        counting_terms = ["count", "tally", "sum", "total", "average"]
        for term in counting_terms:
            if term in task_lower:
                return 0.7
                
        # Low default score
        return 0.2
    
    def execute(self, task: str) -> str:
        """
        Execute a mathematical task based on the task description.
        
        Args:
            task: The task description
            
        Returns:
            Result of the mathematical operation
        """
        # Handle task as dictionary with task and skills (from skills system)
        if isinstance(task, dict) and "task" in task:
            actual_task = task["task"]
            required_skills = task.get("required_skills", [])
            # We can use required_skills for specialized handling if needed
        else:
            actual_task = task
            
        # Extract operation type from task
        task_lower = actual_task.lower()
        
        # Check for sum operations
        if any(term in task_lower for term in ['sum', 'add', 'total']):
            # Extract numbers from the task
            numbers = self.extract_numbers(task)
            if numbers:
                total = sum(numbers)
                return f"The sum of {', '.join(map(str, numbers))} is {total}"
            else:
                return "No numbers found in the task for summation."
                
        # Check for statistics-related operations
        if any(term in task_lower for term in ['statistics', 'stats', 'analyze numbers']):
            # Extract numbers from the task
            numbers = self.extract_numbers(task)
            if numbers:
                stats = self.statistics(numbers)
                return f"Statistical analysis: {stats}"
            else:
                return "No numbers found in the task for statistical analysis."
        
        # Check for word count operations
        if any(term in task_lower for term in ['word count', 'count words']):
            # Extract text after the command
            match = re.search(r'(?:word count|count words)[:\s]+(.*)', task, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                count = self.word_count(text)
                return f"Word count: {count}"
            else:
                return "No text provided for word counting."
        
        # Check for character count operations
        if any(term in task_lower for term in ['character count', 'count characters', 'count chars']):
            match = re.search(r'(?:character count|count characters|count chars)[:\s]+(.*)', task, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                counts = self.count_chars(text)
                return f"Character counts: {counts}"
            else:
                return "No text provided for character counting."
        
        # Default to mathematical expression evaluation
        # Try to extract an expression from the task
        expression = None
        
        # Look for explicit expression patterns (e.g., "calculate 5 + 3")
        expression_match = re.search(r'(?:calculate|compute|solve|evaluate|what is)[:\s]+(.*)', task, re.IGNORECASE)
        if expression_match:
            expression = expression_match.group(1).strip()
        
        # Look for direct expressions (e.g., "5 + 3")
        if not expression:
            expression_match = re.search(r'(\d+\s*[\+\-\*\/\^]\s*\d+(?:\s*[\+\-\*\/\^]\s*\d+)*)', task)
            if expression_match:
                expression = expression_match.group(1)
        
        if expression:
            try:
                result = self.calculate(expression)
                return f"The result of {expression} is {result}"
            except ValueError as e:
                return f"Error: {str(e)}"
        
        return "I couldn't identify a mathematical operation to perform. Please provide a clearer instruction."
