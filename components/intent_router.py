class IntentRouter:
    """
    Determines whether a user query is a simple question or a complex task.
    Enhanced with better simple query detection and proper LLM interface usage.
    """
    def __init__(self):
        from components.llm_interface import get_llm
        self.llm = get_llm()
        
    def _is_simple_greeting(self, query: str) -> bool:
        """Detect simple greetings and common phrases that don't need complex processing"""
        query_lower = query.lower().strip()
        
        # Simple greetings and common phrases
        simple_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what can you do', 'what are you', 'who are you', 'help', 'thanks', 'thank you',
            'bye', 'goodbye', 'exit', 'quit', 'stop', 'how do you work', 'what is your name',
            'what do you know', 'can you help', 'nice to meet you', 'pleased to meet you'
        ]
        
        # Check for exact matches or simple variations
        for pattern in simple_patterns:
            if pattern in query_lower or query_lower.startswith(pattern):
                return True
        
        # Also check if it's very short (likely a greeting or simple question)
        return len(query.split()) <= 3 and len(query) <= 25
        
    def _is_simple_question(self, query: str) -> bool:
        """Detect simple questions that seek direct information"""
        query_lower = query.lower().strip()
        
        # Question indicators
        question_patterns = [
            'what is', 'what are', 'what does', 'what do',
            'who is', 'who are', 'who was', 'who were',
            'when is', 'when was', 'when did', 'when will',
            'where is', 'where are', 'where was', 'where can',
            'why is', 'why are', 'why did', 'why do',
            'how is', 'how are', 'how do', 'how does', 'how can',
            'which is', 'which are'
        ]
        
        # Check for question patterns
        for pattern in question_patterns:
            if query_lower.startswith(pattern):
                return True
        
        # Check for questions ending with question mark
        return query.strip().endswith('?') and len(query.split()) <= 15
        
    def _is_complex_task(self, query: str) -> bool:
        """Detect queries that clearly require task execution"""
        query_lower = query.lower().strip()
        
        # Task indicators
        task_patterns = [
            'create', 'build', 'make', 'generate', 'write', 'compose', 'draft',
            'develop', 'implement', 'design', 'plan', 'execute', 'run',
            'analyze', 'calculate', 'compute', 'process', 'convert',
            'send', 'email', 'search for', 'find and', 'research and',
            'code', 'program', 'script', 'algorithm'
        ]
        
        # Check for task patterns
        for pattern in task_patterns:
            if pattern in query_lower:
                return True
                
        return False
        
    def classify(self, user_input: str) -> str:
        """
        Classify the user input as either a 'question' or 'task'.
        
        Args:
            user_input: The text input from the user
            
        Returns:
            String classification: 'question' or 'task'
        """
        # Skip complex processing for obvious simple cases
        if self._is_simple_greeting(user_input):
            print(f"[IntentRouter] Rule-based classification: 'question' (greeting) -> '{user_input}'")
            return "question"
        
        if self._is_simple_question(user_input):
            print(f"[IntentRouter] Rule-based classification: 'question' (simple) -> '{user_input}'")
            return "question"
            
        if self._is_complex_task(user_input):
            print(f"[IntentRouter] Rule-based classification: 'task' (complex) -> '{user_input}'")
            return "task"
        
        # Use LLM for ambiguous cases
        return self._llm_classify(user_input)
        
    def _llm_classify(self, user_input: str) -> str:
        """Use LLM to classify ambiguous queries"""
        prompt = f"""
        Determine if the following is a simple question or a complex task requiring execution.
        
        Input: {user_input}
        
        Guidelines:
        - Questions seek information or explanations
        - Tasks require actions, creation, or multi-step processes
        
        If this is a question seeking information, respond with 'question'.
        If this requires executing actions or creating something, respond with 'task'.
        
        Classification:
        """
        
        try:
            # Use the correct method name for LLM completion
            result = self.llm.complete(prompt).strip().lower()
            
            # Parse the result
            if "question" in result:
                classification = "question"
            elif "task" in result:
                classification = "task"
            else:
                # Default fallback based on content analysis
                classification = self._fallback_classify(user_input)
                
            print(f"[IntentRouter] LLM classification: '{classification}' -> '{user_input[:50]}...'")
            return classification
            
        except Exception as e:
            print(f"[IntentRouter] LLM call failed: {e}")
            # Use fallback classification
            return self._fallback_classify(user_input)
            
    def _fallback_classify(self, user_input: str) -> str:
        """Fallback classification when LLM is unavailable"""
        user_lower = user_input.lower()
        
        # Question indicators
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'who', 'which', '?']
        
        # Task indicators  
        task_indicators = [
            'create', 'make', 'build', 'generate', 'write', 'code', 'develop',
            'analyze', 'calculate', 'send', 'email', 'search', 'find',
            'help me', 'can you', 'please', 'i need', 'i want'
        ]
        
        # Count indicators
        question_score = sum(1 for indicator in question_indicators if indicator in user_lower)
        task_score = sum(1 for indicator in task_indicators if indicator in user_lower)
        
        # Simple heuristic: if it ends with '?' or has question words, it's likely a question
        if user_input.strip().endswith('?') or question_score > task_score:
            classification = "question"
        else:
            classification = "task"
            
        print(f"[IntentRouter] Fallback classification: '{classification}' -> '{user_input[:50]}...'")
        return classification