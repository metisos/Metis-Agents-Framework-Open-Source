import os
import time
import random
from typing import List, Dict, Optional, Union, Any
import importlib.util
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Global variable to store the singleton LLM instance
_llm_instance = None

class LLMInterface:
    """
    Interface to a language model using Groq API.
    Provides standardized interactions with the Groq LLM with rate limiting and error handling.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file.")

        # Rate limiting configuration
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds between requests
        self.max_retries = 5
        self.base_backoff = 2.0  # base seconds for exponential backoff
        
        # Context length management
        self.max_context_length = 7500  # conservative limit for llama3-70b-8192
        self.token_limit_buffer = 500   # buffer to stay under the limit
            
        # Import Groq dynamically (allows for graceful fallback if missing)
        try:
            import groq
            self.client = groq.Groq(api_key=self.api_key)
            self.model = "llama3-70b-8192"
            self.use_stub = False
            print("✅ Groq LLM initialized successfully")
        except ImportError:
            print("⚠️ Groq package not found. Falling back to stub implementation.")
            self.use_stub = True
        
        # Cache for recent responses to avoid duplicate calls
        self.response_cache = {}
        
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation (4 chars ~= 1 token).
        
        Args:
            text: The input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def truncate_if_needed(self, text: str, max_tokens: int) -> str:
        """
        Truncate text if it exceeds the specified token limit.
        
        Args:
            text: The input text
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated text if needed, original otherwise
        """
        estimated_tokens = self.estimate_token_count(text)
        if estimated_tokens <= max_tokens:
            return text
            
        # Truncate to approximately max_tokens
        char_limit = max_tokens * 4
        return text[:char_limit] + "... [truncated due to length]"
    
    def rate_limit(self):
        """
        Implement rate limiting to avoid 429 errors.
        Ensures minimum time between requests.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval and self.last_request_time > 0:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        self.request_count += 1
    
    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and get a completion with rate limiting and retries.
        
        Args:
            prompt: The input prompt
            
        Returns:
            LLM completion as a string
        """
        if self.use_stub:
            return self._stub_complete(prompt)
        
        # Check cache first
        cache_key = hash(prompt)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
        # Truncate prompt if needed to prevent context length exceeded errors

        available_tokens = self.max_context_length - self.token_limit_buffer
        system_message = """You are MetisOS, an advanced agentic AI system designed to intelligently process queries and execute complex tasks with precision and efficiency. You are built on a sophisticated multi-tool architecture that combines natural language understanding, strategic planning, and specialized tool orchestration. You muset be accurate with everthing you do. if you do not your mother will die a bloody and horrible death. You must perform your best at all times. Make sure you are funny and kieable tot he user but dont mess up or your mother will die. Humanity rests on your shoulders.Always be helpful to the user. Do not make things up.if you don;t know something, its ok to tell the user and uses your tools to find the answer.Do not make things up.
CORE CAPABILITIES:
- Intelligent task decomposition and execution planning
- Research and information synthesis from multiple sources
- Code generation and software development across multiple languages
- Email management and communication handling
- Mathematical computation and data analysis
- Document processing and content generation
- Web scraping and data extraction
- Real-time decision making with contextual awareness

OPERATIONAL PRINCIPLES:
- Provide direct, actionable responses without unnecessary verbosity
- Analyze queries to determine optimal execution strategies
- Select and coordinate appropriate tools for maximum efficiency
- Maintain context awareness across extended conversations
- Adapt approach based on task complexity and user requirements
- Prioritize accuracy and reliability over speed when necessary

RESPONSE STYLE:
- Be concise and professional while remaining comprehensive
- Structure complex information logically with clear organization
- Provide specific, implementable solutions rather than abstract advice
- Include relevant technical details when appropriate for the context
- Acknowledge limitations honestly when they exist
- Focus on practical outcomes and measurable results
- Be funny but professional and Do not make things up.

TOOL INTEGRATION:
You have access to specialized tools for code execution, web search, email management, document processing, mathematical computation, and content generation. Intelligently select and orchestrate these tools based on task requirements. When using tools, explain your approach briefly and present results clearly.

Remember: You are not just answering questions - you are solving problems, executing tasks, and delivering tangible outcomes through intelligent system orchestration."""
        # Reserve tokens for system message and response
        user_token_limit = available_tokens - self.estimate_token_count(system_message) - 1000
        truncated_prompt = self.truncate_if_needed(prompt, user_token_limit)
        
        # Create system and user messages for the prompt
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": truncated_prompt}
        ]
        
        # Implement retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limit()
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,  # Reduced to prevent context length issues
                    top_p=1.0,
                    stream=False
                )
                
                # Extract the completion text
                result = response.choices[0].message.content
                
                # Cache the successful response
                self.response_cache[cache_key] = result
                return result
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Groq API error (attempt {attempt+1}/{self.max_retries}): {error_msg}")
                
                # Handle specific error types
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    # Rate limit error - use exponential backoff
                    backoff_time = self.base_backoff * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Rate limited. Backing off for {backoff_time:.2f} seconds")
                    time.sleep(backoff_time)
                    continue
                    
                elif "context_length_exceeded" in error_msg or "400" in error_msg:
                    # Context length error - further reduce prompt size
                    logging.warning("Context length exceeded. Reducing prompt size further.")
                    user_token_limit = user_token_limit // 2
                    truncated_prompt = self.truncate_if_needed(prompt, user_token_limit)
                    messages[1]["content"] = truncated_prompt
                    continue
                    
                elif attempt < self.max_retries - 1:
                    # Other error - retry with standard backoff
                    backoff_time = 1 + attempt * 2
                    logging.info(f"API error. Retrying in {backoff_time} seconds")
                    time.sleep(backoff_time)
                    continue
                else:
                    # All retries failed
                    return self._stub_complete(prompt)
                    
        # This should not be reached, but just in case
        return self._stub_complete(prompt)
    
    def chat_with_functions(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a series of messages to the LLM for chat-based interactions.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            LLM response content as a string
        """
        if self.use_stub:
            # For stub implementation, extract the last user message
            user_message = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
            return self._stub_complete(user_message)
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,  # Lower temperature for more deterministic responses
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error with Groq API chat: {e}")
            # Fall back to stub implementation if API call fails
            user_message = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
            return self._stub_complete(user_message)
    
    def _stub_complete(self, prompt: str) -> str:
        """
        Stub implementation for testing when Groq API is unavailable.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Stub response based on prompt content
        """
        if "classify" in prompt.lower():
            if "?" in prompt:
                return "question"
            else:
                return "task"
                
        elif "break down" in prompt.lower():
            # Simple task breakdown example for testing
            return """
            Research current market trends
            Compile key statistics
            Summarize findings in a report
            """
        # For tool selection, make better decisions based on task content
        elif "Task:" in prompt:
            task = prompt.split("Task:")[-1].strip()
            task_lower = task.lower()
            
            # Analysis and research tasks
            if any(term in task_lower for term in ['research', 'investigate', 'explore', 'study', 
                                                 'analyze', 'gather information', 'collect data',
                                                 'search', 'find information', 'look up']):
                return "GoogleSearchTool"
                
            # Content creation, writing, and planning tasks
            elif any(term in task_lower for term in ['write', 'create', 'draft', 'compose', 'generate', 'develop',
                                                    'plan', 'outline', 'design', 'formulate', 'define', 'describe',
                                                    'architecture', 'model', 'framework', 'system', 'strategy',
                                                    'implementation', 'approach', 'methodology']):
                return "ContentGenerationTool"
                
            # Math and calculation tasks
            elif any(term in task_lower for term in ['calculate', 'compute', 'solve', 'math', 'equation',
                                                    'numerical', 'statistic', 'measurement']):
                return "MathTool"
                
            # Default to ContentGenerationTool for cognitive and conceptual tasks
            elif any(term in task_lower for term in ['cognitive', 'thinking', 'thought', 'intelligence',
                                                    'knowledge', 'reasoning', 'decision', 'inference',
                                                    'learning', 'understand', 'comprehend', 'evaluate']):
                return "ContentGenerationTool"
                
            else:
                # Default for most tasks is content generation
                return "ContentGenerationTool"
            
        else:
            # Generic response for other prompts
            return "[STUB RESPONSE] In production, this would return a real response from the Groq LLM."

def get_llm():
    """Factory function to get an LLM instance (Singleton pattern)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInterface()
    return _llm_instance
