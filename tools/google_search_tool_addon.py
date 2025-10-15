import re

# Methods to add to GoogleSearchTool

def can_handle(self, task: str) -> bool:
    """
    Determine if this tool can handle the given task.
    
    Args:
        task: The task description
        
    Returns:
        True if this tool can handle the task, False otherwise
    """
    # Check for search-related keywords
    search_keywords = [
        'search', 'find', 'look up', 'google', 'web search', 'information', 
        'research', 'look for', 'find information', 'search the web',
        'find out about', 'learn about', 'what is', 'who is', 'tell me about',
        # Add more research-oriented keywords
        'investigate', 'explore', 'analyze', 'study', 'examine', 'review literature',
        'gather data', 'collect information', 'conduct research', 'preliminary research',
        'background research', 'literature review', 'comprehensive research',
        'identify sources', 'find articles', 'find sources', 'identify literature',
        'scholarly articles', 'academic research', 'find papers', 'academic papers'
    ]
    
    # Research-specific phrases that might appear in research paper tasks
    research_phrases = [
        'conduct preliminary research', 'literature review', 'gather information', 
        'find sources', 'background information', 'academic sources', 'collect data',
        'gather research', 'investigate topic', 'explore the subject', 'analyze existing research'
    ]
    
    task_lower = task.lower()
    
    # Check if task contains any search keywords
    keyword_match = any(keyword in task_lower for keyword in search_keywords)
    
    # Check if task contains any research phrases
    phrase_match = any(phrase in task_lower for phrase in research_phrases)
    
    # Special case for research paper tasks
    research_paper_task = 'research paper' in task_lower and any(action in task_lower for action in ['create', 'write', 'develop'])
    
    return keyword_match or phrase_match or research_paper_task

def execute(self, task: str) -> dict:
    """
    Execute a web search based on the task description.
    
    Args:
        task: The task description
        
    Returns:
        Formatted search results
    """
    # Extract the search query from the task
    query = self._extract_query(task)
    
    # Execute the search
    results = self.search(query)
    
    # Format the results for readability
    if isinstance(results, dict) and 'error' in results and results['error']:
        return f"Error performing search: {results['error']}"
        
    # Format results for display
    formatted_results = []
    
    if isinstance(results, dict) and 'results' in results:
        for i, result in enumerate(results['results'], 1):
            formatted = f"{i}. {result.get('title', 'No title')}\n"
            formatted += f"   URL: {result.get('link', 'No link')}\n"
            formatted += f"   Snippet: {result.get('snippet', 'No description')}\n"
            formatted_results.append(formatted)
    
    if not formatted_results:
        return "No search results found."
    
    output = f"Search results for: {query}\n\n"
    output += "\n".join(formatted_results)
    
    return output

def _extract_query(self, task: str) -> str:
    """
    Extract the search query from a task description.
    
    Args:
        task: The task description
        
    Returns:
        The extracted search query
    """
    task_lower = task.lower()
    
    # Try to extract specific patterns
    patterns = [
        r'(?:research|search for|find|look up|investigate|get information about|tell me about)\s+([^.,?!]+)',
        r'(?:information|data|details|facts)\s+(?:about|on|regarding|concerning|for)\s+([^.,?!]+)',
        r'(?:learn|know|understand)\s+(?:about|more about)\s+([^.,?!]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, task_lower)
        if match:
            return match.group(1).strip()
    
    # Try to extract "about X" or "on X" patterns
    about_match = re.search(r'(?:about|on|regarding|concerning|for)\s+([^.,?!]+)', task_lower)
    if about_match:
        return about_match.group(1).strip()
    
    # Remove common task-related words to isolate the topic
    for word in ['research', 'search', 'find', 'information', 'about', 'for', 'investigate', 'details']:
        task = re.sub(r'\b' + word + r'\b', '', task, flags=re.IGNORECASE)
    
    # Use whatever's left as the query, or the original task if nothing's left
    cleaned_query = task.strip()
    return cleaned_query if cleaned_query else task
