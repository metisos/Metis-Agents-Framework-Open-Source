"""
Email task router for the Metis Agentic Orchestration System.
This helper module detects and routes email-related tasks directly to the GmailTool.
"""

def is_email_task(task_description):
    """
    Determine if a task is primarily related to email handling.
    
    Args:
        task_description: String task description or task dictionary
        
    Returns:
        Boolean indicating if this is an email-related task
    """
    # Handle both string tasks and dictionary task specs
    if isinstance(task_description, dict):
        # Check for email skills in required_skills
        skills = task_description.get("required_skills", [])
        if any(skill in ["email_management", "email_summarization"] for skill in skills):
            return True
        # If no skills match, check the task text
        task_text = task_description.get("task", "")
    else:
        task_text = task_description
    
    # Convert to lowercase for case-insensitive matching
    task_lower = task_text.lower()
    
    # List of email-related keywords to check
    email_keywords = [
        'email', 'gmail', 'read email', 'check email', 'my inbox', 'mail', 'emails',
        'read my email', 'read my emails', 'summarize email', 'summarize emails',
        'email summary', 'inbox summary', 'check my inbox', 'read inbox', 'unread'
    ]
    
    # Return True if any keyword is found in the task
    return any(keyword in task_lower for keyword in email_keywords)
