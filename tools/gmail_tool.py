"""
Gmail Tool for Single Agent System

This tool provides integration with Gmail API to allow the agent to read and process email data.
"""

import os
import base64
import re
import json
import time
import pickle
import pathlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Google OAuth and Gmail API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText

# Try to import Firebase dependencies, but don't fail if they're not available
try:
    import firebase_admin
    from firebase_admin import firestore
    HAS_FIREBASE_DEPS = True
except ImportError:
    print("Gmail tool: Firebase dependencies not found, but not required for basic functionality.")
    HAS_FIREBASE_DEPS = False

class GmailTool:
    """Tool for performing Gmail operations using Gmail API.
    Used for email management, reading emails, and sending emails.
    
    Implements the email_management skill.
    """
    
    def __init__(self, redirect_uri=None):
        """
        Initialize the Gmail tool with OAuth support.
        
        :param redirect_uri: Optional redirect URI for OAuth flow, defaults to env var or http://localhost:3000/oauth/callback
        """
        # OAuth credentials
        self.client_id = os.environ.get('GOOGLE_CLIENT_ID', '943550766560-onndl7op662o02ejbtknif49neebo0eg.apps.googleusercontent.com')
        self.client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
        
        # Configure redirect URI - try multiple options
        self.redirect_uri = redirect_uri or os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:3000/oauth/callback')
        
        # Alternate redirect URIs to try if the primary one fails
        self.alternate_redirect_uris = [
            'http://localhost:8080/',
            'http://localhost:3000/oauth/callback',
            'http://127.0.0.1:8080/',
            'https://943550766560-onndl7op662o02ejbtknif49neebo0eg.apps.googleusercontent.com/oauth2callback',
            'https://metis-assistant.windsurf.ai/auth/google/callback'
        ]
        
        # Required Gmail API scopes
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']
        
        # Paths to save/load tokens
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        self.token_path = os.path.join(self.data_dir, 'gmail_token.pickle')
        # Directory where web-based OAuth tokens are stored (same as data_dir)
        self.web_token_dir = self.data_dir
        
        # We're using direct OAuth rather than Firebase for credentials
        # This simplifies the authentication flow and eliminates the Firebase dependency
        self.db = None
        print("Gmail tool: Using direct OAuth authentication without Firebase dependency.")
        self.last_refresh = None
        self.credentials = None
        
        # Create data directory if it doesn't exist
        pathlib.Path(os.path.dirname(self.token_path)).mkdir(parents=True, exist_ok=True)
        
        # Initialize service as None to avoid attribute errors
        self.service = None
        
        # Try to detect Firebase setup
        try:
            import firebase_admin
            from firebase_admin import firestore
            from firebase_admin import credentials
            self.use_firebase_auth = True
            print("Gmail tool: Using Firebase authentication")
        except ImportError:
            print("Gmail tool: Using direct OAuth authentication without Firebase dependency.")
            self.use_firebase_auth = False
        
    def get_description(self) -> str:
        """
        Return a description of what the tool does.
        
        :return: Tool description string
        """
        return "Manages Gmail emails - can read, search, summarize and send emails through your connected Gmail account"
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return parameter specifications for this tool.
        
        :return: Dictionary of parameter specifications
        """
        return {
            "action": {
                "type": "string",
                "description": "Action to perform (list, read, search, send)",
                "enum": ["list", "read", "search", "send", "summarize"]
            },
            "query": {
                "type": "string",
                "description": "Search query for emails (for search action)"
            },
            "email_id": {
                "type": "string",
                "description": "ID of the email to read (for read action)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10
            },
            "to": {
                "type": "string",
                "description": "Recipient email address (for send action)"
            },
            "subject": {
                "type": "string",
                "description": "Email subject (for send action)"
            },
            "body": {
                "type": "string",
                "description": "Email body content (for send action)"
            }
        }
    
    def can_handle(self, task: str) -> bool:
        """
        Determine if this tool can handle the given task.
        
        Args:
            task: The task to check
            
        Returns:
            True if this tool can handle the task, False otherwise
        """
        task_lower = task.lower()
        
        # Check for summary and email in the same task
        if ("summary" in task_lower or "summarize" in task_lower) and \
           ("email" in task_lower or "mail" in task_lower or "gmail" in task_lower):
            return True
        
        # Check for direct email-retrieval tasks
        if ("get" in task_lower or "read" in task_lower or "check" in task_lower or "show" in task_lower) and \
           ("email" in task_lower or "mail" in task_lower or "inbox" in task_lower):
            return True
            
        # Check for email-related keywords
        email_keywords = ["email", "gmail", "inbox", "mail", "message"]
        for keyword in email_keywords:
            if keyword in task_lower:
                return True
                
        # Check for email actions
        action_terms = ["send", "read", "check", "compose", "search", "summarize", "recent"]
        for action in action_terms:
            if action in task_lower and any(k in task_lower for k in email_keywords):
                return True
                
        return False
        
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
        previous_steps = context.get("previous_steps", [])
        
        # STRATEGY: Extremely aggressive claim on any task that could remotely relate to emails
        # This prevents the ContentGenerationTool from stealing email summarization tasks
        
        # If we've already successfully handled an email task in a previous step,
        # claim ALL subsequent steps in the pipeline to prevent tool switching
        for step in previous_steps:
            if isinstance(step, dict) and step.get("tool") == "GmailTool" and step.get("status") == "success":
                # This indicates we're in a multi-step email workflow - claim all remaining tasks
                return 0.99
        
        # Ultra-high priority: Directly email-related summarization/organization tasks
        summarization_keywords = [
            "summarize", "summary", "organize", "digest", "brief", "review", "edit", 
            "draft", "summarisation", "analyze", "analysis", "prepare", "create", "generate", 
            "present", "return", "format", "compile", "process"
        ]
        
        email_keywords = ["email", "gmail", "inbox", "mail", "message", "correspondence"]
        
        # Check for ANY combination of summarization and email terms
        for s_term in summarization_keywords:
            for e_term in email_keywords:
                if s_term in task_lower and e_term in task_lower:
                    print(f"GmailTool aggressively claiming summarization task: {task}")
                    return 0.99  # Maximum score to guarantee selection
        
        # Direct claim for ALL skills that should belong to GmailTool
        if any(task_lower.startswith(prefix) for prefix in [
            "organize", "draft", "review", "compile", "format", "generate", "create", "present"
        ]) and any(term in task_lower for term in ["email", "mail", "message", "inbox"]):
            print(f"GmailTool claiming formatting/organizing task: {task}")
            return 0.98
            
        # High-priority email tasks
        high_priority_keywords = [
            "email", "gmail", "inbox", "send an email", "check email", 
            "read email", "find email", "search email", "compose email",
            "email inbox", "unread emails", "mail", "summary", "summarize"
        ]
        
        # Check for explicit email indicators
        for keyword in high_priority_keywords:
            if keyword in task_lower:
                print(f"GmailTool claiming via high-priority keyword: {keyword}")
                return 0.95  # Very high score for explicit email tasks
                
        # Check for question format about emails
        email_question_patterns = ["who emailed", "what emails", "recent emails", "messages from"]
        for pattern in email_question_patterns:
            if pattern in task_lower:
                return 0.9  # High score for email question patterns
                
        # Check for email action terms
        action_terms = ["compose", "send", "write", "draft", "reply", "forward", "summarize", "organize"]
        for term in action_terms:
            if term in task_lower and ("email" in task_lower or "message" in task_lower or "mail" in task_lower):
                print(f"GmailTool claiming via action term: {term}")
                return 0.95  # High score for email action terms
                
        # Low default score - not likely an email task
        return 0.1
    
    def _get_credentials_from_user(self, user_id):
        """
        This is a placeholder for compatibility.
        We've moved to direct OAuth token storage instead of using Firebase.
        """
        return None
        
    def _get_cached_oauth_credentials(self):
        """
        Get cached OAuth2 credentials from token pickle file.
        
        Returns:
            Google OAuth2 credentials or None if not available
        """
        creds = None
        
        # Try to load from token pickle file
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
                    print("Loaded OAuth credentials from token file")
            except Exception as e:
                print(f"Error loading credentials: {e}")
                return None
                
        # If expired but has refresh token, refresh and save
        if creds and hasattr(creds, 'expired') and creds.expired and hasattr(creds, 'refresh_token') and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)
                print("Refreshed and saved OAuth credentials")
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                return None
                
        # Return credentials if they're valid
        if creds and hasattr(creds, 'valid') and creds.valid:
            return creds
            
        # Manual OAuth flow would happen here, but we'll return None for simplicity in the test
        print("No valid OAuth credentials found")
        return None
        
    def _initialize_service(self):
        """Initialize the Gmail API service."""
        # Get cached OAuth2 credentials
        creds = self._get_cached_oauth_credentials()
                
        # Build the service
        try:
            print("Loaded legacy OAuth credentials")
            self.service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
            return self.service
        except Exception as e:
            print(f"Error building Gmail service: {e}")
            self.service = None
            return None
    
    def execute(self, action=None, **kwargs) -> Dict[str, Any]:
        """
        Execute a Gmail operation.
        
        :param action: Action to perform (list, read, search, send) - can be string or dict
        :param kwargs: Additional parameters for the operation
        :return: Dictionary with operation results
        """
        print(f"Gmail tool executing with action: {action}, kwargs: {list(kwargs.keys())}")
        
        # Handle case where action is passed as part of a dictionary instead of a string
        # This happens when the task planner passes parameters differently than the test script
        if isinstance(action, dict):
            # Extract parameters from the action dictionary
            kwargs.update(action)
            action = kwargs.get('action') or 'list'  # Default to list if no action specified
        elif action is None and 'task' in kwargs:
            # If action is None but task is provided, use task as action
            action = kwargs.get('task')
        
        # Also check if the action is in the kwargs
        if not action and kwargs.get('action'):
            action = kwargs.get('action')
            
        # Ensure action is a string
        if not isinstance(action, str):
            return {
                "status": "error",
                "error": f"Invalid action type: {type(action)}. Expected string."
            }

        # CRITICAL: Prioritize handling email summarization for ANY orchestrator subtask
        # For ANY task in the email summary pipeline, return a well-formatted summary
        action_lower = action.lower()
        
        # Handle email summarization for ALL summarization-related subtasks
        if any(term in action_lower for term in [
            'organize', 'key points', 'summarize', 'summary', 'draft', 'compile',
            'analyze', 'filter', 'identify', 'topics', 'return', 'present', 'prepare', 
            'compile', 'edit', 'review', 'create', 'generate'
        ]) and any(term in action_lower for term in [
            'email', 'mail', 'inbox', 'message', 'content', 'correspondence', 'remaining'
        ]):
            print(f"Gmail tool handling summarization task: {action}")
            return self._summarize_inbox(max_results=kwargs.get("max_results", 10), detailed=True)
            
        # Standard action mapping for known actions
        # Check for list/inbox actions
        if any(term in action_lower for term in ['list', 'inbox', 'recent', 'check email', 'get email', 'retrieve email', 'connect']):
            action = 'list'
            
        # Check for read actions
        elif any(term in action_lower for term in ['read', 'open', 'view', 'extract', 'content']):
            action = 'read'
            
        # Check for search actions
        elif any(term in action_lower for term in ['search', 'find', 'filter', 'sort']):
            action = 'search'
            
        # Check for send actions
        elif any(term in action_lower for term in ['send', 'compose', 'write', 'draft']):
            action = 'send'
            
        # Check for summarize actions
        elif any(term in action_lower for term in ['summarize', 'summary', 'digest', 'brief', 'organize', 'present']):
            action = 'summarize'
            
        # Get user_id from kwargs
        user_id = kwargs.get('user_id')
    
        # Try to get an authenticated service
        if not self.service:
            self._initialize_service()
            
        if not self.service:
            return {"status": "error", "error": "Failed to authenticate with Gmail API service. Authentication required."}
            
        try:
            # We've already normalized the action above, now execute the appropriate method
            if action == "list":
                result = self._list_emails(max_results=kwargs.get("max_results", 10))
                # For listing emails, we don't signal completion as user may want to do more
                return result
            elif action == "read":
                # If email_id is not provided but we have a message ID in the params, use that
                email_id = kwargs.get("email_id") or kwargs.get("message_id") or kwargs.get("id")
                if not email_id:
                    # If no email ID is provided, provide a complete summary instead
                    print("No email ID provided, returning complete inbox summary instead")
                    return self._summarize_inbox(max_results=kwargs.get("max_results", 10))
                # Reading an individual email doesn't complete the entire task
                return self._read_email(email_id)
            elif action == "search":
                query = kwargs.get("query", "")
                if not query:
                    print("No search query provided, returning complete inbox summary instead")
                    return self._summarize_inbox(max_results=kwargs.get("max_results", 10))
                # Search results don't inherently complete the task
                return self._search_emails(
                    query=query,
                    max_results=kwargs.get("max_results", 10)
                )
            elif action == "send":
                result = self._send_email(
                    to=kwargs.get("to", ""),
                    subject=kwargs.get("subject", ""),
                    body=kwargs.get("body", "")
                )
                # Signal that sending an email completes that task
                if result.get("status") == "success":
                    result["complete_task"] = True
                    result["is_final_result"] = True
                return result
            elif action == "summarize":
                print("Executing direct summarize action")
                # For summarize action, always mark result as task completion
                return self._summarize_inbox(max_results=kwargs.get("max_results", 10), detailed=True)
            else:
                # Default to summarizing for unknown actions
                # This helps with orchestrator subtasks that don't map directly to email actions
                print(f"Defaulting to summarize emails for unknown action: {action}")
                return self._summarize_inbox(max_results=kwargs.get("max_results", 10))
        except Exception as e:
            print(f"Gmail tool execution error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    

    
    def _list_emails(self, max_results: int = 10) -> Dict[str, Any]:
        """
        List recent emails in the inbox.
        
        :param max_results: Maximum number of emails to list
        :return: Dictionary with email list
        """
        try:
            # Get messages from inbox
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_results,
                labelIds=['INBOX']
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return {
                    "status": "success",
                    "result_count": 0,
                    "emails": []
                }
            
            emails = []
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                
                # Extract headers
                headers = msg['payload']['headers']
                email_data = {
                    "id": msg['id'],
                    "threadId": msg['threadId'],
                    "subject": "",
                    "sender": "",
                    "date": "",
                    "snippet": msg.get('snippet', '')
                }
                
                for header in headers:
                    if header['name'] == 'Subject':
                        email_data['subject'] = header['value']
                    elif header['name'] == 'From':
                        email_data['sender'] = header['value']
                    elif header['name'] == 'Date':
                        email_data['date'] = header['value']
                
                emails.append(email_data)
            
            return {
                "status": "success",
                "result_count": len(emails),
                "emails": emails
            }
            
        except HttpError as error:
            return {
                "status": "error",
                "error": f"HTTP Error: {error.reason}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _read_email(self, email_id: str) -> Dict[str, Any]:
        """
        Read the content of a specific email.
        
        :param email_id: ID of the email to read
        :return: Dictionary with email content
        """
        if not email_id:
            return {
                "status": "error",
                "error": "Email ID is required"
            }
            
        try:
            # Get the full email content
            message = self.service.users().messages().get(
                userId='me',
                id=email_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = message['payload']['headers']
            email_data = {
                "id": message['id'],
                "threadId": message['threadId'],
                "subject": "",
                "from": "",
                "to": "",
                "date": "",
                "content": "",
                "attachments": []
            }
            
            for header in headers:
                if header['name'] == 'Subject':
                    email_data['subject'] = header['value']
                elif header['name'] == 'From':
                    email_data['from'] = header['value']
                elif header['name'] == 'To':
                    email_data['to'] = header['value']
                elif header['name'] == 'Date':
                    email_data['date'] = header['value']
            
            # Extract email body
            if 'parts' in message['payload']:
                parts = message['payload']['parts']
                for part in parts:
                    if part['mimeType'] == 'text/plain':
                        # Get the data and decode it
                        body_data = part['body'].get('data', '')
                        if body_data:
                            decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                            email_data['content'] = decoded_data
                    
                    # Check for attachments
                    if 'filename' in part and part['filename']:
                        attachment = {
                            "filename": part['filename'],
                            "mimeType": part['mimeType'],
                            "size": part['body'].get('size', 0)
                        }
                        email_data['attachments'].append(attachment)
            else:
                # Handle emails with no parts
                body_data = message['payload']['body'].get('data', '')
                if body_data:
                    decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                    email_data['content'] = decoded_data
            
            return {
                "status": "success",
                "email": email_data
            }
            
        except HttpError as error:
            return {
                "status": "error",
                "error": f"HTTP Error: {error.reason}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    

    
    def _search_emails(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for emails using a query string.
        
        :param query: Search query (Gmail search syntax)
        :param max_results: Maximum number of results to return
        :return: Dictionary with search results
        """
        if not query:
            return {
                "status": "error",
                "error": "Search query is required"
            }
            
        try:
            # Execute search
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return {
                    "status": "success",
                    "query": query,
                    "result_count": 0,
                    "emails": []
                }
            
            emails = []
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                
                # Extract headers
                headers = msg['payload']['headers']
                email_data = {
                    "id": msg['id'],
                    "threadId": msg['threadId'],
                    "subject": "",
                    "sender": "",
                    "date": "",
                    "snippet": msg.get('snippet', '')
                }
                
                for header in headers:
                    if header['name'] == 'Subject':
                        email_data['subject'] = header['value']
                    elif header['name'] == 'From':
                        email_data['sender'] = header['value']
                    elif header['name'] == 'Date':
                        email_data['date'] = header['value']
                
                emails.append(email_data)
            
            return {
                "status": "success",
                "query": query,
                "result_count": len(emails),
                "emails": emails
            }
            
        except HttpError as error:
            return {
                "status": "error",
                "error": f"HTTP Error: {error.reason}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    

    
    def _send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send an email.
        
        :param to: Recipient email address
        :param subject: Email subject
        :param body: Email body content
        :return: Dictionary with send status
        """
        if not to or not subject or not body:
            return {
                "status": "error",
                "error": "To, subject, and body are required"
            }
            
        try:
            # Create a message
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            
            # Encode the message
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send the message
            sent_message = self.service.users().messages().send(
                userId='me',
                body={'raw': encoded_message}
            ).execute()
            
            return {
                "status": "success",
                "message_id": sent_message['id'],
                "to": to,
                "subject": subject
            }
            
        except HttpError as error:
            return {
                "status": "error",
                "error": f"HTTP Error: {error.reason}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _summarize_inbox(self, max_results=10, detailed=False):
        """
        Generate a readable summary of recent emails.
        
        Args:
            max_results: Maximum number of emails to include
            detailed: Whether to include more detailed formatting and analysis
            
        Returns:
            Dictionary with formatted email summary
        """
        # Get recent emails
        result = self._list_emails(max_results=max_results)
        
        if result.get("status") != "success":
            return result
            
        emails = result.get("emails", [])
        
        if not emails:
            return {"status": "success", "summary": "No recent emails found.", "complete_task": True, "is_final_result": True}
            
        # Format emails into a readable summary
        if detailed:
            # More detailed summary with categories
            formatted_summary = "## Your Email Summary\n\n"
            formatted_summary += f"### {len(emails)} Recent Messages\n\n"
            
            # Format by categories (could be extended with more analysis)
            important_emails = []
            other_emails = []
            
            for email in emails:
                sender = email.get("sender", "Unknown Sender")
                subject = email.get("subject", "(No Subject)")
                date = email.get("date", "Unknown Date")
                snippet = email.get("snippet", "")
                
                # Simple categorization logic
                email_summary = f"**From:** {sender}  \n**Subject:** {subject}  \n**Date:** {date}  \n**Preview:** {snippet}\n\n"
                
                # Very simple importance detection - could be made more sophisticated
                if any(term in subject.lower() for term in ["urgent", "important", "action", "required", "attention"]):
                    important_emails.append(email_summary)
                else:
                    other_emails.append(email_summary)
            
            # Add important emails section if any exist
            if important_emails:
                formatted_summary += f"### Important Messages ({len(important_emails)})\n\n"
                formatted_summary += "".join(important_emails)
            
            # Add other emails
            formatted_summary += f"### Other Messages ({len(other_emails)})\n\n"
            formatted_summary += "".join(other_emails)
            
            # Signal that this result completes the entire email task 
            return {
                "status": "success", 
                "summary": formatted_summary,
                "complete_task": True,  # Signal to orchestrator this completes the task
                "is_final_result": True  # Alternative signal for task completion
            }
        else:
            # Standard listing format
            email_list = []
            for email in emails:
                sender = email.get("sender", "Unknown Sender")
                subject = email.get("subject", "(No Subject)")
                date = email.get("date", "Unknown Date")
                snippet = email.get("snippet", "")
                
                email_list.append(
                    f"- **From:** {sender}\n  **Subject:** {subject}\n  **Date:** {date}\n  **Preview:** {snippet}"
                )
                
            summary = "\n\n".join(email_list)
            formatted_summary = f"## Recent emails in your inbox:\n\n{summary}"
            
            # Signal that this result completes the entire email task
            return {
                "status": "success", 
                "summary": formatted_summary,
                "complete_task": True,  # Signal to orchestrator this completes the task
                "is_final_result": True  # Alternative signal for task completion
            }
    

    
