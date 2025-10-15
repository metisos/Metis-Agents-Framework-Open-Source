import os
import sys
import json
import re
import uuid
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, make_response, send_from_directory, send_file, Response, stream_with_context
from flask_cors import CORS
from datetime import timedelta
import datetime
# Fix import path based on where the script is run from
try:
    from web.output_formatter import OutputFormatter
except ImportError:
    # If run from within the web directory
    from output_formatter import OutputFormatter, format_response_for_frontend

# Add parent directory to path to import Single Agent modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agent import SingleAgent
# Import logging utilities but don't rely on a specific function
import components.logging_utils

# Import the IntentRouter for classifying queries
from components.intent_router import IntentRouter

app = Flask(__name__, static_folder='.')


# Enable CORS for all routes to allow frontend access
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure session management with Flask's built-in session
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session expires after 7 days
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Recommended security setting
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS


# Load environment variables from .env.local

env_path = os.path.join(parent_dir, '..', '.env.local')
if os.path.exists(env_path):
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    print(f"Warning: Environment file {env_path} not found")
# Verify required environment variables
required_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY', 'E2B_ACCESS_TOKEN']
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    print(f"Warning: Missing required environment variables: {', '.join(missing_vars)}")
    print("Some functionality may not work correctly")
else:
    print("✅ All required environment variables found")

# Initialize the Single Agent and Intent Router
print("Initializing Single Agent...")
# Function to get or create a session ID for the current user
def get_session_id():
    if 'user_id' not in session:
        session['user_id'] = f"user_{uuid.uuid4().hex[:8]}"
    return session['user_id']

# Initialize with session support
agent = SingleAgent()
intent_router = IntentRouter()

# Initialize OutputCapture to store and retrieve task plans during execution
print("✅ Single Agent initialized successfully")
print("✅ Intent Router initialized successfully")

class StreamResponse:
    def __init__(self, response_type="update"):
        self.response_type = response_type

class OutputCapture:
    def __init__(self):
        # Dictionary to store session-specific data
        self.sessions = {}
        
    def get_session_data(self, session_id=None):
        """Get session-specific data, creating it if it doesn't exist"""
        if session_id is None:
            session_id = get_session_id()
            
        # Initialize session data if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'output': '',
                'tools_used': [],
                'task_plan': [],
                'formatted_response': {},
                'streaming': True,
                'updates': [],
                'query_intent': 'task',  # Default to task
                'current_query': None
            }
            
        return self.sessions[session_id]
        
    def reset(self, session_id=None):
        """Reset session-specific data"""
        if session_id is None:
            session_id = get_session_id()
            
        # If session exists, reset it; otherwise, it'll be created fresh next time it's accessed
        if session_id in self.sessions:
            self.sessions[session_id] = {
                'output': '',
                'tools_used': [],
                'task_plan': [],
                'formatted_response': {},
                'streaming': True,
                'updates': [],
                'query_intent': 'task',  # Default to task
                'current_query': None
            }
            
    def add_tool(self, tool_name, session_id=None):
        """Add a tool to the list of tools used for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        if tool_name not in session_data['tools_used']:
            session_data['tools_used'].append(tool_name)
            
    def set_task_plan(self, task_plan, session_id=None):
        """Store the task plan for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        session_data['task_plan'] = task_plan
        
    def get_task_plan(self, session_id=None):
        """Get the task plan for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        return session_data['task_plan']
        
    def set_current_query(self, query, session_id=None):
        """Set the current query for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        session_data['current_query'] = query
        
    def get_current_query(self, session_id=None):
        """Get the current query for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        return session_data['current_query']
        
    def set_query_intent(self, intent, session_id=None):
        """Set the query intent for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        session_data['query_intent'] = intent
        
    def get_query_intent(self, session_id=None):
        """Get the query intent for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        return session_data['query_intent']
        
    def get_tools_used(self, session_id=None):
        """Get the tools used for this session"""
        if session_id is None:
            session_id = get_session_id()
            
        session_data = self.get_session_data(session_id)
        return session_data['tools_used']

# Global output capture object
output_capture = OutputCapture()

# Removing HTML interface routes to avoid conflict with Next.js frontend
# The Flask server will now only handle API requests

@app.route('/')
def index():
    return jsonify({
        "status": "ok",
        "message": "Metis Agentic Orchestration System API is running",
        "frontend_note": "Please use the Next.js frontend to interact with this API"
    })

@app.route('/api/agent-identity', methods=['GET'])
def get_agent_identity():
    """Return the agent identity information including current timestamp"""
    try:
        # Get identity information from the agent
        identity = agent.get_agent_identity()
        
        # Update the current timestamp to ensure it's always fresh
        identity['current_time'] = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
        
        # Add uptime information
        uptime_info = {
            "started": agent.agent_creation_date,
            "uptime_hours": round((datetime.datetime.now() - datetime.datetime.strptime(agent.agent_creation_date, "%Y-%m-%d")).total_seconds() / 3600, 1)
        }
        identity['uptime'] = uptime_info
        
        return jsonify(identity)
    except Exception as e:
        print(f"Error getting agent identity: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a query from the user - simplified version without streaming."""
    try:
        # Get the query from the request
        data = request.json
        print(f"\nDEBUG - Received request data: {data}")
        
        query = data.get('query', '')
        
        # Extract user context information if provided
        user_context = data.get('user_context', None)
        if user_context:
            print(f"\nDEBUG - User context provided: {user_context}")
            
        # Fetch any knowledge base entries for this user
        knowledge_base_entries = data.get('knowledge_base', [])
        if knowledge_base_entries:
            print(f"DEBUG - Received {len(knowledge_base_entries)} knowledge base entries")
        
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'})
            
        print(f"\nDEBUG - Processing new query: '{query}' from API")
        
        try:
            # Extract conversation history if provided
            conversation_history = data.get('conversation_history', [])
            if conversation_history:
                print(f"DEBUG - Received {len(conversation_history)} conversation history messages")
                
            # Store knowledge base entries in session data so process_query_thread can access them
            if knowledge_base_entries:
                session_data = output_capture.get_session_data()
                session_data['kb_entries'] = knowledge_base_entries
                print(f"DEBUG - Stored {len(knowledge_base_entries)} KB entries in session data")
            
            # Process the query in a blocking way for simplicity and reliability
            # Pass user context and conversation history if available
            process_query_thread(query, user_context=user_context, conversation_history=conversation_history)
            
            # Get tool usage from the logs for this specific query
            tools_used = get_tools_used(query)
            print(f"DEBUG - Tools used: {tools_used}")
            
            # Get the current session ID
            session_id = get_session_id()
            session_data = output_capture.get_session_data(session_id)
            
            # Get the intent classification
            intent = output_capture.get_query_intent(session_id)
            print(f"DEBUG - Intent classification: {intent}")
            
            # Check if we have a task plan
            task_plan = output_capture.get_task_plan(session_id)
            print(f"DEBUG - Task plan: {len(task_plan)} items")
            
            # Use the formatted response if available, otherwise use the basic output
            # Prepare content output
            if session_data.get('formatted_response'):
                # Use formatted response
                response = session_data['formatted_response']
                # Add tools used and intent if not already there
                response['tools_used'] = tools_used
                response['intent'] = intent
                print(f"DEBUG - Using formatted response")
            else:
                # Fallback to basic output
                response = {
                    'content': session_data.get('output', 'No output generated.'),
                    'tools_used': tools_used,
                    'task_plan': task_plan if intent == "task" else [],
                    'intent': intent
                }
                print(f"DEBUG - Using basic response")
        
            print(f"DEBUG - Response keys: {response.keys()}")
            print(f"DEBUG - Content length: {len(response.get('content', ''))}")
            
            # Apply standardized formatting to ensure frontend compatibility
            formatted_response = format_response_for_frontend(response)
            print(f"DEBUG - Formatted response keys: {formatted_response.keys()}")
            
            return jsonify(formatted_response)
        except Exception as inner_error:
            print(f"INNER ERROR - Exception in query processing: {str(inner_error)}")
            import traceback
            print(traceback.format_exc())
            error_response = format_response_for_frontend({
                'content': f"Error processing query: {str(inner_error)}",
                'content_type': 'markdown',
                'intent': 'error'
            })
            return jsonify(error_response
            )
        # This return is unreachable but kept for completeness
        return jsonify({
                'content': f"An error occurred during processing: {str(inner_error)}",
                'tools_used': [],
                'task_plan': [],
                'intent': 'question'
            })
    except Exception as e:
        print(f"ERROR - Exception in /api/query: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'content': f"An error occurred: {str(e)}",
            'tools_used': [],
            'task_plan': []
        })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current status of query processing."""
    try:
        # Get the current session ID
        session_id = get_session_id()
        session_data = output_capture.get_session_data(session_id)
        
        # Get the current query
        current_query = output_capture.get_current_query(session_id)
            
        # Check if there is an active query being processed
        result = None
        if session_data['formatted_response']:
            result = session_data['formatted_response']
        
        # Get any new updates
        updates = session_data['updates'].copy()
        session_data['updates'] = []  # Clear updates after sending
        
        # Get tool usage from the logs for this specific query
        tools_used = get_tools_used(current_query)
        
        # Check if processing is complete
        # Determine status based on current_query and completion flag
        # current_query is fetched a few lines above
        if not current_query: 
            response_status = 'idle'
            is_complete = False # No active query, so not 'complete' in the sense of a task finishing
        else:
            is_complete = session_data.get('complete', False) 
            response_status = 'complete' if is_complete else 'processing'
        
        # Include the intent classification
        intent = output_capture.get_query_intent(session_id)
        
        # Compile response
        response = {
            'status': response_status,
            'updates': updates,
            'tools_used': tools_used,
            'task_plan': output_capture.get_task_plan(session_id) if intent == "task" else [],
            'intent': intent,
            'result': result
        }
        
        # Add final result if complete
        if is_complete:
            if output_capture.formatted_response:
                response['result'] = output_capture.formatted_response
            else:
                response['result'] = {
                    'content': output_capture.output,
                    'task_plan': output_capture.task_plan
                }
                
        # Apply standardized formatting to ensure frontend compatibility
        formatted_response = format_response_for_frontend(response)
        
        # Return the formatted response
        return jsonify(formatted_response)
    except Exception as e:
        return jsonify({'error': str(e)})

def check_and_update_task_plan(session_id=None):
    """Check if there are updated tasks in the file and send updates if streaming."""
    if session_id is None:
        session_id = get_session_id()
        
    session_data = output_capture.get_session_data(session_id)
    if not session_data['streaming']:
        return
        
    try:
        task_file = Path(parent_dir) / "tasks.md"
        if task_file.exists():
            with open(task_file, 'r', encoding='utf-8', errors='replace') as f:
                task_content = f.read()
                
                # If the file is empty or doesn't contain task markers, return early
                if not task_content or '- [' not in task_content:
                    print("DEBUG - No tasks found in tasks.md")
                    return
                    
                # Extract tasks and their status
                new_tasks = []
                for line in task_content.split('\n'):
                    if line.strip().startswith('- ['):  # Task line
                        status = 'completed' if '[x]' in line else 'pending'
                        # Make sure the line has the correct format before splitting
                        if ']' in line:
                            task_text = line.split(']', 1)[1].strip()
                            new_tasks.append({'text': task_text, 'status': status})
                
                # Get the current task plan for this session
                current_task_plan = output_capture.get_task_plan(session_id)
                        
                # Find new completed tasks
                if len(new_tasks) > len(current_task_plan):
                    # New tasks added
                    added_tasks = new_tasks[len(current_task_plan):]
                    for task in added_tasks:
                        session_data['updates'].append({
                            'type': 'task_added',
                            'task': task
                        })
                
                # Check for status changes in existing tasks
                for i, task in enumerate(current_task_plan):
                    if i < len(new_tasks) and task['status'] != new_tasks[i]['status']:
                        session_data['updates'].append({
                            'type': 'task_updated',
                            'task': new_tasks[i]
                        })
                
                # Update the task plan
                output_capture.set_task_plan(new_tasks, session_id)
    except Exception as e:
        print(f"Error checking task plan: {e}")

def process_query_thread(query, session_id=None, user_context=None, conversation_history=None):
    """Process a query from the user in a separate thread."""
    try:
        # Get or create session ID
        if session_id is None:
            session_id = get_session_id()
            
        # Reset the output capture for a new query
        output_capture.reset(session_id)
        
        # Store the current query for tool tracking
        output_capture.set_current_query(query, session_id)
        
        # Store user context if provided
        if user_context:
            print(f"Processing query with user context: {user_context}")
            # Add the user context as a session data item
            session_data = output_capture.get_session_data(session_id)
            session_data['user_context'] = user_context
        
        # Classify the intent of the query (question or task)
        intent = intent_router.classify(query)
        print(f"\nDEBUG - Query classified as: {intent}")
        
        # Force task intent for queries that look like they're asking for a report or document
        if any(keyword in query.lower() for keyword in ['report', 'document', 'paper', 'analysis', 'write', 'create', 'generate']):
            intent = "task"
            print(f"DEBUG - Overriding intent to 'task' based on keywords")
        
        # Store the intent in the output capture for the frontend
        output_capture.set_query_intent(intent, session_id)
        
        # Call the SingleAgent to process the query
        print(f"DEBUG - Processing query with agent, session_id: {session_id}")
        
        # Store the original query without any augmentation for task tracking
        original_query = query
        
        # Prepare to augment the query with user context if available
        augmented_query = query
        
        # 1. Only add user context for questions, not for tasks
        if user_context:
            # Format user context into a string the AI can use
            context_parts = []
            if user_context.get('name'):
                context_parts.append(f"Name: {user_context['name']}")
            if user_context.get('profession'):
                context_parts.append(f"Profession: {user_context['profession']}")
            if user_context.get('location'):
                context_parts.append(f"Location: {user_context['location']}")
            if user_context.get('interests'):       
                interests = user_context['interests']
                if isinstance(interests, list):
                    context_parts.append(f"Interests: {', '.join(interests)}")
            if user_context.get('context'):
                context_parts.append(f"Additional Context: {user_context['context']}")
            
            # Store user context in session data
            session_data = output_capture.get_session_data(session_id)
            session_data['user_context_parts'] = context_parts
            
            # Only add user context for non-task queries (questions)
            if intent == "question":
                user_context_str = "\n\nUSER CONTEXT:\n" + "\n".join(context_parts)
                augmented_query = f"{query}\n\n{user_context_str}"
                print(f"DEBUG - Using augmented query with user context for question: {augmented_query[:100]}...")
            else:
                print(f"DEBUG - Skipping user context for task query")
        
        # 2. Add conversation history if available (for both questions and tasks)
        if conversation_history and isinstance(conversation_history, list):
            # Limit the number of messages to prevent token overload (take last 5 messages)
            MAX_HISTORY_MESSAGES = 5
            limited_history = conversation_history[-MAX_HISTORY_MESSAGES:] if len(conversation_history) > MAX_HISTORY_MESSAGES else conversation_history
            
            # Store the limited conversation history in session data
            session_data = output_capture.get_session_data(session_id)
            session_data['conversation_history'] = limited_history
            
            # Format the conversation history with character limits per message
            MAX_MESSAGE_LENGTH = 500  # Characters per message
            history_context = "\n\nCONVERSATION HISTORY:\n"
            for msg in limited_history:
                sender = msg.get('sender', 'unknown')
                content = msg.get('content', '')
                
                # Truncate long messages
                if len(content) > MAX_MESSAGE_LENGTH:
                    content = content[:MAX_MESSAGE_LENGTH] + "... [truncated]"
                    
                formatted_sender = "User" if sender == "user" else "Assistant"
                history_context += f"\n{formatted_sender}: {content}\n"
            
            # Add to the augmented query
            augmented_query = f"{augmented_query}\n{history_context}"
            print(f"DEBUG - Added {len(conversation_history)} messages of conversation history")
        
        # 3. Add knowledge base entries if available (only for questions, not tasks)
        try:
            # Check if there are any knowledge base entries in the session data
            session_data = output_capture.get_session_data(session_id)
            kb_entries = session_data.get('kb_entries', [])
            if kb_entries:
                    # Store KB entries in session data
                    session_data = output_capture.get_session_data(session_id)
                    session_data['kb_entries'] = kb_entries
                    
                    # Only add KB entries for questions, not for tasks
                    if intent == "question":
                        kb_context = "\n\nKNOWLEDGE BASE ENTRIES:\n"
                        for idx, entry in enumerate(kb_entries, 1):
                            kb_context += f"\nEntry {idx}: {entry['title']}\n"
                            kb_context += f"Tags: {', '.join(entry.get('tags', []))}\n"
                            kb_context += f"Content: {entry['content']}\n"
                        
                        augmented_query = f"{augmented_query}\n{kb_context}"
                        print(f"DEBUG - Added {len(kb_entries)} knowledge base entries to question context")
                    else:
                        print(f"DEBUG - Skipping knowledge base entries for task query")
        except Exception as e:
            print(f"ERROR - Failed to add knowledge base context: {e}")
        
        # Call the agent to process the query
        start_time = time.time()
        
        # For task tracking, use the original query without context appended
        # For LLM processing, use the augmented query with context for better responses
        
        # Store the clean original query in session data for task viewing
        session_data = output_capture.get_session_data(session_id)
        session_data['original_query'] = original_query
        
        # When we pass to the agent, use the augmented query with context
        # but add a flag indicating the real task portion
        agent_response = agent.process_query(augmented_query, session_id)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DEBUG - Raw result type: {type(agent_response)}")
        # Print truncated result to avoid overwhelming logs
        result_str = str(agent_response)[:300]
        print(f"DEBUG - Raw result preview: {result_str}...\n")
        
        # Use the dedicated formatter to process the output
        print("DEBUG - Using OutputFormatter to parse result")
        formatted_result = OutputFormatter.format_agent_response(agent_response)
        
        # Extra debugging to check if we found content
        content = formatted_result.get('content', '')
        if content and len(content) > 200:
            print(f"DEBUG - Successfully extracted substantial content: {len(content)} chars")
            print(f"DEBUG - Content begins with: {content[:100]}...")
        else:
            print(f"DEBUG - Warning: Extracted content is short or empty: {len(content)} chars")
        
        # Get the session data
        session_data = output_capture.get_session_data(session_id)
        
        # Store the formatted result
        session_data['formatted_response'] = formatted_result
        session_data['output'] = content
        
        # If task plan was extracted by the formatter, use it
        if formatted_result.get('task_plan'):
            output_capture.set_task_plan(formatted_result['task_plan'], session_id)
            print(f"DEBUG - Task plan extracted by formatter: {len(formatted_result['task_plan'])} items")
        else:
            # Otherwise try to read it from the tasks file
            try:
                task_file = Path(parent_dir) / "tasks.md"
                if task_file.exists():
                    with open(task_file, 'r') as f:
                        content = f.read()
                        if content:  # Skip empty file
                            # Extract tasks and their status
                            tasks = []
                            for line in content.split('\n'):
                                if line.strip().startswith('- ['):  # Task line
                                    status = 'completed' if '[x]' in line else 'pending'
                                    task_text = line.split(']', 1)[1].strip()
                                    tasks.append({'text': task_text, 'status': status})
                            
                            if tasks:  # Only update if we found tasks
                                output_capture.set_task_plan(tasks, session_id)
                                print(f"DEBUG - Task plan read from file: {len(tasks)} items")
            except Exception as e:
                print(f"Error reading task plan: {e}")
                # Don't throw here - we can continue without a task plan
        
        # Output some debug info about the result
        print(f"DEBUG - Formatted result keys: {formatted_result.keys()}")
        print(f"DEBUG - Content length: {len(content)}")
        
        # Mark processing as complete
        session_data['complete'] = True
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        # Get the session data
        session_data = output_capture.get_session_data(session_id)
        session_data['output'] = f"Error processing query: {str(e)}"
        session_data['complete'] = True

# The recursive content finding is now handled by OutputFormatter

def get_tools_used(current_query=None, session_id=None):
    """Extract the tools used from the log entries relevant to the current query."""
    try:
        # If no query was provided but session_id was, get the current query from the session
        if current_query is None and session_id is not None:
            current_query = output_capture.get_current_query(session_id)
        elif session_id is None:
            session_id = get_session_id()
            
        # Look for the latest tool usage log
        logs_dir = Path(parent_dir) / "logs"
        latest_log = logs_dir / "tool_usage.md"
        
        if not latest_log.exists():
            return []
        
        # Read the latest tool usage log
        with open(latest_log, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        
        # Extract tool names from the log
        tools = []
        sections = content.split('## ')
        
        if len(sections) > 1:
            # We'll process all sections to find matches for the current query
            for entry in sections[1:]:  # Skip the first section (it's the header)
                lines = entry.strip().split('\n')
                if len(lines) < 3:  # Need at least header, query line, and something else
                    continue
                
                # Extract the tool name from the header
                tool_header = lines[0]
                if ' - ' not in tool_header:
                    continue
                    
                tool_name = tool_header.split(' - ', 1)[1].strip()
                
                # Extract the query this tool was used for
                query_line = None
                for line in lines[1:4]:  # Look at the first few lines for the query
                    if line.strip().startswith('**Query:**'):
                        query_line = line
                        break
                        
                if not query_line:
                    continue
                    
                query_text = query_line.replace('**Query:**', '').strip()
                
                # If current_query is provided, only include tools for this query
                if current_query and current_query.lower().strip() != query_text.lower().strip():
                    continue
                    
                # Special handling for common tools that might not be as interesting to the user
                if tool_name in ['ToolSelector', 'TaskScheduler']:
                    continue  # Skip internal tools
                    
                # Make nicer display names
                display_name = tool_name
                if tool_name.endswith('Tool'):
                    display_name = tool_name[:-4]  # Remove 'Tool' suffix
                    
                if display_name and display_name not in tools:
                    tools.append(display_name)
        
        return tools
    
    except Exception as e:
        print(f"Error getting tools used: {str(e)}")
        return []

# Knowledge Base API endpoint
@app.route('/api/knowledge', methods=['POST'])
def knowledge_base_api():
    """API endpoint for knowledge base operations"""
    try:
        data = request.json
        operation = data.get('operation', 'retrieve')
        user_id = data.get('user_id')
        
        print(f"\nDEBUG - Knowledge Base API: {operation} for user {user_id}")
        
        if operation == 'retrieve':
            # For now, just echo back the request to confirm it works
            return jsonify({
                'status': 'success',
                'message': 'Knowledge base API is working',
                'user_id': user_id
            })
        elif operation == 'create':
            # Handle creating a new knowledge entry (AI contribution)
            entry_data = data.get('entry', {})
            print(f"DEBUG - Creating knowledge entry: {entry_data}")
            
            # In a real implementation, this would store to the database
            # For now, we'll just echo success
            return jsonify({
                'status': 'success',
                'message': 'Knowledge entry created',
                'entry_id': f"kb_{int(time.time())}"
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unknown operation'
            }), 400
    except Exception as e:
        print(f"Error in knowledge base API: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting web interface on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    # Explicitly bind to all interfaces (0.0.0.0) and ensure the port is 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
