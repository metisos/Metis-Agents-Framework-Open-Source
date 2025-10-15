import os
import time
from datetime import datetime

class TaskManager:
    """
    Manages tasks in a markdown file, providing methods to add tasks, 
    mark them as complete, and retrieve their status.
    """
    def __init__(self, task_file):
        self.task_file = task_file
        
        # Track current session tasks
        self.current_session_tasks = []
        self.completed_tasks = []
        self.session_id = int(time.time())  # Use timestamp as session ID
        
        # Create the file if it doesn't exist
        if not os.path.exists(self.task_file):
            with open(self.task_file, "w") as f:
                f.write("# Task Checklist\n\n")
                f.write("This file tracks tasks created and completed by the Single Agent.\n\n")
    
    def start_new_task_within_session(self, query: str = None):
        """Prepare for a new task within the current session.
        
        This ensures completed tasks are properly archived but maintains session context.
        
        Args:
            query: The user query for the new task
        """
        print(f"TASK MANAGER: Starting new task within session: {query[:50] if query else 'No query'}...")
        
        # Archive completed tasks but maintain session ID
        if self.completed_tasks:
            print(f"Archiving {len(self.completed_tasks)} completed tasks from previous request")
            
        # Reset pending tasks while keeping session alive
        self.current_session_tasks = []
        
        return {
            "status": "success",
            "message": "Prepared for new task within current session",
            "session_id": self.session_id
        }
    
    def add_tasks(self, tasks: list, original_query: str = None):
        """
        Add new tasks to the task file with timestamps and query context.
        Preserves session context while adding new task plan for the current task.
        
        Args:
            tasks: List of task descriptions to add
            original_query: The original user query that generated these tasks
        """
        query_info = original_query[:50] + "..." if original_query and len(original_query) > 50 else original_query
        print(f"TASK MANAGER: Adding new task plan for: {query_info}")
        
        # Clear current task list without losing session context
        self.current_session_tasks = list(tasks)  # Replace with new tasks while maintaining session
        
        # Generate a new session ID to ensure proper separation
        self.session_id = int(time.time())  # Use timestamp as session ID
        
        # Now add the new tasks
        
        # Read existing content
        try:
            with open(self.task_file, "r") as f:
                content = f.read()
        except FileNotFoundError:
            content = "# Task Checklist\n\nThis file tracks tasks created and completed by the Single Agent.\n\n"
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append new query section
        with open(self.task_file, "w") as f:
            f.write(content)
            
            # Add query section if provided
            if original_query:
                f.write(f"\n## Query: {original_query}\n")
                f.write(f"*Created at: {timestamp}* | Session ID: {self.session_id}\n\n")
            else:
                f.write(f"\n## New Tasks - {timestamp}\n")
                f.write(f"*Session ID: {self.session_id}*\n\n")
            
            # Add tasks with checkbox format
            for task in tasks:
                f.write(f"- [ ] {task}\n")
    
    def mark_complete(self, task: str):
        """
        Mark a task as complete in the task file with completion timestamp.
        Also track it in the current session's completed tasks.
        
        Args:
            task: The task description to mark as complete
        """
        # Update in-memory tracking
        if task in self.current_session_tasks and task not in self.completed_tasks:
            self.completed_tasks.append(task)
            
        try:
            with open(self.task_file, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.task_file, "w") as f:
            for line in lines:
                if task in line and "- [ ]" in line:
                    # Mark as complete and add completion timestamp
                    f.write(line.replace("- [ ]", f"- [x]") + f"    *(Completed at: {timestamp} | Session: {self.session_id})*\n")
                else:
                    f.write(line)
    
    def get_all_tasks(self):
        """
        Retrieve all tasks and their status from the task file and in-memory tracking.
        Only shows pending tasks from the current session.
        
        Returns:
            Dictionary with lists of completed and pending tasks
        """
        # Use in-memory tracking for current session tasks
        # Make a copy of current completed tasks to avoid modifying the list during iteration
        completed_current_session = list(self.completed_tasks)  
        
        # Determine genuinely pending tasks (in current session but not yet completed)
        pending_current_session = [task for task in self.current_session_tasks 
                                if task not in completed_current_session]
        
        # Get historical completed tasks from file for display
        completed_from_file = []
        try:
            with open(self.task_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("- [x]"):
                        task_text = line[5:].split("*(Completed")[0].strip()
                        if task_text not in completed_from_file:
                            completed_from_file.append(task_text)
        except FileNotFoundError:
            pass
            
        # Combine current session completed tasks with historical ones for display
        # But avoid duplicates
        all_completed = completed_current_session.copy()
        for task in completed_from_file:
            if task not in all_completed:
                all_completed.append(task)
        
        return {
            "completed": all_completed,
            "pending": pending_current_session  # Only show pending tasks from current session
        }
