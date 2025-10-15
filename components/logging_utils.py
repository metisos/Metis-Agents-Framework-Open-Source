"""
Logging utilities for the Single Agent system.
Provides a consistent logging format and functionality.
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def get_tool_logger(base_dir=None):
    """
    Get a logger specifically for tool usage.
    
    Args:
        base_dir: Base directory for log files (optional)
        
    Returns:
        Logger instance for tool usage
    """
    if not base_dir:
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    log_dir = base_dir / "logs"
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"tool_usage_{today}.log"
    
    return setup_logger("ToolUsage", log_file)

def log_tool_usage(tool_name, task, result_summary, query=None):
    """
    Log tool usage with task details.
    
    Args:
        tool_name: Name of the tool
        task: Task the tool was used for
        result_summary: Brief summary of the result
        query: Original user query (optional)
    """
    logger = get_tool_logger()
    logger.info(f"Tool: {tool_name} | Task: {task} | Result: {result_summary}")
    
    # Also log to markdown file for easy viewing
    log_tool_to_markdown(tool_name, task, result_summary, query)
    

def log_tool_selection(task, selected_tool_name, method, candidates=None, query=None):
    """
    Log the tool selection process.
    
    Args:
        task: The task for which a tool was selected
        selected_tool_name: Name of the selected tool (or None if no tool was found)
        method: Method used for selection (LLM or pattern_matching)
        candidates: List of candidate tools considered (optional)
        query: Original user query (optional)
    """
    logger = get_tool_logger()
    selection_info = f"Selection method: {method} | Task: {task} | Selected tool: {selected_tool_name or 'None'}"
    
    if candidates:
        selection_info += f" | Candidates: {candidates}"
        
    logger.info(selection_info)
    
    # Format for markdown
    tool_name = "ToolSelector"
    result_summary = f"Selected {selected_tool_name or 'None'} using {method} method"
    
    # Also log to markdown file for easy viewing
    log_tool_to_markdown(tool_name, task, result_summary, query)

def log_tool_to_markdown(tool_name, task, result_summary, query=None):
    """
    Log tool usage to a markdown file in a timestamped folder.
    
    Args:
        tool_name: Name of the tool
        task: Task the tool was used for
        result_summary: Brief summary of the result
        query: Original user query (optional)
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = base_dir / "logs"
    
    # Create a session-based directory using date for organization
    today = datetime.now().strftime("%Y-%m-%d")
    session_time = datetime.now().strftime("%H-%M-%S")
    session_dir = logs_dir / today
    
    # Create all necessary directories
    os.makedirs(session_dir, exist_ok=True)
    
    # Create a unique session file with timestamp
    # If the file exists, append to it; otherwise create a new one
    latest_file = logs_dir / "tool_usage.md"  # Always keep a "latest" file
    session_file = session_dir / f"tool_usage_{session_time}.md"
    
    # If the session file doesn't exist yet, create it with a header
    if not os.path.exists(session_file):
        with open(session_file, "w") as f:
            f.write("# Tool Usage Log\n\n")
            f.write(f"Session started at: {today} {session_time}\n\n")
            f.write("This file tracks tools used by the Single Agent.\n\n")
    
    # Get current timestamp for the log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the log entry
    entry = f"## {timestamp} - {tool_name}\n\n"
    
    if query:
        entry += f"**Query:** {query}\n\n"
        
    entry += f"**Task:** {task}\n\n"
    entry += f"**Result:** {result_summary}\n\n"
    entry += "---\n\n"
    
    # Append to session file
    with open(session_file, "a") as f:
        f.write(entry)
    
    # Also update the latest file for easy access
    # Create file with header if it doesn't exist
    if not os.path.exists(latest_file):
        with open(latest_file, "w") as f:
            f.write("# Tool Usage Log\n\n")
            f.write("This file tracks tools used by the Single Agent.\n\n")
        
    # Always update the latest file
    with open(latest_file, "a") as f:
        f.write(entry)
