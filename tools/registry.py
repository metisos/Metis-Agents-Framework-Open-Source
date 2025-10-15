"""
Tool Registry for Metis System

This module provides a registry for all available tools in the system.
"""

from typing import Dict, Type, Any, Optional, List
import os
import sys

# Add components directory to path if needed
components_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'components')
if components_path not in sys.path:
    sys.path.append(components_path)

# Skills system has been removed

# Use local imports instead of agentic_system
from .e2b_tool import E2BTool
from .math_tool import MathTool
from .google_search_tool import GoogleSearchTool
from .firecrawl_tool import FirecrawlTool
from .pdf_tool import PDFTool
from .code_generation_tool import CodeGenerationTool
from .gmail_tool import GmailTool

# OutputFormatterAgent doesn't exist in the single_agent implementation
# Removing: from agentic_system.agents.output_formatter_agent import OutputFormatterAgent

# Registry of available tools with their class types
TOOL_REGISTRY = {
    "E2BTool": E2BTool,
    "MathTool": MathTool,
    "GoogleSearchTool": GoogleSearchTool,
    "FirecrawlTool": FirecrawlTool,
    "PDFTool": PDFTool,
    "CodeGenerationTool": CodeGenerationTool,
    "GmailTool": GmailTool,
    # "output_formatter": OutputFormatterAgent, # Removed, not available in single_agent
    # Add other tools here as they are created
}

def get_available_tools() -> Dict[str, str]:
    """
    Get all available tools with their descriptions.
    
    :return: Dictionary mapping tool names to descriptions
    """
    tools = {}
    for name, tool_class in TOOL_REGISTRY.items():
        # Create an instance to get the description
        try:
            tool = tool_class()
            tools[name] = tool.get_description()
        except Exception as e:
            tools[name] = f"Error initializing tool: {str(e)}"
    return tools

def get_tool_instance(name: str, **kwargs) -> Optional[Any]:
    """
    Get an instance of a specific tool.
    
    :param name: Name of the tool
    :param kwargs: Additional parameters for tool initialization
    :return: Tool instance or None if not found
    """
    if name not in TOOL_REGISTRY:
        return None
        
    tool_class = TOOL_REGISTRY[name]
    try:
        return tool_class(**kwargs)
    except Exception as e:
        print(f"Error initializing tool '{name}': {str(e)}")
        return None

def initialize_tools() -> Dict[str, Any]:
    """
    Initialize all registered tools with default settings.
    
    :return: Dictionary of initialized tool instances
    """
    tools = {}
    for name, tool_class in TOOL_REGISTRY.items():
        try:
            tools[name] = tool_class()
            print(f"Initialized tool: {name}")
        except Exception as e:
            print(f"Failed to initialize tool '{name}': {str(e)}")
    
    # Skills system has been removed
    return tools


# Skills system has been removed
