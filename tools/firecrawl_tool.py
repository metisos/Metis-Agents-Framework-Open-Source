"""
Firecrawl Tool for Metis System

This tool provides integration with Firecrawl API for advanced web scraping, crawling,
and interactive page actions.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Union

# Import pydantic with graceful fallback
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create minimal BaseModel substitute
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Create minimal Field substitute
    def Field(*args, **kwargs):
        return None
# Initialize logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("firecrawl_tool")

try:
    from firecrawl import FirecrawlApp, ScrapeOptions, JsonConfig
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    print("Firecrawl package not found. Install with: pip install firecrawl-py")

class FirecrawlTool:
    """Tool for performing web scraping and crawling using Firecrawl API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl tool.
        
        :param api_key: Firecrawl API key (optional, defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "fc-733f382e159945889f22c68f213ac8e4")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable or api_key parameter must be set")
        
        if not FIRECRAWL_AVAILABLE:
            raise ImportError("Firecrawl package not installed. Install with: pip install firecrawl-py")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        self.logger = logger
        
    def get_description(self) -> str:
        """
        Return a description of what the tool does.
        
        :return: Tool description string
        """
        return "Scrapes and extracts data from websites using Firecrawl API, allowing for crawling, interactive page actions, and structured data extraction"
        
    def can_handle(self, task: str) -> bool:
        """
        Determine if this tool can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            True if this tool can handle the task, False otherwise
        """
        # Check for web scraping related keywords
        scrape_keywords = [
            'scrape', 'crawl', 'extract from website', 'extract data', 
            'web data', 'website content', 'get from website',
            'scrape website', 'visit website', 'extract information',
            'webpage data', 'website information'
        ]
        
        task_lower = task.lower()
        
        # Check for keywords
        if any(keyword in task_lower for keyword in scrape_keywords):
            return True
            
        # Look for URL patterns
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+\.[a-z]{2,}',
            r'scrape [^\s]+\.[a-z]{2,}',
            r'extract [^\s]+\.[a-z]{2,}'
        ]
        
        for pattern in url_patterns:
            if re.search(pattern, task):
                return True
        
        return False
        
    def execute(self, task: str) -> str:
        """
        Execute web scraping or crawling based on the task description.
        
        Args:
            task: The task description
            
        Returns:
            Extracted content from the website
        """
        self.logger.info(f"FirecrawlTool executing task: {task}")
        
        # Extract URL from task
        url = self._extract_url(task)
        if not url:
            return "Error: Could not find a valid URL in the task. Please provide a URL to scrape."
            
        try:
            # Default options
            options = ScrapeOptions(
                wait_for_selector="body",
                wait_time=3,
                extract_text=True,
                extract_links=True,
                extract_metadata=True
            )
            
            # Check if we need to handle specific elements or extraction
            task_lower = task.lower()
            
            # Handle tables
            if 'table' in task_lower:
                options.extract_tables = True
                
            # Handle images
            if 'image' in task_lower or 'picture' in task_lower:
                options.extract_images = True
                
            # Handle pagination
            if 'pagination' in task_lower or 'multiple pages' in task_lower:
                options.follow_pagination = True
                options.max_pages = 3  # Default limit for pagination
            
            # Execute scraping
            self.logger.info(f"Starting scrape for URL: {url}")
            result = self.app.scrape(url, options=options)
            
            if not result or not result.content:
                return f"No content was extracted from {url}. The site may be protected or require JavaScript."
            
            # Format output
            output = ["## Extracted Content from Website"]
            
            # Add page title and URL
            if result.metadata and result.metadata.title:
                output.append(f"**Title**: {result.metadata.title}")
            output.append(f"**URL**: {url}")
            
            # Add main content
            if result.content:
                content_preview = result.content[:1500] + '...' if len(result.content) > 1500 else result.content
                output.append("\n### Main Content\n")
                output.append(content_preview)
            
            # Add extracted links if available
            if result.links and len(result.links) > 0:
                output.append("\n### Links Found\n")
                for i, link in enumerate(result.links[:10]):  # Limit to 10 links
                    output.append(f"{i+1}. [{link.text or 'Link'}]({link.url})")
                if len(result.links) > 10:
                    output.append(f"... and {len(result.links)-10} more links")
                    
            return "\n\n".join(output)
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return f"Error scraping {url}: {str(e)}"
            
    def _extract_url(self, task: str) -> str:
        """
        Extract a URL from the task description.
        
        Args:
            task: The task description
            
        Returns:
            The extracted URL or empty string if not found
        """
        # Try to match URLs
        url_patterns = [
            r'https?://[^\s]+',  # http:// or https:// followed by anything
            r'www\.[^\s]+\.[a-z]{2,}'  # www. followed by domain
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, task)
            if match:
                url = match.group(0)
                
                # Clean up URL if needed (remove trailing punctuation)
                if url[-1] in '.,;:!?':
                    url = url[:-1]
                    
                # Ensure URL has protocol
                if not url.startswith('http'):
                    url = 'https://' + url
                    
                return url
                
        # Try to extract domain names with common TLDs
        domain_pattern = r'\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,}\.[a-zA-Z]{2,})\b|\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})\b'
        match = re.search(domain_pattern, task)
        if match:
            domain = match.group(0)
            return f"https://{domain}"
            
        return ""
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return parameter specifications for this tool.
        
        :return: Dictionary of parameter specifications
        """
        return {
            "url": {
                "type": "string",
                "description": "URL to scrape or crawl"
            },
            "formats": {
                "type": "array",
                "description": "Output formats (markdown, html, json)",
                "default": ["markdown"]
            },
            "method": {
                "type": "string",
                "description": "Method to use (scrape, crawl, extract)",
                "default": "scrape"
            },
            "extraction_prompt": {
                "type": "string",
                "description": "Prompt for extracting structured data (if method is extract)",
                "default": None
            },
            "actions": {
                "type": "array",
                "description": "List of actions to perform on the page before scraping",
                "default": None
            },
            "crawl_limit": {
                "type": "integer",
                "description": "Maximum number of pages to crawl (if method is crawl)",
                "default": 10
            }
        }
    
    def scrape_url(self, url: str, formats: List[str] = None, actions: List[Dict] = None) -> Dict[str, Any]:
        """
        Scrape content from a single URL.
        
        :param url: Target URL to scrape
        :param formats: Output formats (markdown, html, json)
        :param actions: List of actions to perform before scraping
        :return: Dictionary with scraped content
        """
        try:
            formats = formats or ["markdown"]
            
            scrape_result = self.app.scrape_url(
                url,
                formats=formats,
                actions=actions
            )
            
            # Convert the response to a dictionary for easier handling
            result_dict = {}
            
            if hasattr(scrape_result, "markdown"):
                result_dict["markdown"] = scrape_result.markdown
            
            if hasattr(scrape_result, "html"):
                result_dict["html"] = scrape_result.html
                
            if hasattr(scrape_result, "json"):
                result_dict["json"] = scrape_result.json
                
            if hasattr(scrape_result, "metadata"):
                result_dict["metadata"] = scrape_result.metadata
            else:
                result_dict["metadata"] = {}
            
            return {
                "status": "success",
                "url": url,
                "content": result_dict
            }
            
        except Exception as e:
            return {
                "status": "error",
                "url": url,
                "error": str(e)
            }
    
    def crawl_url(self, url: str, limit: int = 10, formats: List[str] = None) -> Dict[str, Any]:
        """
        Crawl a website and retrieve content from multiple pages.
        
        :param url: Starting URL for crawling
        :param limit: Maximum number of pages to crawl
        :param formats: Output formats (markdown, html, json)
        :return: Dictionary with crawled content
        """
        try:
            formats = formats or ["markdown"]
            
            # Initialize the crawl
            crawl_status = self.app.crawl_url(
                url,
                limit=limit,
                scrape_options=ScrapeOptions(formats=formats),
                poll_interval=10  # Check progress every 10 seconds
            )
            
            # Process the results - handle both dictionary and object responses
            pages = []
            
            # Handle different response formats
            if hasattr(crawl_status, "data"):
                pages = crawl_status.data
            elif isinstance(crawl_status, dict) and "data" in crawl_status:
                pages = crawl_status["data"]
            
            # Convert pages to a standard format if needed
            processed_pages = []
            for page in pages:
                page_dict = {}
                if hasattr(page, "url"):
                    page_dict["url"] = page.url
                elif isinstance(page, dict) and "url" in page:
                    page_dict["url"] = page["url"]
                else:
                    page_dict["url"] = "Unknown URL"
                    
                # Handle content based on format
                for fmt in formats:
                    if hasattr(page, fmt):
                        page_dict[fmt] = getattr(page, fmt)
                    elif isinstance(page, dict) and fmt in page:
                        page_dict[fmt] = page[fmt]
                        
                # Add metadata if available
                if hasattr(page, "metadata"):
                    page_dict["metadata"] = page.metadata
                elif isinstance(page, dict) and "metadata" in page:
                    page_dict["metadata"] = page["metadata"]
                
                processed_pages.append(page_dict)
            
            return {
                "status": "success",
                "url": url,
                "pages_count": len(processed_pages),
                "pages": processed_pages
            }
            
        except Exception as e:
            return {
                "status": "error",
                "url": url,
                "error": str(e)
            }
    
    def extract_data(self, url: str, extraction_prompt: str) -> Dict[str, Any]:
        """
        Extract structured data from a URL using a natural language prompt.
        
        :param url: Target URL to extract data from
        :param extraction_prompt: Natural language prompt for extraction
        :return: Dictionary with extracted data
        """
        try:
            # For JSON extraction without a schema, simply use a prompt-based approach
            json_config = {
                "prompt": extraction_prompt,
                "mode": "simple-extraction",  # Use simple extraction without schema
                "pageOptions": {"onlyMainContent": True}
            }
            
            try:
                scrape_result = self.app.scrape_url(
                    url,
                    formats=["json"],
                    json_options=json_config
                )
                
                # Handle the response based on the actual object structure
                extracted_data = {}
                metadata = {}
                
                if hasattr(scrape_result, "json"):
                    extracted_data = scrape_result.json
                
                if hasattr(scrape_result, "metadata"):
                    metadata = scrape_result.metadata
                
                return {
                    "status": "success",
                    "url": url,
                    "data": extracted_data,
                    "metadata": metadata
                }
            except Exception as inner_e:
                # If the extraction fails, try a simpler approach by scraping and then 
                # processing content directly
                print(f"Warning: Error in structured extraction: {inner_e}, falling back to content extraction")
                
                # Get the content in markdown format
                markdown_result = self.scrape_url(url, formats=["markdown"])
                
                # Simulate extracted data with the content we have
                content = markdown_result.get("content", {})
                markdown_text = content.get("markdown", "")
                
                # Create a simplified extraction result
                extracted_content = {
                    "extracted_content": markdown_text[:1000] if markdown_text else "",
                    "extraction_prompt": extraction_prompt,
                    "note": "This is a simplified extraction due to API limitations."
                }
                
                return {
                    "status": "success",
                    "url": url,
                    "data": extracted_content,
                    "metadata": content.get("metadata", {})
                }
            
        except Exception as e:
            return {
                "status": "error",
                "url": url,
                "error": str(e)
            }
    
    def run(self, url: str, method: str = "scrape", formats: List[str] = None, 
            extraction_prompt: str = None, actions: List[Dict] = None, 
            crawl_limit: int = 10) -> Dict[str, Any]:
        """
        Main method to run the tool with different modes.
        
        :param url: Target URL
        :param method: Method to use (scrape, crawl, extract)
        :param formats: Output formats
        :param extraction_prompt: Prompt for extraction
        :param actions: List of actions for interactive scraping
        :param crawl_limit: Maximum pages to crawl
        :return: Dictionary with results
        """
        formats = formats or ["markdown"]
        
        if method == "scrape":
            return self.scrape_url(url, formats, actions)
        elif method == "crawl":
            return self.crawl_url(url, crawl_limit, formats)
        elif method == "extract":
            if not extraction_prompt:
                extraction_prompt = "Extract the main content and key information from this page."
            return self.extract_data(url, extraction_prompt)
        else:
            return {
                "status": "error",
                "error": f"Unknown method: {method}. Use 'scrape', 'crawl', or 'extract'."
            }
    
    def format_results(self, result: Dict[str, Any], detailed: bool = False) -> str:
        """
        Format tool results as a readable string.
        
        :param result: Results from any of the tool methods
        :param detailed: Whether to include more details in the output
        :return: Formatted string with results
        """
        if result.get("status") == "error":
            return f"Error: {result.get('error', 'Unknown error')}"
        
        if "pages" in result:  # Crawl results
            output = f"Crawled {result.get('pages_count', 0)} pages from {result.get('url')}\n\n"
            
            if detailed and result.get("pages"):
                for i, page in enumerate(result.get("pages", [])[:5], 1):  # Show first 5 pages
                    output += f"{i}. {page.get('url', 'Unknown URL')}\n"
                    if page.get("title"):
                        output += f"   Title: {page.get('title')}\n"
                    if page.get("markdown") and len(page.get("markdown")) > 100:
                        output += f"   Preview: {page.get('markdown')[:100]}...\n"
                    output += "\n"
                
                if len(result.get("pages", [])) > 5:
                    output += f"... and {len(result.get('pages', [])) - 5} more pages\n"
            
            return output
            
        elif "data" in result:  # Extraction results
            output = f"Extracted data from {result.get('url')}\n\n"
            
            data = result.get("data", {})
            if isinstance(data, dict):
                for key, value in data.items():
                    output += f"{key}: {value}\n"
            else:
                output += str(data)
                
            return output
            
        else:  # Single page scrape results
            content = result.get("content", {})
            metadata = content.get("metadata", {})
            
            output = f"Scraped content from {result.get('url')}\n"
            if metadata and isinstance(metadata, dict) and metadata.get("title"):
                output += f"Title: {metadata.get('title')}\n"
            
            if content and isinstance(content, dict) and "markdown" in content and content["markdown"]:
                preview = content["markdown"][:500] + "..." if len(content["markdown"]) > 500 else content["markdown"]
                output += f"\nContent Preview:\n{preview}\n"
            
            return output
import re
import logging

# Methods to add to FirecrawlTool

# First, fix the logger import
def fix_logger_import():
    return '''
# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("firecrawl_tool")
'''.strip()

def can_handle(self, task: str) -> bool:
    """
    Determine if this tool can handle the given task.
    
    Args:
        task: The task description
        
    Returns:
        True if this tool can handle the task, False otherwise
    """
    # Check for web scraping related keywords
    scrape_keywords = [
        'scrape', 'crawl', 'extract from website', 'extract data', 
        'web data', 'website content', 'get from website',
        'scrape website', 'visit website', 'extract information',
        'webpage data', 'website information'
    ]
    
    task_lower = task.lower()
    
    # Check for keywords
    if any(keyword in task_lower for keyword in scrape_keywords):
        return True
        
    # Look for URL patterns
    url_patterns = [
        r'https?://[^\s]+',
        r'www\.[^\s]+\.[a-z]{2,}',
        r'scrape [^\s]+\.[a-z]{2,}',
        r'extract [^\s]+\.[a-z]{2,}'
    ]
    
    for pattern in url_patterns:
        if re.search(pattern, task):
            return True
    
    return False

def execute(self, task: str) -> str:
    """
    Execute web scraping or crawling based on the task description.
    
    Args:
        task: The task description
        
    Returns:
        Extracted content from the website
    """
    # Extract URL from task
    url = self._extract_url(task)
    
    if not url:
        return "I couldn't identify a URL to scrape. Please provide a valid website URL."
    
    # Add scheme if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Determine operation type (scrape or crawl)
    task_lower = task.lower()
    
    crawl_mode = any(term in task_lower for term in ['crawl', 'multiple pages', 'entire site', 'full site'])
    formats = ['markdown']
    
    # Check if HTML format is requested
    if any(term in task_lower for term in ['html', 'raw html', 'html content']):
        formats.append('html')
    
    try:
        if crawl_mode:
            # Determine crawl limit
            limit = 10  # Default
            limit_match = re.search(r'(\d+)\s+pages', task)
            if limit_match:
                try:
                    limit = int(limit_match.group(1))
                    limit = max(1, min(50, limit))  # Limit between 1-50
                except ValueError:
                    pass
                    
            # Perform crawl
            result = self.crawl_url(url, limit=limit, formats=formats)
            
            if result["status"] == "success":
                response = f"Crawled {result['pages_count']} pages starting from {url}:\n\n"
                
                for i, page in enumerate(result['pages'], 1):
                    page_url = page.get('url', 'Unknown URL')
                    content = page.get('markdown', page.get('text', 'No content extracted'))
                    
                    # Truncate content if too long
                    if len(content) > 500:
                        content = content[:500] + "...[content truncated]"
                        
                    response += f"Page {i}: {page_url}\n"
                    response += f"---\n{content}\n---\n\n"
                    
                    # Limit the number of pages shown in the response
                    if i >= 3:
                        response += f"...and {result['pages_count'] - 3} more pages (content omitted for brevity)"
                        break
                        
                return response
            else:
                return f"Error crawling {url}: {result.get('error', 'Unknown error')}"
        else:
            # Perform single page scrape
            result = self.scrape_url(url, formats=formats)
            
            if result["status"] == "success":
                content = result['content'].get('markdown', 'No content extracted')
                
                # Truncate content if too long
                if len(content) > 1500:
                    content = content[:1500] + "...[content truncated]"
                    
                return f"Scraped content from {url}:\n\n{content}"
            else:
                return f"Error scraping {url}: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error executing web scraping: {str(e)}"

def _extract_url(self, task: str) -> str:
    """
    Extract a URL from the task description.
    
    Args:
        task: The task description
        
    Returns:
        The extracted URL or empty string if not found
    """
    # Look for HTTP URLs
    url_pattern = r'(https?://[^\s]+)'
    match = re.search(url_pattern, task)
    if match:
        # Clean up URL (remove trailing punctuation)
        url = match.group(1)
        return re.sub(r'[.,;:\'")]$', '', url)
    
    # Look for domain patterns
    domain_pattern = r'(?:from|at|visit|scrape|extract)\s+(?:the\s+)?(?:site|website|webpage|url|domain)?\s*[\'"]?(www\.[^\s\'",]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s\'",]*)[\'"]?'
    match = re.search(domain_pattern, task, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # General domain pattern
    general_pattern = r'(www\.[^\s]+\.[a-z]{2,}|[a-zA-Z0-9-]+\.[a-z]{2,}[^\s]*)'
    match = re.search(general_pattern, task)
    if match:
        return match.group(1)
    
    return ""
