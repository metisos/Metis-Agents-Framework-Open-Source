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
