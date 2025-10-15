"""
PDF Tool for Metis System

This tool provides capabilities for extracting and processing content from PDF files.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from io import BytesIO

# Import PyPDF2 with graceful fallback
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. PDF processing features will be limited.")

class PDFTool:
    """Tool for extracting and processing PDF content."""
    
    def __init__(self):
        """Initialize the PDF extraction tool."""
        pass
        
    def get_description(self) -> str:
        """
        Return a description of what the tool does.
        
        :return: Tool description string
        """
        return "Extracts and processes content from PDF files, including text extraction, metadata parsing, and table detection"
        
    def can_handle(self, task: str) -> bool:
        """
        Determine if this tool can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            True if this tool can handle the task, False otherwise
        """
        # Check for PDF related keywords
        pdf_keywords = [
            'pdf', 'document', 'extract pdf', 'read pdf', 
            'analyze pdf', 'parse pdf', 'extract document', 
            'document analysis', 'extract text from pdf',
            'pdf content', 'pdf information'
        ]
        
        task_lower = task.lower()
        
        # Check for keywords
        if any(keyword in task_lower for keyword in pdf_keywords):
            return True
            
        # Look for PDF file extension
        if re.search(r'\b\w+\.pdf\b', task_lower):
            return True
        
        return False
    
    def execute(self, task: str) -> str:
        """
        Execute PDF processing tasks based on the task description.
        
        Args:
            task: The task description
            
        Returns:
            Extracted information from the PDF
        """
        # Extract file path from task
        file_path = self._extract_file_path(task)
        if not file_path:
            return "Error: Could not find a valid PDF file path in the task. Please provide a path to a PDF file."
            
        # Check if the file exists
        if not os.path.exists(file_path):
            return f"Error: The PDF file '{file_path}' does not exist. Please provide a valid file path."
            
        # Determine what pages to extract
        pages = self._extract_page_numbers(task)
        
        # Determine if we should extract tables
        extract_tables = 'table' in task.lower()
        
        try:
            # Extract text
            result = self.extract_text(file_path, pages)
            if not result or not result.get('text'):
                return f"No text content was extracted from '{file_path}'. The file might be corrupted or empty."
            
            # Format output
            output = [f"## Extracted Content from PDF: {os.path.basename(file_path)}"]  
            
            # Add metadata if available
            if result.get('metadata'):
                output.append("\n### Document Metadata\n")
                for key, value in result['metadata'].items():
                    output.append(f"**{key}**: {value}")
            
            # Add content
            output.append("\n### Document Content\n")
            
            if pages:
                # If specific pages were extracted
                for page_num, page_text in result['text_by_page'].items():
                    output.append(f"\n#### Page {page_num + 1}\n")
                    
                    # Limit text length for readability
                    page_preview = page_text[:1000] + '...' if len(page_text) > 1000 else page_text
                    output.append(page_preview)
            else:
                # If all pages were extracted, just show a preview
                text_preview = result['text'][:2000] + '...' if len(result['text']) > 2000 else result['text']
                output.append(text_preview)
            
            # Add table information if requested and available
            if extract_tables and result.get('tables'):
                output.append("\n### Tables Detected\n")
                output.append(f"Found {len(result['tables'])} tables in the document.")
                
                for i, table in enumerate(result['tables']):
                    output.append(f"\n#### Table {i+1}\n")
                    output.append(table)
            
            return "\n\n".join(output)
            
        except Exception as e:
            return f"Error processing PDF '{file_path}': {str(e)}"
    
    def _extract_file_path(self, task: str) -> str:
        """
        Extract a PDF file path from the task description.
        
        Args:
            task: The task description
            
        Returns:
            The extracted file path or empty string if not found
        """
        # Look for path patterns
        path_patterns = [
            r'[\w\-\.\/ ]+\.pdf',  # Unix-style paths
            r'[\w\-\.\\ ]+\.pdf'   # Windows-style paths
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, task)
            if match:
                return match.group(0).strip()
                
        # Try to extract just the filename
        filename_pattern = r'\b\w+\.pdf\b'
        match = re.search(filename_pattern, task)
        if match:
            return match.group(0)
                
        return ""
        
    def _extract_page_numbers(self, task: str) -> List[int]:
        """
        Extract page numbers from the task description.
        
        Args:
            task: The task description
            
        Returns:
            List of page numbers (0-indexed) or empty list if not specified
        """
        task_lower = task.lower()
        
        # Look for page specifications
        # Examples: "page 5", "pages 1-3", "pages 1,2,3"
        
        # Single page
        single_page_pattern = r'page\s+(\d+)'
        match = re.search(single_page_pattern, task_lower)
        if match:
            try:
                page_num = int(match.group(1)) - 1  # Convert to 0-indexed
                return [page_num] if page_num >= 0 else []
            except ValueError:
                pass
                
        # Page range
        page_range_pattern = r'pages\s+(\d+)\s*-\s*(\d+)'
        match = re.search(page_range_pattern, task_lower)
        if match:
            try:
                start_page = int(match.group(1)) - 1  # Convert to 0-indexed
                end_page = int(match.group(2)) - 1
                if start_page >= 0 and end_page >= start_page:
                    return list(range(start_page, end_page + 1))
            except ValueError:
                pass
                
        # Comma-separated pages
        pages_list_pattern = r'pages\s+(\d+(?:\s*,\s*\d+)+)'
        match = re.search(pages_list_pattern, task_lower)
        if match:
            try:
                page_nums = [int(p.strip()) - 1 for p in match.group(1).split(',')]  # Convert to 0-indexed
                return [p for p in page_nums if p >= 0]
            except ValueError:
                pass
                
        return []
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return parameter specifications for this tool.
        
        :return: Dictionary of parameter specifications
        """
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the PDF file to process"
            },
            "pages": {
                "type": "array",
                "description": "Specific pages to extract (optional, defaults to all)"
            },
            "extract_tables": {
                "type": "boolean",
                "description": "Whether to attempt to identify and extract tables"
            }
        }
    
    def extract_text(self, file_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Extract text from a PDF file.
        
        :param file_path: Path to the PDF file
        :param pages: Optional list of specific pages to extract (0-indexed)
        :return: Dictionary with extracted text and metadata
        """
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}"
                }
            
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Get metadata
                metadata = reader.metadata
                formatted_metadata = {}
                if metadata:
                    for key, value in metadata.items():
                        if key.startswith('/'):
                            formatted_key = key[1:]  # Remove leading slash
                            formatted_metadata[formatted_key] = str(value)
                
                # Process all pages or specific pages
                page_range = pages if pages else range(total_pages)
                extracted_content = {}
                full_text = ""
                
                for i in page_range:
                    if 0 <= i < total_pages:
                        page = reader.pages[i]
                        text = page.extract_text()
                        extracted_content[f"page_{i+1}"] = text
                        full_text += text + "\n\n"
                
                return {
                    "status": "success",
                    "total_pages": total_pages,
                    "metadata": formatted_metadata,
                    "full_text": full_text,
                    "pages": extracted_content
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to extract text: {str(e)}"
            }
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        :param file_path: Path to the PDF file
        :return: Dictionary with metadata
        """
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}"
                }
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                
                # Format metadata
                formatted_metadata = {}
                if metadata:
                    for key, value in metadata.items():
                        if key.startswith('/'):
                            formatted_key = key[1:]  # Remove leading slash
                            formatted_metadata[formatted_key] = str(value)
                
                return {
                    "status": "success",
                    "metadata": formatted_metadata,
                    "total_pages": len(reader.pages)
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to extract metadata: {str(e)}"
            }
    
    def detect_tables(self, file_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Attempt to detect tables in PDF content using heuristics.
        This is a simple implementation; for more complex table detection, 
        specialized libraries like tabula-py would be needed.
        
        :param file_path: Path to the PDF file
        :param pages: Optional list of specific pages to examine
        :return: Dictionary with detected potential table regions
        """
        try:
            # Extract text first
            extraction_result = self.extract_text(file_path, pages)
            if extraction_result["status"] != "success":
                return extraction_result
            
            # Simple heuristic table detection
            table_patterns = [
                r"\n[ \t]*[-+|]+[ \t]*\n",  # ASCII-style tables
                r"\n[ \t]*(\w+[ \t]+){3,}\w+[ \t]*\n" * 3  # Aligned columns
            ]
            
            potential_tables = {}
            
            for page_key, text in extraction_result["pages"].items():
                page_tables = []
                
                for pattern in table_patterns:
                    matches = re.finditer(pattern, text, re.MULTILINE)
                    for match in matches:
                        start, end = match.span()
                        context_start = max(0, start - 100)
                        context_end = min(len(text), end + 100)
                        
                        page_tables.append({
                            "table_text": text[start:end],
                            "context": text[context_start:context_end],
                            "position": f"Characters {start}-{end}"
                        })
                
                if page_tables:
                    potential_tables[page_key] = page_tables
            
            return {
                "status": "success",
                "potential_tables": potential_tables,
                "note": "This is a simple heuristic detection. For more accurate table extraction, specialized libraries would be required."
            }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to detect tables: {str(e)}"
            }
    
    def summarize_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze and summarize the structure of a PDF document.
        
        :param file_path: Path to the PDF file
        :return: Dictionary with document structure summary
        """
        try:
            extraction_result = self.extract_text(file_path)
            if extraction_result["status"] != "success":
                return extraction_result
            
            # Analyze page lengths
            page_lengths = {page: len(text) for page, text in extraction_result["pages"].items()}
            
            # Detect potential headings using regex (simplified approach)
            headings = {}
            heading_pattern = r"(?:^|\n)([A-Z][A-Za-z0-9 :]{1,50})(?:\n)"
            
            for page_key, text in extraction_result["pages"].items():
                page_headings = []
                matches = re.finditer(heading_pattern, text)
                for match in matches:
                    heading = match.group(1).strip()
                    if len(heading) > 3:  # Filter out very short matches
                        page_headings.append(heading)
                
                if page_headings:
                    headings[page_key] = page_headings
            
            return {
                "status": "success",
                "total_pages": extraction_result["total_pages"],
                "metadata": extraction_result["metadata"],
                "page_lengths": page_lengths,
                "potential_headings": headings,
                "avg_chars_per_page": sum(page_lengths.values()) / len(page_lengths) if page_lengths else 0
            }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to summarize document structure: {str(e)}"
            }
import re
import os

# Methods to add to PDFTool

def can_handle(self, task: str) -> bool:
    """
    Determine if this tool can handle the given task.
    
    Args:
        task: The task description
        
    Returns:
        True if this tool can handle the task, False otherwise
    """
    # Check for PDF-related keywords
    pdf_keywords = [
        'pdf', 'document', 'extract', 'read pdf', 'process pdf', 
        'get text from pdf', 'extract text', 'analyze pdf',
        'pdf content', 'pdf file', 'document analysis'
    ]
    
    task_lower = task.lower()
    
    # Check for keywords
    if any(keyword in task_lower for keyword in pdf_keywords):
        return True
        
    # Look for file paths that might be PDFs
    pdf_pattern = r'([a-zA-Z0-9_\-/\\]+\.pdf)'
    return bool(re.search(pdf_pattern, task))

def execute(self, task: str) -> str:
    """
    Execute PDF processing tasks based on the task description.
    
    Args:
        task: The task description
        
    Returns:
        Extracted information from the PDF
    """
    # Extract file path from task
    file_path = self._extract_file_path(task)
    
    if not file_path:
        return "I couldn't determine which PDF file to process. Please specify a file path."
    
    # Check if file exists
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"The specified PDF file '{file_path}' does not exist or is not accessible."
    
    # Determine operation type
    task_lower = task.lower()
    
    # Check for metadata request
    if any(term in task_lower for term in ['metadata', 'info', 'information', 'details']):
        try:
            metadata = self.get_metadata(file_path)
            if metadata["status"] == "success":
                meta_str = "\n".join([f"{k}: {v}" for k, v in metadata["metadata"].items()])
                return f"PDF Metadata for {file_path}:\n\nPages: {metadata['total_pages']}\n\n{meta_str}"
            else:
                return f"Error getting metadata: {metadata.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error processing PDF metadata: {str(e)}"
    
    # Check for table detection
    if any(term in task_lower for term in ['table', 'tables', 'tabular']):
        try:
            tables = self.detect_tables(file_path)
            if tables["status"] == "success":
                if not tables.get("potential_tables"):
                    return f"No tables detected in {file_path}."
                
                result = f"Detected tables in {file_path}:\n\n"
                for page, page_tables in tables["potential_tables"].items():
                    result += f"Page {page}:\n"
                    for i, table in enumerate(page_tables, 1):
                        result += f"Table {i}: {table['table_text'][:200]}...\n\n"
                return result
            else:
                return f"Error detecting tables: {tables.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error detecting tables: {str(e)}"
    
    # Default to text extraction
    try:
        # Extract specific pages if mentioned
        pages = self._extract_page_numbers(task)
        
        extraction = self.extract_text(file_path, pages)
        if extraction["status"] == "success":
            if pages:
                # Return specific pages
                result = f"Extracted text from specified pages of {file_path}:\n\n"
                for page_key, text in extraction["pages"].items():
                    result += f"--- {page_key} ---\n{text[:500]}"
                    if len(text) > 500:
                        result += "...\n[Text truncated]\n\n"
                    else:
                        result += "\n\n"
                return result
            else:
                # Return full text (truncated if very long)
                full_text = extraction["full_text"]
                if len(full_text) > 1000:
                    return f"Extracted text from {file_path} (first 1000 characters):\n\n{full_text[:1000]}...\n[Text truncated]"
                else:
                    return f"Extracted text from {file_path}:\n\n{full_text}"
        else:
            return f"Error extracting text: {extraction.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def _extract_file_path(self, task: str) -> str:
    """
    Extract a PDF file path from the task description.
    
    Args:
        task: The task description
        
    Returns:
        The extracted file path or empty string if not found
    """
    # Look for direct file path mentions
    pdf_pattern = r'([a-zA-Z0-9_\-./\\]+\.pdf)'
    match = re.search(pdf_pattern, task)
    if match:
        return match.group(1)
    
    # Look for path in quotes
    quoted_pattern = r'[\'"]([^\'\"]+\.pdf)[\'"]'
    match = re.search(quoted_pattern, task)
    if match:
        return match.group(1)
    
    # Look for descriptions like "from the file X"
    descriptor_pattern = r'(?:from|in|of|the)\s+(?:file|pdf|document)\s+[\'"]?([a-zA-Z0-9_\-./\\]+\.pdf)[\'"]?'
    match = re.search(descriptor_pattern, task, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return ""

def _extract_page_numbers(self, task: str) -> List[int]:
    """
    Extract page numbers from the task description.
    
    Args:
        task: The task description
        
    Returns:
        List of page numbers (0-indexed) or empty list if not specified
    """
    pages = []
    
    # Look for "page X" or "pages X, Y, Z" patterns
    page_pattern = r'page\s+(\d+)'
    pages_pattern = r'pages\s+((?:\d+(?:\s*,\s*\d+)*)|(?:\d+\s*-\s*\d+))'
    
    # Single page
    for match in re.finditer(page_pattern, task, re.IGNORECASE):
        try:
            page_num = int(match.group(1)) - 1  # Convert to 0-indexed
            if page_num >= 0:
                pages.append(page_num)
        except ValueError:
            pass
    
    # Multiple pages
    match = re.search(pages_pattern, task, re.IGNORECASE)
    if match:
        pages_str = match.group(1)
        
        # Check for range (e.g., "1-5")
        if '-' in pages_str:
            try:
                start, end = map(int, pages_str.split('-'))
                pages.extend(range(start - 1, end))  # Convert to 0-indexed
            except ValueError:
                pass
        # Check for comma-separated (e.g., "1, 2, 3")
        else:
            for page_str in re.split(r'\s*,\s*', pages_str):
                try:
                    page_num = int(page_str) - 1  # Convert to 0-indexed
                    if page_num >= 0:
                        pages.append(page_num)
                except ValueError:
                    pass
    
    return pages
