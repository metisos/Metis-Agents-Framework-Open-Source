"""
Content Generation Tool for Single Agent System

This tool handles content generation tasks like writing blog posts,
LinkedIn posts, emails, articles, and other written content.
"""

import os
import re
from typing import Dict, Any

class ContentGenerationTool:
    """Tool for generating written content across various formats."""
    
    def __init__(self):
        """Initialize the Content Generation tool."""
        from components.llm_interface import get_llm
        self.llm = get_llm()
        
    def get_description(self) -> str:
        """
        Return a description of what the tool does.
        
        :return: Tool description string
        """
        return "Generates high-quality written content such as social media posts, articles, emails, and other text formats"
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return parameter specifications for this tool.
        
        :return: Dictionary of parameter specifications
        """
        return {
            "content_type": {
                "type": "string",
                "description": "Type of content to generate (e.g., blog post, LinkedIn post)"
            },
            "topic": {
                "type": "string",
                "description": "Topic to write about"
            },
            "tone": {
                "type": "string",
                "description": "Tone of the content (professional, casual, etc.)",
                "default": "professional"
            },
            "length": {
                "type": "string",
                "description": "Desired length (short, medium, long)",
                "default": "medium"
            }
        }
    
    def can_handle(self, task: str) -> bool:
        """
        Determine if this tool can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            True if this tool can handle the task, False otherwise
        """
        # Check for content generation keywords
        content_keywords = [
            'write', 'create', 'draft', 'compose', 'generate', 'article', 
            'post', 'content', 'blog', 'LinkedIn', 'social media', 'message',
            'story', 'essay', 'summary', 'report', 'newsletter', 'email content',
            'press release', 'bio', 'description', 'author', 'copywriting',
            'research paper', 'academic paper', 'research article', 'thesis', 'dissertation',
            'academic writing', 'scientific paper', 'journal article', 'conference paper',
            'technical paper', 'white paper', 'paper about', 'paper on'
        ]
        
        task_lower = task.lower()
        
        # Check for keywords
        return any(keyword.lower() in task_lower for keyword in content_keywords)
    
    def execute(self, task: str, original_query: str = None) -> str:
        """
        Generate content based on the task description.
        
        Args:
            task: The task description
            original_query: The original user query that led to this task (optional)
            
        Returns:
            The generated content
        """
        # Extract parameters from the task
        content_type, topic, tone, length = self._extract_parameters(task, original_query)
        
        # Generate the content using our enhanced method
        content = self._generate_content(content_type, topic, tone, length)
        
        # Format the result
        result = {
            "content": content,
            "content_type": content_type,
            "topic": topic,
            "parameters": {
                "tone": tone,
                "length": length
            }
        }
        
        return result
    
    def _extract_parameters(self, task: str, original_query: str = None):
        """
        Extract content generation parameters from the task description.
        
        Args:
            task: The task description
            original_query: The original user query that led to this task (optional)
            
        Returns:
            Tuple of (content_type, topic, tone, length)
        """
        # Default values
        content_type = "general"
        topic = task
        tone = "professional"
        length = "medium"
        
        task_lower = task.lower()
        
        # Extract content type
        content_type_mapping = {
            'blog': 'blog post',
            'article': 'article',
            'post': 'social media post',
            'linkedin': 'LinkedIn post',
            'social media': 'social media post',
            'twitter': 'Twitter post',
            'facebook': 'Facebook post',
            'instagram': 'Instagram caption',
            'email': 'email',
            'newsletter': 'newsletter',
            'press release': 'press release',
            'ad': 'advertisement',
            'bio': 'biography',
            'description': 'description',
            'research paper': 'research paper',
            'academic paper': 'research paper',
            'thesis': 'thesis',
            'abstract': 'research paper abstract',
            'introduction': 'research paper introduction',
            'methodology': 'research methodology',
            'literature review': 'literature review',
            'outline': 'outline',
            'research question': 'research question formulation',
            'conclusion': 'conclusion',
            'discussion': 'discussion section'
        }
        
        # First check for research paper-specific content types
        for research_type in ['research paper', 'academic paper', 'thesis', 'literature review',
                            'outline', 'abstract', 'introduction', 'methodology', 'discussion',
                            'conclusion', 'research question']:
            if research_type in task_lower:
                content_type = content_type_mapping.get(research_type, 'research paper section')
                break
        
        # If not a research paper type, check other content types
        if content_type == "general":
            for key, value in content_type_mapping.items():
                if key in task_lower:
                    content_type = value
                    break
                    
        # Extract tone - use academic for research papers by default
        if 'research paper' in task_lower or 'academic' in task_lower or 'thesis' in task_lower:
            tone = 'academic'
        else:
            tone_mapping = {
                'professional': 'professional',
                'casual': 'casual',
                'formal': 'formal',
                'informal': 'informal',
                'friendly': 'friendly',
                'conversational': 'conversational',
                'humorous': 'humorous',
                'serious': 'serious',
                'authoritative': 'authoritative',
                'persuasive': 'persuasive',
                'informative': 'informative',
                'academic': 'academic',
                'scholarly': 'academic'
            }
            
            for key, value in tone_mapping.items():
                if key in task_lower:
                    tone = value
                    break
        
        # Extract length - research papers default to long
        if 'research paper' in task_lower or 'thesis' in task_lower:
            length = 'long'
        else:
            length_mapping = {
                'short': 'short',
                'brief': 'short',
                'concise': 'short',
                'medium': 'medium',
                'long': 'long',
                'detailed': 'long',
                'comprehensive': 'long',
                'extensive': 'long'
            }
            
            for key, value in length_mapping.items():
                if key in task_lower:
                    length = value
                    break
                    
        # Extract topic
        # If this is part of a larger query, extract the real topic from the original query
        if original_query:
            # First check for research paper topics
            research_topic_match = re.search(r'(?:write|create|generate|draft|compose)\s+a\s+(?:research paper|academic paper|article|thesis)\s+(?:about|on|regarding|concerning|for|related to)\s+(.+)', original_query, re.IGNORECASE)
            
            if research_topic_match:
                topic = research_topic_match.group(1).strip()
            else:
                # Look for other content creation verbs followed by topics
                topic_match = re.search(r'(?:write|create|generate|draft|compose)\s+a\s+(?:blog|article|post|content|essay|report|paper)\s+(?:about|on|regarding|concerning|for|related to)\s+(.+)', original_query, re.IGNORECASE)
                
                if topic_match:
                    topic = topic_match.group(1).strip()
                else:
                    # Fallback: use the substantive parts of the original query
                    keywords = [word for word in original_query.split() 
                              if word.lower() not in ['write', 'create', 'draft', 'a', 'an', 'the', 'and', 'or', 'but']]
                    if keywords:
                        topic = ' '.join(keywords[-3:])  # Use the last few words as the topic
                    else:
                        topic = original_query if original_query else task
        else:
            # Extract from task if no original query
            # Try to find "about X" or "on X" patterns
            topic_match = re.search(r'(?:about|on|regarding|concerning|for)\s+([^.,]+)', task_lower)
            if topic_match:
                topic = topic_match.group(1).strip()
            else:
                # Clean up task to extract topic
                cleaned_topic = task
                for action in ['write', 'create', 'generate', 'draft', 'compose', 'develop']:
                    cleaned_topic = re.sub(r'\b' + action + r'\b', '', cleaned_topic, flags=re.IGNORECASE)
                for content_word in ['outline', 'research paper', 'academic paper', 'article', 'post']:
                    cleaned_topic = re.sub(r'\b' + content_word + r'\b', '', cleaned_topic, flags=re.IGNORECASE)
                
                topic = cleaned_topic.strip()
        
        # Length extraction was already handled above
        return content_type, topic, tone, length
    
    def _generate_content(self, content_type: str, topic: str, tone: str, length: str) -> str:
        """Generate the content using the LLM."""
        # Build the prompt based on content type
        if 'research paper' in content_type.lower() or 'academic paper' in content_type.lower() or 'scientific paper' in content_type.lower():
            prompt = f"""You are an expert academic researcher and writer. Generate a comprehensive research paper about {topic}.
            The tone should be {tone} and the length should be {length}.
            
            Format the output as a complete research paper with the following sections:
            1. Title (bold and centered)
            2. Abstract (brief summary of the paper)
            3. Introduction (background and significance of the topic)
            4. Literature Review (relevant prior research)
            5. Methodology or Theoretical Framework (if applicable)
            6. Analysis/Discussion (main arguments and findings)
            7. Conclusion (implications and future directions)
            8. References (in APA or MLA format)
            
            Make sure to:  
            - Include appropriate academic language and terminology
            - Use proper citations and references
            - Maintain a logical flow between sections
            - Provide substantive content with specific details, examples, and evidence
            - Present balanced perspectives on any controversies
            - Use appropriate headings and subheadings
            
            Return only the formatted research paper without any explanations."""
        else:
            # Default prompt for other content types
            prompt = f"""You are a professional content writer. Generate {content_type} content about {topic}.
            The tone should be {tone} and the length should be {length}.
            
            Format the output appropriately for the content type, including any necessary sections, headings, or formatting.
            Be informative, engaging, and accurate.
            
            Return only the content without any explanations."""
        
        # Get the response from the LLM - remove extra parameters
        # LLMInterface.complete() only accepts the prompt parameter
        response = self.llm.complete(prompt)
        
        return response.strip()
    
    def _build_generation_prompt(self, content_type: str, topic: str, tone: str, length: str) -> str:
        """
        Build a prompt for content generation based on parameters.
        
        Args:
            content_type: Type of content to generate
            topic: Topic to write about
            tone: Tone of the content
            length: Desired length
            
        Returns:
            Prompt for content generation
        """
        # Define length in words
        length_guide = {
            'short': '100-200 words',
            'medium': '300-500 words',
            'long': '700-1000 words'
        }
        
        # Content type specific instructions
        type_instructions = {
            'linkedin post': "Create a professional LinkedIn post with a compelling hook, valuable insights, and a clear call to action. Use appropriate hashtags.",
            'blog post': "Write a blog post with a strong introduction, well-structured body with subheadings, and a conclusion that summarizes key points.",
            'tweet': "Create a concise, engaging tweet (max 280 characters) that conveys the key message with relevant hashtags.",
            'email': "Compose an email with an attention-grabbing subject line, personalized greeting, clear body content, and professional signature.",
            'article': "Write an informative article with a captivating headline, engaging introduction, detailed body with supporting evidence, and a thoughtful conclusion.",
            'social media post': "Create an engaging social media post with eye-catching opening, valuable content, and an invitation for engagement.",
            'press release': "Write a formal press release with a clear headline, dateline, introduction with key information, body with quotes and details, and standard boilerplate.",
            'newsletter': "Compose a newsletter with a personalized greeting, valuable main content divided into sections, and a friendly sign-off.",
            'summary': "Create a concise summary capturing the key points and main ideas in a well-structured format.",
            'general': "Create well-written content that communicates the main ideas clearly and effectively."
        }
        
        # Set default instruction if content type not found
        instruction = type_instructions.get(content_type, type_instructions['general'])
        
        # Build the prompt
        prompt = f"""
Generate {tone} {content_type} about "{topic}".

Content requirements:
- Length: {length_guide.get(length, '300-500 words')}
- Tone: {tone}
- Format: {content_type}

Specific instructions:
{instruction}

For the topic "{topic}", create content that is engaging, informative, and valuable to the reader.
"""
        
        return prompt
