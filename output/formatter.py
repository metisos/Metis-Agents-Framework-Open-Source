class OutputFormatter:
    """
    Formats agent outputs into standardized structures similar to the OutputFormatterAgent in Metis.
    """
    def __init__(self):
        pass
    
    def format_output(self, output, intent="task"):
        """
        Format output based on the type of response and intent.
        
        Args:
            output: The raw output to format
            intent: The intent type ('question' or 'task')
            
        Returns:
            Formatted output as a dictionary
        """
        try:
            if intent == "question":
                return self._format_question_answer(output)
            else:
                return self._format_task_result(output)
        except Exception as e:
            # Fallback error handler - ensure we always return a valid response
            print(f"Error in OutputFormatter: {str(e)}")
            return {
                "type": "task_result" if intent == "task" else "question_answer",
                "data": {
                    "summary": f"Error formatting output: {str(e)}" if intent == "task" else "Error formatting answer",
                    "steps": [] if intent == "task" else None,
                    "answer": f"Error formatting answer: {str(e)}" if intent != "task" else None
                },
                "metadata": {
                    "format_version": "1.0",
                    "intent": intent,
                    "error": True
                }
            }
    
    def _format_question_answer(self, answer):
        """Format a simple question answer."""
        # Handle the case where answer could be various types
        if isinstance(answer, dict):
            # Try to extract answer from dictionary if it exists
            if "answer" in answer:
                answer_text = answer["answer"]
            elif "content" in answer:
                answer_text = answer["content"]
            elif "response" in answer:
                answer_text = answer["response"]
            else:
                # Convert dict to string if no specific answer field found
                answer_text = str(answer)
        else:
            # Convert non-dict answers to string
            answer_text = str(answer) if answer is not None else "No answer available"
        
        return {
            "type": "question_answer",
            "data": {
                "answer": answer_text
            },
            "metadata": {
                "format_version": "1.0",
                "intent": "question"
            }
        }
    
    def _format_task_result(self, result):
        """Format a task execution result."""
        # Initialize default values
        summary = "Task completed"
        steps = []
        
        # Process based on result type
        if isinstance(result, str):
            # Simple string result
            summary = result
            
        elif isinstance(result, dict):
            # Extract steps from results if available
            if "results" in result and isinstance(result["results"], dict):
                for task_id, task_info in result["results"].items():
                    # Extract task info with safe dict access
                    if isinstance(task_info, dict):
                        task_description = task_info.get("task", "Unknown task")
                        task_result = task_info.get("result", "No result")
                        steps.append({
                            "id": task_id,
                            "description": task_description,
                            "result": task_result
                        })
                    else:
                        # Handle non-dict task_info
                        steps.append({
                            "id": task_id,
                            "description": "Task information",
                            "result": str(task_info) if task_info is not None else "No result"
                        })
            
            # Extract summary if available
            if "summary" in result:
                summary = str(result["summary"])
            
            # Use steps from the result if directly available
            if not steps and "steps" in result and isinstance(result["steps"], list):
                steps = result["steps"]
        
        elif result is None:
            # Handle None result
            summary = "No task result provided"
            
        else:
            # For any other type, convert to string
            summary = str(result)
        
        # Create an entry for canvas display only - do not include content in the chat message
        # This keeps the task result only in the canvas view and not in the chat interface
        return {
            "type": "task_result",
            "data": {
                "summary": summary,
                "steps": steps
            },
            "metadata": {
                "format_version": "1.0",
                "intent": "task",
                "display_in_chat": False,  # Flag to indicate this should not be displayed in chat
                "canvas_only": True        # Flag to indicate this should only be displayed in canvas
            },
            "chat_message": "Task completed. Results are available in the canvas view."  # Simple message for chat
        }
