class Scheduler:
    """
    Determines the execution order of tasks.
    Simplified version compared to Metis, with basic prioritization.
    """
    def __init__(self):
        from components.llm_interface import get_llm
        self.llm = get_llm()
    
    def prioritize_tasks(self, tasks: list) -> list:
        """
        Determine the optimal order for executing tasks.
        
        Args:
            tasks: List of tasks to prioritize
            
        Returns:
            Ordered list of tasks
        """
        if not tasks:
            return []
            
        # For simple cases with few tasks, keep the original order
        if len(tasks) <= 2:
            return tasks
            
        # For more complex cases, use the LLM to prioritize
        prompt = f"""
        Below is a list of tasks. Reorder them for optimal execution efficiency.
        Consider dependencies and logical order.
        
        Tasks:
        {chr(10).join([f"- {task}" for task in tasks])}
        
        Return the tasks in priority order, one per line.
        Only include the task text, without any numbering or bullets.
        """
        
        result = self.llm.complete(prompt)
        prioritized_tasks = [task.strip() for task in result.strip().split('\n') 
                           if task.strip() and any(t in task for t in tasks)]
        
        # Ensure all original tasks are included
        for task in tasks:
            if not any(task in pt for pt in prioritized_tasks):
                prioritized_tasks.append(task)
                
        return prioritized_tasks
