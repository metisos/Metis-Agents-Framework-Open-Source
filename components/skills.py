"""Simplified skills module for backward compatibility."""

class SkillRegistry:
    """Empty skill registry for backward compatibility."""
    
    def __init__(self):
        self.skills = {}
        
    def get_skill(self, skill_name):
        # Return None for any skill request
        return None
        
    def find_skills_for_task(self, task):
        # Return empty list for any task
        return []
        
    def get_all_skills(self):
        # Return empty dictionary of skills
        return self.skills
        
    def get_tools_for_skill(self, skill_name):
        # Return empty list for any skill
        return []

def get_skill_registry():
    """Return an empty skill registry for backward compatibility."""
    return SkillRegistry()

def initialize_standard_skills():
    """Dummy function for backward compatibility."""
    return {}
