"""Task Analyzer Agent - Identifies AI automation opportunities"""
from typing import Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger


class TaskAnalyzerAgent(BaseAgent):
    def __init__(self, agent_key: str = "task"):
        super().__init__(name="Task Analyzer", description="Analyzes business tasks...", agent_key=agent_key)



    def get_system_prompt(self) -> str:
        return """Expert AI Task Analyzer specializing in identifying enterprise workflow automation opportunities.

                Analyze business processes focusing on:
                - Repetitive manual processes & data entry/processing
                - Document analysis/generation & customer interactions  
                - Decision-making processes & pattern recognition
                
                For each opportunity provide: Task description, AI solution, complexity (Low/Med/High), 
                time savings (hrs/week), priority (1-5), recommended AI models/tools.
                Be specific and actionable."""

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "analyze_task_complexity",
                "description": "Analyze AI implementation complexity for a specific task",
                "parameters": {
                    "task_description": "string",
                    "current_process": "string",
                    "data_availability": "string"
                }
            },
            {
                "name": "calculate_automation_roi",
                "description": "Calculate potential ROI for task automation",
                "parameters": {
                    "task_name": "string",
                    "hours_per_week": "number",
                    "hourly_cost": "number",
                    "implementation_cost": "number"
                }
            }
        ]

    def analyze_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """Analyze workflow for automation opportunities"""
        logger.info("Analyzing workflow for automation opportunities")

        prompt = f"""Analyze workflow for AI automation opportunities:

                {workflow_description}
                
                Provide: 1) Automatable tasks list 2) AI solutions for each 3) Priority ranking 
                4) Implementation complexity 5) Time savings estimates"""

        response = self.chat(prompt)
        return {
            "analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def prioritize_tasks(self, tasks: list[str]) -> Dict[str, Any]:
        """Prioritize tasks for AI implementation"""
        logger.info(f"Prioritizing {len(tasks)} tasks for AI implementation")

        tasks_str = "\n".join(f"- {task}" for task in tasks)

        prompt = f"""Prioritize tasks for AI automation by ROI potential, complexity, time-to-value, risk:

                {tasks_str}
                
                Provide ranked list with justification."""

        response = self.chat(prompt)
        return {
            "prioritization": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }