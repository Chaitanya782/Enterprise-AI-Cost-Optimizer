"""Base Agent class for all AI agents"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import json
import uuid
from functools import cached_property

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger
from core.lyzr_client import get_lyzr_client
from core.llm_manager import get_llm_manager

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(
        self,
        name: str,
        description: str,
        model: str = "gpt-3.5-turbo",
        use_lyzr: bool = True,
        agent_key: str = None  # <- Accept agent_key
    ):
        self.name = name
        self.description = description
        self.model = model
        self.use_lyzr = use_lyzr
        self.agent_key = agent_key or name.lower()
        self.conversation_history = []
        logger.info(f"Initializing agent: {name}")

    @cached_property
    def lyzr_client(self):
        return get_lyzr_client() if self.use_lyzr else None

    @cached_property
    def llm_manager(self):
        return get_llm_manager()

    @cached_property
    def agent_id(self):
        return self.lyzr_client.agent_ids.get(self.agent_key) if self.lyzr_client else None

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        pass

    def _build_prompt(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        return f"""You are acting as: {self.name}
        Role Description: {self.description}
        
        {self.get_system_prompt()}
        
        User Context:
        {json.dumps(context, indent=2) if context else 'No specific context provided'}
        
        User Query: {message}
        
        Instructions:
        1. Provide specific, actionable insights (not generic advice)
        2. Include concrete numbers, calculations, and data points
        3. Structure your response with clear sections
        4. Focus on the exact question asked
        5. If analyzing costs, provide actual cost breakdowns
        6. If analyzing ROI, provide specific projections with timeline
        7. If analyzing tasks, prioritize with clear reasoning
        
        Respond in a structured format that directly addresses the query."""

    def chat(self, message: str, context: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            session_id = f"{self.name.lower().replace(' ', '-')}-{session_id or str(uuid.uuid4())[:8]}"
            prompt = self._build_prompt(message, context)

            response = self.lyzr_client.chat(
                message=prompt,
                session_id=session_id,
                agent_key=self.agent_key  # <- Use passed key
            )


            content = response.get('response', response.get('message', ''))

            result = {
                "content": content,
                "agent": self.name,
                "session_id": session_id,
                "metadata": {"original_response": response, "context_used": context}
            }

            self.conversation_history.append({
                "user": message,
                "assistant": content,
                "context": context
            })

            logger.info(f"{self.name} response generated successfully")
            return result

        except Exception as e:
            logger.error(f"Agent chat error: {str(e)}")
            return {
                "content": f"Error generating response: {str(e)}",
                "agent": self.name,
                "error": True,
                "session_id": session_id
            }



