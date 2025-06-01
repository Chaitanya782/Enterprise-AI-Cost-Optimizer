import requests
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from utils.logger import logger


class LyzrClient:
    """Modern Lyzr Studio API client supporting multiple agents."""

    BASE_URL = "https://agent-prod.studio.lyzr.ai/v3"

    def __init__(self, user_id: Optional[str] = None):
        if not config.lyzr_api_key:
            raise ValueError("LYZR_API_KEY not found in environment variables")

        if not config.lyzr_agent_ids:
            raise ValueError("LYZR_AGENT_IDS not configured properly in config")

        self.api_key = config.lyzr_api_key
        self.agent_ids = config.lyzr_agent_ids  # <-- dict: {"roi": "...", "cost": "...", "task": "..."}
        self.workspace_id = config.lyzr_workspace_id
        self.user_id = user_id or config.lyzr_user_id

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        })

        logger.info("Lyzr client initialized with multiple agents")

    def chat(self, message: str, agent_key: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Chat with a specific Lyzr agent by key: 'roi', 'cost', 'task'."""
        agent_id = self.agent_ids.get(agent_key)
        if not agent_id:
            raise ValueError(f"Agent ID for key '{agent_key}' not found")

        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id or f"default-session-{agent_key}",
            "message": message,
            **kwargs
        }

        try:
            response = self.session.post(
                f"{self.BASE_URL}/inference/chat/",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Lyzr chat success | Agent: {agent_key} | Session: {payload['session_id']}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Lyzr chat error ({agent_key}): {str(e)}")
            raise


_lyzr_client: Optional[LyzrClient] = None

def get_lyzr_client(user_id: Optional[str] = None) -> LyzrClient:
    global _lyzr_client
    if _lyzr_client is None:
        _lyzr_client = LyzrClient(user_id)
    return _lyzr_client


# def test_all_agents_prompts():
#     from core.lyzr_client import get_lyzr_client
#     client = get_lyzr_client()
#
#     test_queries = {
#         "cost": "Please analyze the cost breakdown of our cloud infrastructure for last quarter.",
#         "roi": "Calculate the expected ROI for a $10,000 investment over the next year.",
#         "task": "Prioritize these project tasks based on impact and effort: A, B, C, D."
#     }
#
#     all_success = True
#
#     for agent_key, query in test_queries.items():
#         print(f"\n🧪 Testing agent '{agent_key}' with query:\n{query}\n")
#         try:
#             response = client.chat(query, agent_key=agent_key)
#             content = response.get("response") or response.get("message") or str(response)
#             print(f"✅ Response from '{agent_key}':\n{content}\n{'-'*60}")
#         except Exception as e:
#             print(f"❌ Error testing '{agent_key}': {e}")
#             all_success = False
#
#     assert all_success, "One or more agents failed to respond."
#
#
#

