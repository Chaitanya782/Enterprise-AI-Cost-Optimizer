"""
Unified LLM Manager for handling multiple LLM providers
"""
from typing import Dict, Any, Optional, List  # Added List
from enum import Enum
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from utils.logger import logger
from core.gemini_client import get_gemini_client
from core.lyzr_client import get_lyzr_client

# Rest of the code remains the same...


class LLMProvider(Enum):
    """Available LLM providers"""
    LYZR = "lyzr"
    GEMINI = "gemini"
    BEDROCK = "bedrock"  # Future implementation


class LLMManager:
    """Manages LLM interactions across different providers"""

    def __init__(self):
        """Initialize LLM Manager"""
        self.default_provider = config.default_llm
        self.providers = {}

        # Initialize available providers
        try:
            if config.gemini_api_key:
                self.providers[LLMProvider.GEMINI] = get_gemini_client()
                logger.info("Gemini provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {str(e)}")

        try:
            if config.lyzr_api_key:
                self.providers[LLMProvider.LYZR] = get_lyzr_client()
                logger.info("Lyzr provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Lyzr: {str(e)}")

    def generate(
            self,
            prompt: str,
            provider: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Generate response using specified or default provider"""
        # Select provider
        provider_name = provider or self.default_provider

        try:
            if provider_name == LLMProvider.GEMINI.value:
                client = self.providers.get(LLMProvider.GEMINI)
                if not client:
                    raise ValueError("Gemini client not initialized")
                return client.generate(prompt, **kwargs)

            elif provider_name == LLMProvider.LYZR.value:
                # For Lyzr, we need to use an agent
                # This will be implemented when we create agents
                raise NotImplementedError("Lyzr agent chat will be implemented next")

            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        except Exception as e:
            logger.error(f"LLM generation error with {provider_name}: {str(e)}")

            # Try fallback provider
            if provider_name != LLMProvider.GEMINI.value and LLMProvider.GEMINI in self.providers:
                logger.info("Falling back to Gemini")
                return self.providers[LLMProvider.GEMINI].generate(prompt, **kwargs)

            raise

    def estimate_cost(
            self,
            prompt: str,
            provider: Optional[str] = None,
            max_tokens: int = 2000
    ) -> float:
        """Estimate cost for a prompt"""
        provider_name = provider or self.default_provider

        # Approximate token counts
        input_tokens = len(prompt) // 4
        output_tokens = max_tokens

        # Cost per 1K tokens (approximate)
        costs = {
            LLMProvider.GEMINI.value: {
                "input": 0.00025,  # $0.25 per 1M chars
                "output": 0.0005  # $0.50 per 1M chars
            },
            LLMProvider.LYZR.value: {
                "input": 0.0015,  # Depends on underlying model
                "output": 0.002
            }
        }

        provider_costs = costs.get(provider_name, costs[LLMProvider.GEMINI.value])

        total_cost = (
                (input_tokens / 1000) * provider_costs["input"] +
                (output_tokens / 1000) * provider_costs["output"]
        )

        return total_cost

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [p.value for p in self.providers.keys()]


# Create singleton instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get or create LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager