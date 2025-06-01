"""
Optimized configuration management for Enterprise AI Cost Optimizer
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from functools import cached_property
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    """Immutable application configuration with validation and caching"""

    # API Keys
    lyzr_api_key: str = field(default_factory=lambda: os.getenv("LYZR_API_KEY", ""))
    lyzr_agent_ids: Dict[str, str] = field(default_factory=lambda: {
        "roi": os.getenv("LYZR_AGENT_ID_ROI", ""),
        "cost": os.getenv("LYZR_AGENT_ID_COST", ""),
        "task": os.getenv("LYZR_AGENT_ID_TASK", ""),
    })
    lyzr_workspace_id: str = field(default_factory=lambda: os.getenv("LYZR_WORKSPACE_ID", ""))
    lyzr_user_id: str = field(default_factory=lambda: os.getenv("LYZR_USER_ID", "default_user@example.com"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # AWS (optional)
    aws_access_key_id: Optional[str] = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: Optional[str] = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))

    # App settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "True").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    default_llm: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM", "gemini"))
    use_mock_responses: bool = field(default_factory=lambda: os.getenv("USE_MOCK_RESPONSES", "False").lower() == "true")

    # UI
    app_title: str = "🚀 Enterprise AI Cost Optimizer"
    app_icon: str = "🤖"

    # Model params
    temperature: float = 0.7
    max_tokens: int = 1000
    requests_per_minute: int = 15

    @cached_property
    def validation_errors(self) -> list[str]:
        """Get validation errors (cached)"""

        def validation_errors(self) -> list[str]:
            if self.use_mock_responses:
                return []

            errors = []

            # Check each expected agent ID is set
            for key in ["roi", "cost", "task"]:
                if not self.lyzr_agent_ids.get(key):
                    errors.append(f"LYZR_AGENT_ID_{key.upper()} is required")

            # Also check API key
            if not self.lyzr_api_key:
                errors.append("LYZR_API_KEY is required")

            if self.default_llm == "gemini" and not self.gemini_api_key:
                errors.append("GEMINI_API_KEY is required")

            return errors

    @cached_property
    def is_valid(self) -> bool:
        """Check if configuration is valid (cached)"""
        return not self.validation_errors

    @cached_property
    def aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration dict (cached)"""
        return {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.aws_region
        } if self.aws_access_key_id else {}

    def validate(self) -> bool:
        """Validate configuration and print errors"""
        if not self.is_valid:
            for error in self.validation_errors:
                print(f"❌ Configuration Error: {error}")
        return self.is_valid

    def get_env_summary(self) -> Dict[str, str]:
        """Get environment summary for debugging"""
        return {
            "Environment": "Development" if self.debug_mode else "Production",
            "LLM": self.default_llm,
            "Mock Mode": str(self.use_mock_responses),
            "AWS": "Configured" if self.aws_config else "Not configured",
            "Log Level": self.log_level
        }

# Global config instance
config = Config()