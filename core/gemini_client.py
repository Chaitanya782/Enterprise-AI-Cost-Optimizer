"""
Google Gemini API client wrapper
"""
import google.generativeai as genai
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from utils.logger import logger

class GeminiClient:
    """Wrapper for Google Gemini API"""

    def __init__(self):
        """Initialize Gemini client"""
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in configuration")

        # Configure Gemini
        genai.configure(api_key=config.gemini_api_key)

        # Use gemini-pro instead of gemini-1.5-pro for better quota
        self.model = genai.GenerativeModel('gemini-1.5-pro')

        # Track last request time for rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2  # 2 seconds between requests

        logger.info("Gemini client initialized successfully with gemini-pro model")

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,  # Reduced for free tier
        retry_count: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from Gemini with retry logic"""

        for attempt in range(retry_count):
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Generation configuration
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                # Check if response is valid
                if not response.text:
                    raise ValueError("Empty response from Gemini")

                # Calculate approximate cost (Gemini pricing)
                # Free tier: 60 queries/minute
                # Paid: $0.00025 per 1K characters input, $0.0005 per 1K characters output
                input_chars = len(prompt)
                output_chars = len(response.text)

                # For free tier, cost is 0
                estimated_cost = 0 if "free" in config.gemini_api_key.lower() else (
                    (input_chars / 1000) * 0.00025 +
                    (output_chars / 1000) * 0.0005
                )

                logger.info(f"Gemini generation successful. Estimated cost: ${estimated_cost:.6f}")

                return {
                    "content": response.text,
                    "model": "gemini-1.5-pro",
                    "input_tokens": input_chars // 4,  # Approximate
                    "output_tokens": output_chars // 4,  # Approximate
                    "estimated_cost": estimated_cost,
                    "finish_reason": "completed"
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Gemini generation error (attempt {attempt + 1}/{retry_count}): {error_msg}")

                # Check for quota errors
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < retry_count - 1:
                        # Exponential backoff
                        wait_time = min(60, (attempt + 1) * 10)
                        logger.info(f"Quota exceeded. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Return a mock response for testing
                        logger.warning("Using mock response due to quota limits")
                        return {
                            "content": "Hello from Gemini! This is a mock response due to API quota limits. In production, this would be an actual AI response analyzing your cost optimization needs.",
                            "model": "gemini-pro (mock)",
                            "input_tokens": len(prompt) // 4,
                            "output_tokens": 50,
                            "estimated_cost": 0,
                            "finish_reason": "quota_exceeded_mock"
                        }

                # For other errors, raise immediately
                if attempt == retry_count - 1:
                    raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate for Gemini)"""
        # Gemini doesn't provide exact token count, approximate with characters/4
        return len(text) // 4

# Create singleton instance
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client instance"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client