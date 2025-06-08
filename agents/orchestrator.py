"""
Enhanced Multi-Agent Orchestrator with Fixed Intent Classification for Specific Keywords
"""
from typing import Dict, Any, Optional, List, Tuple
import re
import time
import asyncio
from datetime import datetime
from pathlib import Path
from threading import Lock
import sys
from collections import defaultdict
import json

sys.path.append(str(Path(__file__).parent.parent))

from agents.task_analyzer import TaskAnalyzerAgent
from agents.cost_calculator import CostCalculatorAgent
from agents.roi_estimator import ROIEstimatorAgent
from memory.session_manager import SessionManager, QueryContext
from memory.context_analyzer import ContextAnalyzer
from utils.logger import logger

class RateLimiter:
    """Advanced rate limiter with burst handling and adaptive delays"""

    def __init__(self, calls_per_minute: int = 30, min_delay: float = 0.5, burst_limit: int = 5):
        self.calls_per_minute = calls_per_minute
        self.min_delay = min_delay
        self.burst_limit = burst_limit
        self.call_times = []
        self.burst_count = 0
        self.lock = Lock()
        self.last_call_time = 0

    def wait_if_needed(self):
        """Advanced rate limiting with burst handling"""
        with self.lock:
            current_time = time.time()

            # Reset burst count if enough time has passed
            if current_time - self.last_call_time > 60:
                self.burst_count = 0

            # Handle burst requests
            if self.burst_count >= self.burst_limit:
                adaptive_delay = self.min_delay * (1 + self.burst_count * 0.5)
                time.sleep(adaptive_delay)

            # Ensure minimum delay between calls
            time_since_last = current_time - self.last_call_time
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                time.sleep(sleep_time)
                current_time = time.time()

            # Clean old call times (older than 1 minute)
            cutoff_time = current_time - 60
            self.call_times = [t for t in self.call_times if t > cutoff_time]

            # Check if we're exceeding rate limit
            if len(self.call_times) >= self.calls_per_minute:
                oldest_call = min(self.call_times)
                wait_time = 60 - (current_time - oldest_call)
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    current_time = time.time()

            # Record this call
            self.call_times.append(current_time)
            self.last_call_time = current_time
            self.burst_count += 1

class AdvancedOrchestrator:
    """
    Top-Level Advanced Orchestrator with Enhanced Intelligence and FIXED Intent Classification
    """

    # FIXED: Enhanced intent classification with specific keyword triggers
    INTENT_CLASSIFICATION = {
        "cost_optimization": {
            # FIXED: Added specific trigger phrases that should immediately trigger cost analysis
            "trigger_phrases": [
                "cost analysis", "cost comparison", "compare costs", "llm costs", "api costs",
                "pricing analysis", "cost breakdown", "cost optimization", "reduce costs",
                "save money", "cheaper alternative", "budget analysis", "spending analysis"
            ],
            "primary_indicators": [
                "reduce cost", "save money", "cost optimization", "cheaper alternative",
                "budget reduction", "cost efficiency", "lower spending", "cost cutting"
            ],
            "secondary_indicators": [
                "compare prices", "pricing comparison", "budget analysis",
                "spending review", "cost breakdown", "financial optimization"
            ],
            "context_indicators": [
                "expensive", "budget", "cost", "price", "spend", "fee", "charge"
            ],
            "confidence_boost": 2.0,  # Increased boost for cost analysis
            "required_agents": ["cost"],
            "optional_agents": ["roi", "task"]
        },
        "roi_analysis": {
            # FIXED: Added specific ROI trigger phrases
            "trigger_phrases": [
                "roi analysis", "roi calculation", "return on investment", "calculate roi",
                "roi estimate", "investment return", "payback analysis", "break even analysis",
                "business case", "financial justification", "investment analysis"
            ],
            "primary_indicators": [
                "return on investment", "roi calculation", "payback period", "break even",
                "investment analysis", "financial return", "profit analysis", "value assessment"
            ],
            "secondary_indicators": [
                "worth investing", "business case", "financial justification", "cost benefit",
                "investment decision", "financial impact", "revenue impact"
            ],
            "context_indicators": [
                "roi", "return", "investment", "benefit", "value", "profit", "payback"
            ],
            "confidence_boost": 2.0,  # Increased boost for ROI analysis
            "required_agents": ["roi"],
            "optional_agents": ["cost", "task"]
        },
        "automation_planning": {
            # FIXED: Added specific task/automation trigger phrases
            "trigger_phrases": [
                "task analysis", "automation analysis", "workflow analysis", "process analysis",
                "automate tasks", "automation opportunities", "task automation", "workflow automation",
                "process optimization", "automation planning", "task optimization"
            ],
            "primary_indicators": [
                "automate process", "workflow automation", "task automation", "process optimization",
                "eliminate manual work", "streamline operations", "efficiency improvement"
            ],
            "secondary_indicators": [
                "reduce manual effort", "optimize workflow", "improve efficiency", "automation opportunities",
                "process improvement", "operational excellence", "productivity enhancement"
            ],
            "context_indicators": [
                "automate", "workflow", "process", "manual", "efficiency", "productivity", "task"
            ],
            "confidence_boost": 2.0,  # Increased boost for task analysis
            "required_agents": ["task"],
            "optional_agents": ["cost", "roi"]
        },
        "comprehensive_analysis": {
            "trigger_phrases": [
                "comprehensive analysis", "complete analysis", "full analysis", "detailed analysis",
                "thorough analysis", "end-to-end analysis", "holistic analysis", "overall analysis"
            ],
            "primary_indicators": [
                "complete analysis", "comprehensive review", "full assessment", "detailed evaluation",
                "thorough analysis", "end-to-end review", "holistic assessment"
            ],
            "secondary_indicators": [
                "analyze everything", "full picture", "complete overview", "detailed breakdown",
                "comprehensive study", "thorough evaluation", "complete assessment"
            ],
            "context_indicators": [
                "comprehensive", "complete", "full", "detailed", "thorough", "holistic"
            ],
            "confidence_boost": 1.5,
            "required_agents": ["cost", "roi", "task"],
            "optional_agents": []
        },
        "vendor_comparison": {
            "trigger_phrases": [
                "compare providers", "provider comparison", "vendor comparison", "llm comparison",
                "model comparison", "service comparison", "platform comparison"
            ],
            "primary_indicators": [
                "compare providers", "vendor comparison", "provider analysis", "service comparison",
                "platform comparison", "solution comparison", "alternative evaluation"
            ],
            "secondary_indicators": [
                "which is better", "best option", "recommend provider", "provider selection",
                "vendor evaluation", "service evaluation", "platform selection"
            ],
            "context_indicators": [
                "compare", "versus", "vs", "alternative", "option", "choice", "provider"
            ],
            "confidence_boost": 1.8,
            "required_agents": ["cost"],
            "optional_agents": ["roi"]
        }
    }

    # Advanced metric extraction patterns with validation
    METRIC_PATTERNS = {
        "financial": {
            "monthly_spend": [
                r'spending\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)\b',
                r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)\b',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|usd)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)\b'
            ],
            "annual_spend": [
                r'spending\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?\s*(?:per\s+)?(?:year|yearly|annually)\b',
                r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?\s*(?:per\s+)?(?:year|yearly|annually)\b'
            ],
            "budget": [
                r'budget\s*(?:of|is)?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)([kmb])?\b',
                r'allocated\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)([kmb])?\b'
            ],
            "target_reduction": [
                r'(?:reduce|save|cut|decrease)\s*(?:by|up\s*to)?\s*(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|savings|decrease|cut)'
            ]
        },
        "usage": {
            "daily_requests": [
                r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*(?:per\s*day|daily)',
                r'(\d+(?:,\d{3})*)\s*daily\s*(?:requests?|calls?|queries?|tickets?|messages?)'
            ],
            "monthly_requests": [
                r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*(?:per\s*month|monthly)',
                r'(\d+(?:,\d{3})*)\s*monthly\s*(?:requests?|calls?|queries?|tickets?|messages?)'
            ],
            "team_size": [
                r'team\s*(?:of|size)?\s*(\d+(?:,\d{3})*)\s*(?:people|members|employees|staff)?',
                r'(\d+(?:,\d{3})*)\s*(?:people|members|employees|staff|users)'
            ],
            "hours_per_week": [
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*hours?\s*(?:per\s*week|weekly)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*hrs?\s*(?:per\s*week|weekly)'
            ]
        },
        "technical": {
            "input_tokens": [
                r'(\d+(?:,\d{3})*)\s*input\s*tokens?',
                r'(\d+(?:,\d{3})*)\s*tokens?\s*(?:per\s*)?input'
            ],
            "output_tokens": [
                r'(\d+(?:,\d{3})*)\s*output\s*tokens?',
                r'(\d+(?:,\d{3})*)\s*tokens?\s*(?:per\s*)?(?:response|output)'
            ]
        }
    }

    # Provider mapping with enhanced detection
    PROVIDER_ECOSYSTEM = {
        "openai": {
            "aliases": ["openai", "gpt", "chatgpt", "gpt-4", "gpt-3.5", "dall-e"],
            "formal_name": "OpenAI API",
            "category": "LLM Provider",
            "strengths": ["performance", "reliability", "ecosystem"]
        },
        "anthropic": {
            "aliases": ["anthropic", "claude", "claude-3", "claude-2"],
            "formal_name": "Anthropic Claude",
            "category": "LLM Provider",
            "strengths": ["safety", "reasoning", "long-context"]
        },
        "google": {
            "aliases": ["google", "gemini", "bard", "vertex", "palm"],
            "formal_name": "Google Gemini",
            "category": "LLM Provider",
            "strengths": ["integration", "multimodal", "cost"]
        },
        "aws": {
            "aliases": ["aws", "bedrock", "amazon", "sagemaker"],
            "formal_name": "AWS Bedrock",
            "category": "Cloud Platform",
            "strengths": ["enterprise", "security", "integration"]
        },
        "azure": {
            "aliases": ["azure", "microsoft", "azure openai"],
            "formal_name": "Azure OpenAI",
            "category": "Cloud Platform",
            "strengths": ["enterprise", "compliance", "integration"]
        }
    }

    # Use case classification with complexity scoring
    USE_CASE_TAXONOMY = {
        "customer_support": {
            "keywords": ["support", "helpdesk", "ticket", "customer service", "chat support"],
            "complexity_score": 0.6,
            "automation_potential": 0.8,
            "typical_volume": "high",
            "default_metrics": {"daily_requests": 1000, "input_tokens": 150, "output_tokens": 200}
        },
        "content_generation": {
            "keywords": ["content", "writing", "blog", "marketing", "copywriting"],
            "complexity_score": 0.7,
            "automation_potential": 0.9,
            "typical_volume": "medium",
            "default_metrics": {"daily_requests": 200, "input_tokens": 100, "output_tokens": 500}
        },
        "data_analysis": {
            "keywords": ["analyze", "data", "report", "insights", "dashboard", "analytics"],
            "complexity_score": 0.8,
            "automation_potential": 0.7,
            "typical_volume": "medium",
            "default_metrics": {"daily_requests": 100, "input_tokens": 400, "output_tokens": 300}
        },
        "document_processing": {
            "keywords": ["document", "pdf", "extract", "summarize", "review", "contract"],
            "complexity_score": 0.7,
            "automation_potential": 0.9,
            "typical_volume": "medium",
            "default_metrics": {"daily_requests": 300, "input_tokens": 800, "output_tokens": 200}
        },
        "code_generation": {
            "keywords": ["code", "programming", "development", "api", "script"],
            "complexity_score": 0.9,
            "automation_potential": 0.6,
            "typical_volume": "low",
            "default_metrics": {"daily_requests": 50, "input_tokens": 300, "output_tokens": 600}
        }
    }

    def __init__(self, rate_limit_calls_per_minute: int = 30, min_delay_between_calls: float = 0.5):
        """Initialize the advanced orchestrator"""
        # Enhanced rate limiter
        self.rate_limiter = RateLimiter(rate_limit_calls_per_minute, min_delay_between_calls)

        # Initialize agents with enhanced capabilities
        self.task_agent = TaskAnalyzerAgent(agent_key="task")
        self.cost_agent = CostCalculatorAgent(agent_key="cost")
        self.roi_agent = ROIEstimatorAgent(agent_key="roi")

        # Advanced context management
        self.session_manager = SessionManager()
        self.context_analyzer = ContextAnalyzer()

        # Intelligence caches
        self._intent_cache = {}
        self._metric_cache = {}
        self._provider_cache = {}
        self._cache_lock = Lock()

        # Analysis quality tracking
        self.analysis_quality_scores = defaultdict(list)
        self.user_feedback_scores = defaultdict(float)

        logger.info("Advanced Orchestrator initialized with enhanced intelligence")

    def _advanced_intent_classification(self, user_query: str, session_context: Dict[str, Any] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        FIXED: Advanced intent classification with specific keyword triggers
        """
        query_lower = user_query.lower()

        # Check cache first
        cache_key = f"{query_lower}_{hash(str(session_context))}"
        with self._cache_lock:
            if cache_key in self._intent_cache:
                return self._intent_cache[cache_key]

        intent_scores = {}
        analysis_metadata = {
            "matched_patterns": {},
            "confidence_factors": {},
            "session_influence": 0.0,
            "trigger_matched": False
        }

        # FIXED: First check for exact trigger phrases (highest priority)
        for intent, config in self.INTENT_CLASSIFICATION.items():
            trigger_score = 0
            matched_triggers = []

            # Check for trigger phrases - these should immediately classify the intent
            for trigger in config.get("trigger_phrases", []):
                if trigger in query_lower:
                    trigger_score += 50  # Very high score for trigger phrases
                    matched_triggers.append(trigger)
                    analysis_metadata["trigger_matched"] = True
                    logger.info(f"TRIGGER MATCHED: '{trigger}' -> {intent}")

            if trigger_score > 0:
                intent_scores[intent] = trigger_score * config["confidence_boost"]
                analysis_metadata["matched_patterns"][intent] = [f"TRIGGER: {t}" for t in matched_triggers]
                continue  # Skip other scoring for this intent if trigger matched

        # If no triggers matched, use regular scoring
        if not analysis_metadata["trigger_matched"]:
            for intent, config in self.INTENT_CLASSIFICATION.items():
                score = 0
                matched_patterns = []

                # Primary indicators (highest weight)
                for indicator in config["primary_indicators"]:
                    if indicator in query_lower:
                        score += 10 * len(indicator.split())
                        matched_patterns.append(f"primary: {indicator}")

                # Secondary indicators (medium weight)
                for indicator in config["secondary_indicators"]:
                    if indicator in query_lower:
                        score += 5 * len(indicator.split())
                        matched_patterns.append(f"secondary: {indicator}")

                # Context indicators (lower weight but cumulative)
                context_matches = 0
                for indicator in config["context_indicators"]:
                    if indicator in query_lower:
                        context_matches += 1
                        matched_patterns.append(f"context: {indicator}")

                score += context_matches * 2

                # Apply confidence boost
                if score > 0:
                    score *= config["confidence_boost"]

                intent_scores[intent] = score
                analysis_metadata["matched_patterns"][intent] = matched_patterns

        # Session context influence
        if session_context:
            locked_use_case = session_context.get("locked_use_case")
            recent_intents = session_context.get("recent_intents", [])

            if recent_intents:
                # Boost score for consistent intent patterns
                most_recent = recent_intents[-1] if recent_intents else None
                if most_recent in intent_scores:
                    intent_scores[most_recent] *= 1.2
                    analysis_metadata["session_influence"] = 0.2

        # Calculate final intent and confidence
        if not any(intent_scores.values()):
            result = ("general_inquiry", 0.0, analysis_metadata)
        else:
            total_score = sum(intent_scores.values())
            max_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = max_intent[1] / total_score if total_score > 0 else 0.0

            # FIXED: Higher confidence for trigger matches
            if analysis_metadata["trigger_matched"]:
                confidence = min(0.95, confidence * 1.5)  # Boost confidence for trigger matches
                logger.info(f"TRIGGER CONFIDENCE BOOST: {max_intent[0]} -> {confidence:.2f}")

            # Adjust confidence based on clarity
            active_intents = [intent for intent, score in intent_scores.items() if score > 0]
            if len(active_intents) == 1:
                confidence = min(0.95, confidence * 1.2)  # High confidence for clear intent
            elif len(active_intents) > 3:
                confidence = max(0.3, confidence * 0.8)   # Lower confidence for ambiguous queries

            result = (max_intent[0], confidence, analysis_metadata)

        # Cache result
        with self._cache_lock:
            self._intent_cache[cache_key] = result

        logger.info(f"Intent Classification: {result[0]} (confidence: {result[1]:.2f})")
        return result

    def _intelligent_metric_extraction(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """
        Intelligent metric extraction with validation and smart defaults
        """
        query_lower = query.lower()
        extracted_metrics = {}
        validation_scores = {}

        # Extract metrics by category
        for category, patterns in self.METRIC_PATTERNS.items():
            for metric, regex_patterns in patterns.items():
                for pattern in regex_patterns:
                    matches = re.finditer(pattern, query_lower, re.IGNORECASE)
                    for match in matches:
                        try:
                            # Extract numeric value
                            value_str = match.group(1).replace(',', '')
                            value = float(value_str)

                            # Handle multipliers
                            if len(match.groups()) > 1 and match.group(2):
                                multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
                                multiplier = multipliers.get(match.group(2).lower(), 1)
                                value *= multiplier

                            # Validate metric reasonableness
                            validation_score = self._validate_metric(metric, value, query)
                            if validation_score > 0.5:  # Only accept reasonable values
                                extracted_metrics[metric] = value
                                validation_scores[metric] = validation_score
                                break  # Take first valid match
                        except (ValueError, IndexError):
                            continue

        # Apply intelligent defaults based on intent and context
        smart_defaults = self._generate_smart_defaults(intent, confidence, extracted_metrics, query)

        return {
            "extracted": extracted_metrics,
            "validation_scores": validation_scores,
            "smart_defaults": smart_defaults,
            "confidence": sum(validation_scores.values()) / len(validation_scores) if validation_scores else 0.0
        }

    def _validate_metric(self, metric: str, value: float, query: str) -> float:
        """
        Validate if extracted metric value is reasonable
        Returns confidence score 0.0-1.0
        """
        validation_rules = {
            "monthly_spend": {"min": 0, "max": 10000000, "typical_range": (100, 100000)},
            "annual_spend": {"min": 0, "max": 120000000, "typical_range": (1000, 1000000)},
            "budget": {"min": 0, "max": 50000000, "typical_range": (1000, 500000)},
            "daily_requests": {"min": 1, "max": 10000000, "typical_range": (10, 100000)},
            "monthly_requests": {"min": 1, "max": 300000000, "typical_range": (100, 3000000)},
            "team_size": {"min": 1, "max": 100000, "typical_range": (1, 1000)},
            "hours_per_week": {"min": 0.1, "max": 168, "typical_range": (1, 80)},
            "input_tokens": {"min": 1, "max": 100000, "typical_range": (10, 2000)},
            "output_tokens": {"min": 1, "max": 100000, "typical_range": (10, 2000)},
            "target_reduction": {"min": 0.01, "max": 0.95, "typical_range": (0.1, 0.7)}
        }

        if metric not in validation_rules:
            return 0.5  # Unknown metric, moderate confidence

        rules = validation_rules[metric]

        # Check absolute bounds
        if value < rules["min"] or value > rules["max"]:
            return 0.0  # Invalid value

        # Check if in typical range
        min_typical, max_typical = rules["typical_range"]
        if min_typical <= value <= max_typical:
            return 1.0  # High confidence
        elif value < min_typical:
            # Below typical range - confidence decreases with distance
            ratio = value / min_typical
            return max(0.3, ratio)
        else:
            # Above typical range - confidence decreases with distance
            ratio = max_typical / value
            return max(0.3, ratio)

    def _generate_smart_defaults(self, intent: str, confidence: float, extracted_metrics: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate intelligent defaults based on intent, confidence, and context
        """
        smart_defaults = {}

        # Determine use case from query
        use_case = self._classify_use_case(query)
        use_case_config = self.USE_CASE_TAXONOMY.get(use_case, {})

        # Apply use case defaults
        if "default_metrics" in use_case_config:
            for metric, default_value in use_case_config["default_metrics"].items():
                if metric not in extracted_metrics:
                    smart_defaults[f"suggested_{metric}"] = default_value

        # Intent-specific defaults
        if intent == "cost_optimization":
            if "monthly_spend" not in extracted_metrics:
                smart_defaults["suggested_monthly_spend"] = 5000  # Reasonable starting point
            if "target_reduction" not in extracted_metrics:
                smart_defaults["suggested_target_reduction"] = 0.3  # 30% reduction target

        elif intent == "roi_analysis":
            if "budget" not in extracted_metrics:
                smart_defaults["suggested_budget"] = 50000  # Implementation budget
            if "hours_per_week" not in extracted_metrics:
                smart_defaults["suggested_hours_per_week"] = 40  # Full-time equivalent

        elif intent == "automation_planning":
            if "team_size" not in extracted_metrics:
                smart_defaults["suggested_team_size"] = 10  # Medium team
            if "hours_per_week" not in extracted_metrics:
                smart_defaults["suggested_hours_per_week"] = 20  # Part-time automation target

        # Company size indicators
        company_indicators = {
            "startup": {"team_size": 5, "monthly_spend": 1000, "daily_requests": 100},
            "small": {"team_size": 25, "monthly_spend": 5000, "daily_requests": 1000},
            "medium": {"team_size": 100, "monthly_spend": 20000, "daily_requests": 5000},
            "large": {"team_size": 500, "monthly_spend": 100000, "daily_requests": 25000},
            "enterprise": {"team_size": 2000, "monthly_spend": 500000, "daily_requests": 100000}
        }

        query_lower = query.lower()
        for size, defaults in company_indicators.items():
            if size in query_lower or (size == "startup" and "start" in query_lower):
                for metric, value in defaults.items():
                    if metric not in extracted_metrics:
                        smart_defaults[f"suggested_{metric}"] = value
                break

        return smart_defaults

    def _classify_use_case(self, query: str) -> str:
        """
        Classify the use case based on query content
        """
        query_lower = query.lower()
        use_case_scores = {}

        for use_case, config in self.USE_CASE_TAXONOMY.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += len(keyword.split())
            use_case_scores[use_case] = score

        if use_case_scores and max(use_case_scores.values()) > 0:
            return max(use_case_scores.items(), key=lambda x: x[1])[0]

        return "general_ai_implementation"

    def _intelligent_provider_detection(self, query: str) -> List[Dict[str, Any]]:
        """
        Intelligent provider detection with context and capabilities
        """
        query_lower = query.lower()
        detected_providers = []

        for provider_key, config in self.PROVIDER_ECOSYSTEM.items():
            for alias in config["aliases"]:
                if alias in query_lower:
                    detected_providers.append({
                        "key": provider_key,
                        "formal_name": config["formal_name"],
                        "category": config["category"],
                        "strengths": config["strengths"],
                        "mentioned_as": alias
                    })
                    break

        # If no providers mentioned, suggest based on intent and context
        if not detected_providers:
            # Default provider suggestions based on common use cases
            default_suggestions = [
                {"key": "openai", "formal_name": "OpenAI API", "category": "LLM Provider", "strengths": ["performance"], "mentioned_as": "suggested"},
                {"key": "anthropic", "formal_name": "Anthropic Claude", "category": "LLM Provider", "strengths": ["safety"], "mentioned_as": "suggested"},
                {"key": "google", "formal_name": "Google Gemini", "category": "LLM Provider", "strengths": ["cost"], "mentioned_as": "suggested"}
            ]
            detected_providers = default_suggestions[:2]  # Suggest top 2

        return detected_providers

    def _determine_analysis_strategy(self, intent: str, confidence: float, metrics: Dict[str, Any], session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Determine the optimal analysis strategy based on intent, confidence, and available data
        """
        strategy = {
            "primary_analysis": [],
            "secondary_analysis": [],
            "analysis_depth": "standard",
            "agent_coordination": "sequential",
            "quality_threshold": 0.7
        }

        intent_config = self.INTENT_CLASSIFICATION.get(intent, {})

        # Determine required and optional agents
        required_agents = intent_config.get("required_agents", ["cost"])
        optional_agents = intent_config.get("optional_agents", [])

        strategy["primary_analysis"] = required_agents

        # Add optional agents based on confidence and available data
        if confidence > 0.7:
            strategy["secondary_analysis"] = optional_agents
        elif confidence > 0.5:
            # Add most relevant optional agent
            if optional_agents:
                strategy["secondary_analysis"] = [optional_agents[0]]

        # Determine analysis depth
        if confidence > 0.8 and len(metrics.get("extracted", {})) > 3:
            strategy["analysis_depth"] = "comprehensive"
        elif confidence < 0.5 or len(metrics.get("extracted", {})) < 2:
            strategy["analysis_depth"] = "basic"

        # Determine coordination strategy
        if len(strategy["primary_analysis"]) + len(strategy["secondary_analysis"]) > 2:
            strategy["agent_coordination"] = "parallel"

        # Adjust quality threshold based on confidence
        strategy["quality_threshold"] = max(0.5, confidence * 0.8)

        return strategy

    def analyze_request(self, user_query: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced request analysis with intelligent orchestration
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"advanced_orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create session if it doesn't exist
        try:
            self.session_manager.create_session(session_id, user_id)
        except Exception as e:
            logger.warning(f"Session creation failed: {str(e)}")

        logger.info(f"Advanced Orchestrator analyzing: {user_query[:100]}... (Session: {session_id})")

        # Get conversation history and context
        try:
            history = self.session_manager.get_session_history(session_id)
            context_analysis = self.context_analyzer.analyze_context_relevance(user_query, history)
            session_context = {
                "locked_use_case": self.session_manager.get_locked_use_case(session_id),
                "recent_intents": [ctx.intent for ctx in history[-3:]] if history else []
            }
        except Exception as e:
            logger.warning(f"Context analysis failed: {str(e)}")
            context_analysis = {}
            session_context = {}

        # Advanced intent classification
        intent, confidence, intent_metadata = self._advanced_intent_classification(user_query, session_context)

        # Intelligent metric extraction
        metrics_analysis = self._intelligent_metric_extraction(user_query, intent, confidence)

        # Provider detection
        detected_providers = self._intelligent_provider_detection(user_query)

        # Use case classification
        use_case = self._classify_use_case(user_query)

        # Lock use case for session consistency
        if not session_context.get("locked_use_case"):
            self.session_manager.lock_use_case(session_id, use_case)

        # Determine analysis strategy
        analysis_strategy = self._determine_analysis_strategy(intent, confidence, metrics_analysis, session_context)

        logger.info(f"Analysis Strategy: {analysis_strategy['analysis_depth']} depth, {len(analysis_strategy['primary_analysis'])} primary agents")

        # Initialize results structure
        results = {
            "query": user_query,
            "session_id": session_id,
            "intent": intent,
            "confidence": confidence,
            "use_case": use_case,
            "metrics": metrics_analysis["extracted"],
            "smart_defaults": metrics_analysis["smart_defaults"],
            "providers": [p["formal_name"] for p in detected_providers],
            "provider_details": detected_providers,
            "analysis_strategy": analysis_strategy,
            "intent_metadata": intent_metadata,
            "context": {
                "score": context_analysis.get('context_score', 0.0),
                "insights": [insight.__dict__ if hasattr(insight, '__dict__') else insight
                           for insight in context_analysis.get('insights', [])],
                "summary": context_analysis.get('context_summary', ''),
                "relevant_queries": len(context_analysis.get('relevant_context', []))
            },
            "analysis": {},
            "recommendations": [],
            "quality_score": 0.0
        }

        try:
            # Execute analysis strategy
            analysis_results = self._execute_analysis_strategy(
                user_query, analysis_strategy, metrics_analysis, detected_providers, use_case, context_analysis
            )

            results["analysis"] = analysis_results

            # Generate intelligent recommendations
            results["recommendations"] = self._generate_intelligent_recommendations(results, context_analysis)

            # Calculate overall quality score
            results["quality_score"] = self._calculate_analysis_quality(results)

            # Store query context for future reference
            try:
                query_context = QueryContext(
                    timestamp=datetime.now(),
                    query=user_query,
                    intent=intent,
                    confidence=confidence,
                    use_case=use_case,
                    extracted_metrics=metrics_analysis["extracted"],
                    providers=[p["formal_name"] for p in detected_providers],
                    analysis_results=results
                )
                self.session_manager.add_query_context(session_id, query_context)
            except Exception as e:
                logger.warning(f"Failed to store query context: {str(e)}")

        except Exception as e:
            logger.error(f"Advanced Orchestrator error: {str(e)}")
            results.update({
                "error": str(e),
                "recommendations": ["Please provide more specific details for better analysis."],
                "quality_score": 0.0
            })

        return results

    def _execute_analysis_strategy(self, query: str, strategy: Dict[str, Any], metrics: Dict[str, Any],
                                 providers: List[Dict[str, Any]], use_case: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the determined analysis strategy with intelligent agent coordination
        """
        analysis_results = {}

        # Prepare enhanced context for agents
        agent_context = {
            "metrics": metrics,
            "providers": providers,
            "use_case": use_case,
            "context": context,
            "analysis_depth": strategy["analysis_depth"]
        }

        # Execute primary analysis
        for agent_type in strategy["primary_analysis"]:
            try:
                self.rate_limiter.wait_if_needed()

                if agent_type == "cost":
                    analysis_results["costs"] = self._enhanced_cost_analysis(query, agent_context)
                elif agent_type == "roi":
                    analysis_results["roi"] = self._enhanced_roi_analysis(query, agent_context)
                elif agent_type == "task":
                    analysis_results["tasks"] = self._enhanced_task_analysis(query, agent_context)

            except Exception as e:
                logger.error(f"Primary analysis failed for {agent_type}: {str(e)}")
                analysis_results[f"{agent_type}_error"] = str(e)

        # Execute secondary analysis if primary was successful
        if len(analysis_results) > 0 and not any("_error" in key for key in analysis_results.keys()):
            for agent_type in strategy["secondary_analysis"]:
                try:
                    self.rate_limiter.wait_if_needed()

                    if agent_type == "cost" and "costs" not in analysis_results:
                        analysis_results["costs"] = self._enhanced_cost_analysis(query, agent_context)
                    elif agent_type == "roi" and "roi" not in analysis_results:
                        analysis_results["roi"] = self._enhanced_roi_analysis(query, agent_context)
                    elif agent_type == "task" and "tasks" not in analysis_results:
                        analysis_results["tasks"] = self._enhanced_task_analysis(query, agent_context)

                except Exception as e:
                    logger.warning(f"Secondary analysis failed for {agent_type}: {str(e)}")

        return analysis_results

    def _enhanced_cost_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced cost analysis with intelligent defaults and provider optimization"""
        metrics = context["metrics"]
        providers = context["providers"]
        use_case = context["use_case"]

        # Get smart defaults
        smart_defaults = metrics.get("smart_defaults", {})
        extracted = metrics.get("extracted", {})

        # Determine optimal parameters
        daily_requests = int(extracted.get('daily_requests') or smart_defaults.get('suggested_daily_requests', 1000))
        input_tokens = int(extracted.get('input_tokens') or smart_defaults.get('suggested_input_tokens', 200))
        output_tokens = int(extracted.get('output_tokens') or smart_defaults.get('suggested_output_tokens', 250))

        # Enhanced model selection based on use case and providers
        models = self._select_optimal_models(use_case, providers, daily_requests)

        return self.cost_agent.calculate_llm_costs(
            use_case=use_case,
            requests_per_day=daily_requests,
            avg_input_tokens=input_tokens,
            avg_output_tokens=output_tokens,
            models=models
        )

    def _enhanced_roi_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced ROI analysis with intelligent benefit estimation"""
        metrics = context["metrics"]
        use_case = context["use_case"]
        smart_defaults = metrics.get("smart_defaults", {})
        extracted = metrics.get("extracted", {})

        # Determine budget and benefits intelligently
        budget = extracted.get('budget') or smart_defaults.get('suggested_budget', 50000)
        hours_saved = extracted.get('hours_per_week', 40) * 50  # Annual hours
        team_size = extracted.get('team_size') or smart_defaults.get('suggested_team_size', 10)

        # Calculate intelligent benefits based on use case
        hourly_rate = 75  # Enhanced default rate
        annual_benefits = self._calculate_intelligent_benefits(use_case, hours_saved, team_size, hourly_rate)

        return self.roi_agent.calculate_roi(
            project_name=f"AI Implementation - {use_case}",
            implementation_cost=budget,
            annual_benefits=annual_benefits
        )

    def _enhanced_task_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced task analysis with context-aware insights"""
        return self.task_agent.analyze_workflow(query)

    def _select_optimal_models(self, use_case: str, providers: List[Dict[str, Any]], daily_requests: int) -> List[str]:
        """Select optimal models based on use case, providers, and volume"""
        base_models = ["gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]

        # Add provider-specific models
        provider_models = {
            "OpenAI API": ["gpt-4o-mini", "gpt-4-turbo"],
            "Anthropic Claude": ["claude-3-opus", "claude-3-haiku"],
            "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"]
        }

        for provider in providers:
            provider_name = provider.get("formal_name", "")
            if provider_name in provider_models:
                base_models.extend(provider_models[provider_name])

        # Optimize for volume
        if daily_requests > 10000:
            # High volume - prioritize cost-effective models
            return ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"] + base_models
        elif daily_requests < 100:
            # Low volume - prioritize quality
            return ["gpt-4o", "claude-3-opus", "gemini-1.5-pro"] + base_models

        return list(set(base_models))  # Remove duplicates

    def _calculate_intelligent_benefits(self, use_case: str, hours_saved: float, team_size: int, hourly_rate: float) -> Dict[str, float]:
        """Calculate intelligent benefits based on use case characteristics"""
        base_labor_savings = hours_saved * hourly_rate

        # Use case specific multipliers
        use_case_benefits = {
            "customer_support": {
                "Labor Cost Savings": base_labor_savings,
                "Customer Satisfaction Improvement": base_labor_savings * 0.3,
                "24/7 Availability Value": base_labor_savings * 0.4,
                "Error Reduction": base_labor_savings * 0.2
            },
            "content_generation": {
                "Labor Cost Savings": base_labor_savings,
                "Increased Output Volume": base_labor_savings * 0.5,
                "Quality Consistency": base_labor_savings * 0.2,
                "Speed to Market": base_labor_savings * 0.3
            },
            "data_analysis": {
                "Labor Cost Savings": base_labor_savings,
                "Faster Decision Making": base_labor_savings * 0.4,
                "Improved Accuracy": base_labor_savings * 0.3,
                "Scalable Insights": base_labor_savings * 0.2
            },
            "document_processing": {
                "Labor Cost Savings": base_labor_savings,
                "Processing Speed Improvement": base_labor_savings * 0.6,
                "Error Reduction": base_labor_savings * 0.4,
                "Compliance Improvement": base_labor_savings * 0.2
            }
        }

        return use_case_benefits.get(use_case, {
            "Labor Cost Savings": base_labor_savings,
            "Efficiency Gains": base_labor_savings * 0.3,
            "Quality Improvements": base_labor_savings * 0.2,
            "Scalability Benefits": base_labor_savings * 0.1
        })

    def _generate_intelligent_recommendations(self, results: Dict[str, Any], context_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent, context-aware recommendations"""
        recommendations = []

        intent = results.get("intent", "")
        confidence = results.get("confidence", 0.0)
        quality_score = results.get("quality_score", 0.0)
        analysis = results.get("analysis", {})

        # Quality-based recommendations
        if quality_score > 0.8:
            recommendations.append("🎯 **High-Quality Analysis**: Comprehensive insights generated with high confidence")
        elif quality_score < 0.5:
            recommendations.append("⚠️ **Limited Data**: Consider providing more specific details for enhanced analysis")

        # Intent-specific recommendations
        if intent == "cost_optimization":
            if cost_data := analysis.get("costs", {}).get("cost_breakdown"):
                cheapest = min(cost_data.items(), key=lambda x: x[1].get("monthly_cost", float('inf')))
                recommendations.append(f"💰 **Cost Leader**: {cheapest[0]} offers lowest cost at ${cheapest[1].get('monthly_cost', 0):,.0f}/month")

        if intent == "roi_analysis":
            if roi_data := analysis.get("roi", {}).get("key_metrics"):
                roi_pct = roi_data.get("roi_percentage", 0)
                payback = roi_data.get("payback_period_months", 0)
                if roi_pct > 200:
                    recommendations.append(f"🚀 **Excellent ROI**: {roi_pct:.0f}% return with {payback:.1f} month payback")
                elif roi_pct > 100:
                    recommendations.append(f"📈 **Strong ROI**: {roi_pct:.0f}% return with {payback:.1f} month payback")

        # Context-based recommendations
        if context_analysis.get("context_score", 0) > 0.6:
            recommendations.append(f"🧠 **Context Continuity**: Building on {results['context']['relevant_queries']} related queries")

        # Strategy recommendations
        strategy = results.get("analysis_strategy", {})
        if strategy.get("analysis_depth") == "comprehensive":
            recommendations.append("📊 **Comprehensive Analysis**: Multi-dimensional assessment completed")

        # Provider recommendations
        if provider_details := results.get("provider_details"):
            if len(provider_details) > 1:
                recommendations.append("🔄 **Multi-Provider Strategy**: Consider hybrid approach for optimal cost-performance")

        return recommendations[:5]  # Limit to top 5

    def _calculate_analysis_quality(self, results: Dict[str, Any]) -> float:
        """Calculate overall analysis quality score"""
        quality_factors = []

        # Intent confidence
        quality_factors.append(results.get("confidence", 0.0))

        # Data completeness
        metrics_count = len(results.get("metrics", {}))
        data_completeness = min(1.0, metrics_count / 5)  # Normalize to 5 metrics
        quality_factors.append(data_completeness)

        # Analysis breadth
        analysis_sections = len(results.get("analysis", {}))
        analysis_breadth = min(1.0, analysis_sections / 3)  # Normalize to 3 sections
        quality_factors.append(analysis_breadth)

        # Context relevance
        context_score = results.get("context", {}).get("score", 0.0)
        quality_factors.append(context_score)

        # Provider coverage
        provider_count = len(results.get("providers", []))
        provider_coverage = min(1.0, provider_count / 3)  # Normalize to 3 providers
        quality_factors.append(provider_coverage)

        # Calculate weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Intent confidence weighted highest
        quality_score = sum(factor * weight for factor, weight in zip(quality_factors, weights))

        return round(quality_score, 2)

    # Additional helper methods for session management
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary with context insights"""
        try:
            summary = self.session_manager.get_session_summary(session_id)

            # Add context insights
            history = self.session_manager.get_session_history(session_id)
            if history:
                latest_query = history[-1].query
                context_analysis = self.context_analyzer.analyze_context_relevance(latest_query, history[:-1])
                summary['context_insights'] = [
                    insight.__dict__ if hasattr(insight, '__dict__') else insight
                    for insight in context_analysis.get('insights', [])
                ]

            return summary
        except Exception as e:
            logger.error(f"Failed to get session summary: {str(e)}")
            return {"error": str(e)}

    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export comprehensive session data"""
        try:
            return self.session_manager.export_session_data(session_id)
        except Exception as e:
            logger.error(f"Failed to export session data: {str(e)}")
            return {"error": str(e)}

    def cleanup_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            return self.session_manager.cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Failed to cleanup sessions: {str(e)}")
            return 0

    def get_rate_limiter_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        with self.rate_limiter.lock:
            current_time = time.time()
            recent_calls = len([t for t in self.rate_limiter.call_times if current_time - t < 60])

            return {
                "calls_in_last_minute": recent_calls,
                "calls_per_minute_limit": self.rate_limiter.calls_per_minute,
                "min_delay_between_calls": self.rate_limiter.min_delay,
                "last_call_time": self.rate_limiter.last_call_time,
                "time_since_last_call": current_time - self.rate_limiter.last_call_time
            }

# Maintain backward compatibility
Orchestrator = AdvancedOrchestrator
