"""Enhanced Multi-Agent Orchestrator with Rate Limiting and Chat Context Memory"""
from typing import Dict, Any, Optional, List, Tuple
import re
import time
import asyncio
from datetime import datetime
from pathlib import Path
from threading import Lock
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agents.task_analyzer import TaskAnalyzerAgent
from agents.cost_calculator import CostCalculatorAgent
from agents.roi_estimator import ROIEstimatorAgent
from memory.session_manager import SessionManager, QueryContext
from memory.context_analyzer import ContextAnalyzer
from utils.logger import logger

class RateLimiter:
    """Simple rate limiter to prevent API calls from being made too quickly"""

    def __init__(self, calls_per_minute: int = 30, min_delay: float = 0.5):
        self.calls_per_minute = calls_per_minute
        self.min_delay = min_delay
        self.call_times = []
        self.lock = Lock()
        self.last_call_time = 0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()

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

class Orchestrator:
    """Enhanced Orchestrator with rate limiting and conversation context memory"""

    INTENT_KEYWORDS_ENHANCED = {
        "cost_analysis": {
            # High priority cost terms (weight 3x)
            "high_priority": ["cost", "price", "pricing", "budget", "expensive", "cheap", "spend", "fee"],
            # Medium priority cost terms (weight 2x)
            "medium_priority": ["compare llm", "llm cost", "api cost", "monthly cost", "save money", "reduce cost",
                                "cost of implementing", "budget for", "expense", "financial"],
            # Context cost terms (weight 1x)
            "context": ["gpt-4 cost", "claude cost", "api pricing", "token cost", "usage cost"]
        },
        "task_analysis": {
            "high_priority": ["automate", "workflow", "process", "streamline", "optimize"],
            "medium_priority": ["task", "implement", "ai for", "replace", "manual", "repetitive", "efficiency",
                                "productivity"],
            "context": ["automation", "digitize", "transform"]
        },
        "roi_analysis": {
            "high_priority": ["roi", "return", "investment", "payback", "benefit", "value"],
            "medium_priority": ["worth", "profit", "savings", "break even", "business case", "justification"],
            "context": ["cost benefit", "financial return"]
        }
    }

    PROVIDER_MAP = {
        "aws": "AWS Bedrock", "bedrock": "AWS Bedrock", "amazon": "AWS Bedrock",
        "openai": "OpenAI API", "gpt": "OpenAI API", "chatgpt": "OpenAI API",
        "azure": "Azure OpenAI", "microsoft": "Azure OpenAI",
        "anthropic": "Anthropic Claude", "claude": "Anthropic Claude",
        "gemini": "Google Gemini", "google": "Google Gemini", "vertex": "Vertex AI",
        "cohere": "Cohere", "huggingface": "Hugging Face", "meta": "Meta Llama"
    }

    USE_CASE_KEYWORDS = {
        "customer_support": ["support", "helpdesk", "ticket", "customer service"],
        "content_generation": ["content", "writing", "blog", "marketing", "copywriting"],
        "data_analysis": ["analyze", "data", "report", "insights", "dashboard"],
        "code_generation": ["code", "programming", "development", "api", "script"],
        "document_processing": ["document", "pdf", "extract", "summarize", "review"],
        "automation": ["automate", "workflow", "process", "streamline", "efficiency"]
    }

    # USE_CASE_DEFAULTS = {
    #     "Customer Support": {"input_tokens": 100, "output_tokens": 150, "daily_requests": 500},
    #     "Content Generation": {"input_tokens": 50, "output_tokens": 400, "daily_requests": 200},
    #     "Data Analysis": {"input_tokens": 300, "output_tokens": 200, "daily_requests": 100},
    #     "Code Generation": {"input_tokens": 200, "output_tokens": 500, "daily_requests": 150}
    # }

    FINANCIAL_PATTERNS = [
        r'spending\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|usd)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)?',
        r'budget\s*(?:of|is)?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)?'
    ]

    PERCENT_PATTERNS = [
        r'(?:reduce|save|cut|decrease)\s*(?:by|up\s*to)?\s*(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|savings|decrease)',
        r'target\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*%'
    ]

    USAGE_PATTERNS = {
        'daily_requests': [
            r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*(?:per\s*day|daily)',
            r'(\d+(?:,\d{3})*)\s*daily\s*(?:requests?|calls?|queries?|tickets?|messages?)',
            r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*a\s*day',
            r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*each\s*day'
        ],
        'monthly_requests': [
            r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*(?:per\s*month|monthly)',
            r'(\d+(?:,\d{3})*)\s*monthly\s*(?:requests?|calls?|queries?|tickets?|messages?)',
            r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?|tickets?|messages?)\s*a\s*month'
        ],
        'users': [
            r'(\d+(?:,\d{3})*)\s*(?:users?|customers?|employees?|agents?|staff)',
            r'team\s*of\s*(\d+(?:,\d{3})*)',
            r'(\d+(?:,\d{3})*)\s*people'
        ],
        'hours_saved': [
            r'save\s*(\d+(?:,\d{3})*)\s*hours?',
            r'(\d+(?:,\d{3})*)\s*hours?\s*(?:saved|reduction|per\s*week)',
            r'reduce\s*(?:by\s*)?(\d+(?:,\d{3})*)\s*hours?'
        ]
    }
    TOKEN_PATTERNS = {
        'input_tokens': [
            r'(\d+(?:,\d{3})*)\s*input\s*tokens?',
            r'(\d+(?:,\d{3})*)\s*tokens?\s*(?:per\s*)?input',
            r'prompt\s*(?:of|is)?\s*(\d+(?:,\d{3})*)\s*tokens?',
            r'send\s*(\d+(?:,\d{3})*)\s*tokens?'
        ],
        'output_tokens': [
            r'(\d+(?:,\d{3})*)\s*output\s*tokens?',
            r'(\d+(?:,\d{3})*)\s*tokens?\s*(?:per\s*)?(?:response|output)',
            r'generate\s*(\d+(?:,\d{3})*)\s*tokens?',
            r'response\s*(?:of|is)?\s*(\d+(?:,\d{3})*)\s*tokens?'
        ],
        'avg_tokens': [
            r'(\d+(?:,\d{3})*)\s*tokens?\s*(?:average|avg|typical)',
            r'(?:average|avg|typical)\s*(\d+(?:,\d{3})*)\s*tokens?'
        ]
    }

    COMPLEXITY_INDICATORS = {
        'high_complexity': [
            'detailed', 'comprehensive', 'complex', 'thorough', 'in-depth',
            'technical', 'legal', 'medical', 'scientific', 'analytical',
            'long responses', 'detailed answers', 'extensive output'
        ],
        'medium_complexity': [
            'moderate', 'standard', 'typical', 'normal', 'regular',
            'balanced', 'structured'
        ],
        'low_complexity': [
            'simple', 'basic', 'brief', 'short', 'quick', 'minimal',
            'concise', 'straightforward', 'summary', 'bullet points'
        ]
    }

    VOLUME_INDICATORS = {
        'high_volume': [
            'enterprise', 'large scale', 'high volume', 'thousands', 'millions',
            'busy', 'heavy usage', 'peak hours', '24/7', 'continuous'
        ],
        'medium_volume': [
            'moderate', 'regular', 'standard', 'typical', 'normal business hours'
        ],
        'low_volume': [
            'small', 'startup', 'limited', 'few', 'occasional', 'light usage',
            'testing', 'pilot', 'prototype'
        ]
    }

    INDUSTRY_INDICATORS = {
        'legal': ['legal', 'law', 'contract', 'compliance', 'regulation', 'attorney'],
        'medical': ['medical', 'healthcare', 'patient', 'diagnosis', 'clinical'],
        'technical': ['technical', 'engineering', 'code', 'API', 'software', 'development'],
        'financial': ['financial', 'banking', 'trading', 'investment', 'accounting'],
        'education': ['education', 'learning', 'student', 'academic', 'training'],
        'ecommerce': ['ecommerce', 'shopping', 'product', 'order', 'customer service']
    }

    def __init__(self, rate_limit_calls_per_minute: int = 30, min_delay_between_calls: float = 0.5):
        """Initialize enhanced orchestrator with rate limiting and context memory"""
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate_limit_calls_per_minute, min_delay_between_calls)

        # Initialize existing agents
        self.task_agent = TaskAnalyzerAgent(agent_key="task")
        self.cost_agent = CostCalculatorAgent(agent_key="cost")
        self.roi_agent = ROIEstimatorAgent(agent_key="roi")

        # Initialize new context components
        self.session_manager = SessionManager()
        self.context_analyzer = ContextAnalyzer()

        # Cache for intent classification to avoid repeated processing
        self._intent_cache = {}
        self._cache_lock = Lock()

        # Initialize agents
        for agent in [self.task_agent, self.cost_agent, self.roi_agent]:
            if hasattr(agent, 'create_or_get_agent'):
                agent.create_or_get_agent()

        logger.info("Enhanced Orchestrator initialized with rate limiting")

    def _get_dynamic_use_case_defaults(self, query: str, use_case: str) -> Dict[str, Any]:
        """Get dynamic defaults based on query content and use case"""
        query_lower = query.lower()

        # Enhanced base defaults
        base_defaults = {
            "Customer Support": {"input_tokens": 100, "output_tokens": 150, "daily_requests": 500},
            "Content Generation": {"input_tokens": 50, "output_tokens": 400, "daily_requests": 200},
            "Data Analysis": {"input_tokens": 300, "output_tokens": 200, "daily_requests": 100},
            "Code Generation": {"input_tokens": 200, "output_tokens": 500, "daily_requests": 150},
            "Document Processing": {"input_tokens": 500, "output_tokens": 250, "daily_requests": 100},
            "Automation": {"input_tokens": 150, "output_tokens": 200, "daily_requests": 300},
            "General AI Implementation": {"input_tokens": 150, "output_tokens": 250, "daily_requests": 300}
        }

        defaults = base_defaults.get(use_case, base_defaults["General AI Implementation"]).copy()

        # Adjust for complexity
        complexity_multiplier = 1.0
        for level, indicators in self.COMPLEXITY_INDICATORS.items():
            if any(indicator in query_lower for indicator in indicators):
                if level == 'high_complexity':
                    complexity_multiplier = 2.0
                elif level == 'low_complexity':
                    complexity_multiplier = 0.6
                break

        defaults["input_tokens"] = int(defaults["input_tokens"] * complexity_multiplier)
        defaults["output_tokens"] = int(defaults["output_tokens"] * complexity_multiplier)

        # Adjust for volume
        volume_multiplier = 1.0
        for level, indicators in self.VOLUME_INDICATORS.items():
            if any(indicator in query_lower for indicator in indicators):
                if level == 'high_volume':
                    volume_multiplier = 3.0
                elif level == 'low_volume':
                    volume_multiplier = 0.5
                break

        defaults["daily_requests"] = int(defaults["daily_requests"] * volume_multiplier)

        # Industry-specific adjustments
        for industry, indicators in self.INDUSTRY_INDICATORS.items():
            if any(indicator in query_lower for indicator in indicators):
                if industry in ['legal', 'medical', 'technical']:
                    defaults["input_tokens"] = int(defaults["input_tokens"] * 1.5)
                    defaults["output_tokens"] = int(defaults["output_tokens"] * 1.3)
                elif industry == 'financial':
                    defaults["input_tokens"] = int(defaults["input_tokens"] * 1.3)
                break

        return defaults

    def _extract_token_metrics(self, query: str, contextual_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract token-specific metrics from query"""
        metrics = {}
        query_lower = query.lower()

        # Extract explicit token mentions
        for metric, patterns in self.TOKEN_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    metrics[metric] = float(matches[0].replace(',', ''))
                    break

        # Handle average tokens -> split into input/output
        if 'avg_tokens' in metrics:
            if 'input_tokens' not in metrics:
                metrics['input_tokens'] = metrics['avg_tokens'] * 0.4  # 40% input
            if 'output_tokens' not in metrics:
                metrics['output_tokens'] = metrics['avg_tokens'] * 0.6  # 60% output
            del metrics['avg_tokens']

        # Apply contextual defaults
        if contextual_defaults:
            for key in ['suggested_input_tokens', 'suggested_output_tokens']:
                if key in contextual_defaults and key.replace('suggested_', '') not in metrics:
                    base_key = key.replace('suggested_', '')
                    metrics[base_key] = contextual_defaults[key]

        return metrics

    # def reset_session_context(self, session_id: str):
    #     """Reset session context to allow new use case"""
    #     if session_id in self.session_manager.locked_use_cases:
    #         del self.session_manager.locked_use_cases[session_id]
    #     self.session_manager.clear_session(session_id)  # If this method exists
    #     logger.info(f"Reset context for session {session_id}")

    def _classify_intent(self, user_query: str, session_id: str = None) -> Tuple[str, float]:
        """Enhanced intent classification with weighted scoring and cost prioritization"""

        # Check session context first
        if session_id and self.session_manager.get_locked_use_case(session_id):
            query_lower = user_query.lower()
            if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
                return ("cost_analysis", 0.8)

        # Check cache
        with self._cache_lock:
            if user_query in self._intent_cache:
                return self._intent_cache[user_query]

        query_lower = user_query.lower()
        weighted_scores = {}

        # Enhanced weighted scoring using your new INTENT_KEYWORDS_ENHANCED
        for intent, keyword_groups in self.INTENT_KEYWORDS_ENHANCED.items():
            score = 0

            # High priority keywords (3x weight)
            for keyword in keyword_groups.get("high_priority", []):
                if keyword in query_lower:
                    score += 3 * len(keyword.split())

            # Medium priority keywords (2x weight)
            for keyword in keyword_groups.get("medium_priority", []):
                if keyword in query_lower:
                    score += 2 * len(keyword.split())

            # Context keywords (1x weight)
            for keyword in keyword_groups.get("context", []):
                if keyword in query_lower:
                    score += 1 * len(keyword.split())

            weighted_scores[intent] = score

        # Special case: If explicit cost terms + model names, boost cost_analysis
        cost_terms = ["cost", "price", "pricing", "budget", "expensive", "cheap"]
        model_names = ["gpt-4", "gpt-3.5", "claude", "gemini", "llama"]

        has_cost_term = any(term in query_lower for term in cost_terms)
        has_model_name = any(model in query_lower for model in model_names)

        if has_cost_term and has_model_name:
            weighted_scores["cost_analysis"] += 5  # Strong boost for cost + model queries

        # Calculate result
        total_score = sum(weighted_scores.values())
        if total_score == 0:
            result = ("general", 0.0)
        else:
            max_intent = max(weighted_scores.items(), key=lambda x: x[1])
            confidence = max_intent[1] / total_score

            # Enhanced confidence threshold - favor focused analysis over comprehensive
            active_intents = [intent for intent, score in weighted_scores.items() if score > 0]
            # In _classify_intent method, replace the comprehensive logic with:
            if len(active_intents) > 1:
                # Check for explicit comprehensive indicators
                comprehensive_indicators = ["analysis", "compare", "evaluation", "assessment", "overview", "complete"]
                has_comprehensive_intent = any(indicator in query_lower for indicator in comprehensive_indicators)

                if has_comprehensive_intent or confidence < 0.5:  # Smart comprehensive detection
                    result = ("comprehensive", confidence)
                else:
                    result = (max_intent[0], confidence)  # Stick with primary intent

        # Cache and return
        with self._cache_lock:
            self._intent_cache[user_query] = result

        return result

    def _extract_financial_metrics(self,query: str, contextual_defaults: dict = None) -> dict:
        """Fixed version of financial data extraction"""
        metrics = {}
        query_lower = query.lower()
        multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}



        print(f"Processing query: '{query_lower}'")

        # Extract from current query
        for i, pattern in enumerate(self.FINANCIAL_PATTERNS):
            print(f"Testing pattern {i + 1}: {pattern}")

            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                print(f"  Match found: '{match.group(0)}'")
                print(f"  Groups: {match.groups()}")

                # Parse the amount
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                print(f"  Base amount: {amount}")

                # Check for multiplier - THIS IS THE CRITICAL FIX
                multiplier_char = match.group(2)
                print(f"  Multiplier group: '{multiplier_char}' (None: {multiplier_char is None})")

                if multiplier_char is not None and multiplier_char.strip():  # FIXED: Check for None AND empty string
                    multiplier = multipliers.get(multiplier_char.lower().strip(), 1)
                    amount *= multiplier
                    print(f"  Applied multiplier {multiplier}: {amount}")
                else:
                    print(f"  No multiplier applied")

                # Determine context
                context = match.group(0).lower()
                print(f"  Context: '{context}'")

                if any(term in context for term in ['month', 'monthly', 'mo']):
                    metrics.update({'monthly_spend': amount, 'annual_spend': amount * 12})
                    print(f"  Set as monthly spend: ${amount:,.2f}")
                elif any(term in context for term in ['year', 'annual', 'yearly']):
                    metrics.update({'annual_spend': amount, 'monthly_spend': amount / 12})
                    print(f"  Set as annual spend: ${amount:,.2f}")
                else:
                    metrics['budget'] = amount
                    print(f"  Set as budget: ${amount:,.2f}")

                break  # Take first match per pattern

        # Handle percentage patterns
        for pattern in self.PERCENT_PATTERNS:
            matches = re.findall(pattern, query_lower)
            if matches:
                metrics['target_reduction'] = float(matches[0]) / 100
                print(f"  Found target reduction: {metrics['target_reduction']:.1%}")
                break

        # Apply contextual defaults if no explicit values found
        if contextual_defaults and not metrics:
            for key in ['suggested_budget', 'suggested_monthly_spend', 'suggested_annual_spend']:
                if key in contextual_defaults:
                    base_key = key.replace('suggested_', '')
                    metrics[base_key] = contextual_defaults[key]
                    print(f"  Applied contextual default {base_key}: {metrics[base_key]}")

        return metrics

    def _extract_usage_metrics(self, query: str, contextual_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced usage metrics extraction with context awareness"""
        metrics = {}
        query_lower = query.lower()

        # Extract from current query
        for metric, patterns in self.USAGE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    metrics[metric] = float(matches[0].replace(',', ''))
                    break

        # Calculate derived metrics
        if 'monthly_requests' in metrics and 'daily_requests' not in metrics:
            metrics['daily_requests'] = metrics['monthly_requests'] / 30
        elif 'daily_requests' in metrics and 'monthly_requests' not in metrics:
            metrics['monthly_requests'] = metrics['daily_requests'] * 30

        # Apply contextual defaults
        if contextual_defaults:
            for key in ['suggested_daily_requests', 'suggested_monthly_requests']:
                if key in contextual_defaults and key.replace('suggested_', '') not in metrics:
                    base_key = key.replace('suggested_', '')
                    metrics[base_key] = contextual_defaults[key]

        return metrics

    def _extract_providers(self, query: str, contextual_defaults: Dict[str, Any] = None) -> List[str]:
        """Enhanced provider extraction with context awareness"""
        query_lower = query.lower()
        providers = list(dict.fromkeys(provider for key, provider in self.PROVIDER_MAP.items()
                                     if key in query_lower))

        # If no providers found in query, use contextual suggestions
        if not providers and contextual_defaults and 'suggested_providers' in contextual_defaults:
            providers = contextual_defaults['suggested_providers'][:2]  # Top 2 from context

        return providers

    def _determine_use_case(self, query: str, session_id: str, contextual_defaults: Dict[str, Any] = None) -> str:
        """Enhanced use case determination with session locking"""

        # FIRST: Check if use case is already locked for this session
        locked_use_case = self.session_manager.get_locked_use_case(session_id)
        if locked_use_case:
            logger.info(f"Using locked use case for session {session_id}: {locked_use_case}")
            return locked_use_case

        # EXISTING LOGIC: Only runs for first query in session
        query_lower = query.lower()
        scores = {use_case: sum(1 for keyword in keywords if keyword in query_lower)
                  for use_case, keywords in self.USE_CASE_KEYWORDS.items()}

        scores = {k: v for k, v in scores.items() if v > 0}
        if scores:
            determined_use_case = max(scores.items(), key=lambda x: x[1])[0].replace("_", " ").title()
        elif contextual_defaults and 'suggested_use_case' in contextual_defaults:
            determined_use_case = contextual_defaults['suggested_use_case']
        else:
            determined_use_case = "General AI Implementation"

        # LOCK the use case for this session
        self.session_manager.lock_use_case(session_id, determined_use_case)
        logger.info(f"Locked use case for session {session_id}: {determined_use_case}")

        return determined_use_case

    def _safe_api_call(self, api_function, *args, **kwargs):
        """Safely make API calls with rate limiting"""
        try:
            # Wait for rate limiter
            self.rate_limiter.wait_if_needed()

            # Make the API call
            logger.info(f"Making API call to {api_function.__name__}")
            result = api_function(*args, **kwargs)

            logger.info(f"API call to {api_function.__name__} completed successfully")
            return result

        except Exception as e:
            logger.error(f"API call to {api_function.__name__} failed: {str(e)}")
            # Return a safe default or re-raise based on your needs
            return {"error": f"API call failed: {str(e)}"}

    def analyze_request(self, user_query: str, session_id: Optional[str] = None,
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced request analysis with rate limiting and conversation context memory"""
        # Generate session ID if not provided
        if not session_id:
            session_id = f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create session if it doesn't exist
        try:
            self.session_manager.create_session(session_id, user_id)
        except Exception as e:
            logger.warning(f"Session creation failed: {str(e)}")

        logger.info(f"Enhanced Orchestrator analyzing: {user_query[:100]}... (Session: {session_id})")

        # Get conversation history and context analysis
        try:
            history = self.session_manager.get_session_history(session_id)
            context_analysis = self.context_analyzer.analyze_context_relevance(user_query, history)
            contextual_defaults = context_analysis.get('recommended_defaults', {}) if context_analysis else {}
        except Exception as e:
            logger.warning(f"Context analysis failed: {str(e)}")
            context_analysis = {}
            contextual_defaults = {}

        # Enhanced extraction with context awareness
        intent, confidence = self._classify_intent(user_query, session_id)
        # financial_metrics = self._extract_financial_metrics(user_query, contextual_defaults)
        # usage_metrics = self._extract_usage_metrics(user_query, contextual_defaults)
        providers = self._extract_providers(user_query, contextual_defaults)
        use_case = self._determine_use_case(user_query, session_id, contextual_defaults)
        # all_metrics = {**financial_metrics, **usage_metrics}

        financial_metrics = self._extract_financial_metrics(user_query, contextual_defaults)
        usage_metrics = self._extract_usage_metrics(user_query, contextual_defaults)
        token_metrics = self._extract_token_metrics(user_query, contextual_defaults)
        all_metrics = {**financial_metrics, **usage_metrics, **token_metrics}

        logger.info(f"Intent: {intent} ({confidence:.2f}), Context Score: {context_analysis.get('context_score', 0):.2f}")

        # Initialize results structure
        results = {
            "query": user_query,
            "session_id": session_id,
            "intent": intent,
            "confidence": confidence,
            "use_case": use_case,
            "metrics": all_metrics,
            "providers": providers,
            "context": {
                "score": context_analysis.get('context_score', 0.0),
                "insights": [insight.__dict__ if hasattr(insight, '__dict__') else insight
                           for insight in context_analysis.get('insights', [])],
                "summary": context_analysis.get('context_summary', ''),
                "relevant_queries": len(context_analysis.get('relevant_context', []))
            },
            "analysis": {},
            "recommendations": []
        }

        try:
            # Perform analysis based on intent with rate limiting
            # Process sequentially to avoid overwhelming APIs

            if intent == "cost_analysis":
                logger.info("Starting cost analysis...")
                results["analysis"]["costs"] = self._safe_api_call(
                    self._get_cost_analysis, user_query, all_metrics, providers, use_case, contextual_defaults
                )

            elif intent == "task_analysis":
                logger.info("Starting task analysis...")
                results["analysis"]["tasks"] = self._safe_api_call(
                    self._get_task_analysis, user_query, context_analysis
                )

            elif intent == "roi_analysis":
                logger.info("Starting ROI analysis...")
                results["analysis"]["roi"] = self._safe_api_call(
                    self._get_roi_analysis, all_metrics, use_case, contextual_defaults
                )

            elif intent == "comprehensive":
                logger.info("Starting comprehensive analysis...")

                # Always do task analysis for comprehensive (it's the base analysis)
                results["analysis"]["tasks"] = self._safe_api_call(
                    self._get_task_analysis, user_query, context_analysis
                )

                # Add cost analysis if we have any metrics or providers
                if financial_metrics or usage_metrics or providers:
                    results["analysis"]["costs"] = self._safe_api_call(
                        self._get_cost_analysis, user_query, all_metrics, providers, use_case, contextual_defaults
                    )

                # Add ROI analysis if we have sufficient financial data
                if financial_metrics and (usage_metrics or 'budget' in financial_metrics):
                    results["analysis"]["roi"] = self._safe_api_call(
                        self._get_roi_analysis, all_metrics, use_case, contextual_defaults
                    )

            else:  # general intent
                logger.info("Starting general task analysis...")
                results["analysis"]["tasks"] = self._safe_api_call(
                    self._get_task_analysis, user_query, context_analysis
                )

            # Generate context-aware recommendations
            results["recommendations"] = self._generate_enhanced_recommendations(results, context_analysis)

            # Store query context for future reference
            try:
                query_context = QueryContext(
                    timestamp=datetime.now(),
                    query=user_query,
                    intent=intent,
                    confidence=confidence,
                    use_case=use_case,
                    extracted_metrics=all_metrics,
                    providers=providers,
                    analysis_results=results
                )
                self.session_manager.add_query_context(session_id, query_context)
            except Exception as e:
                logger.warning(f"Failed to store query context: {str(e)}")

        except Exception as e:
            logger.error(f"Enhanced Orchestrator error: {str(e)}")
            results.update({
                "error": str(e),
                "recommendations": ["Please provide more specific details for better analysis."]
            })

        return results

    def _get_cost_analysis(self, query: str, metrics: Dict, providers: List[str],
                           use_case: str, contextual_defaults: Dict[str, Any] = None) -> Dict:
        defaults = self._get_dynamic_use_case_defaults(query, use_case)

        # Override with contextual defaults if available
        if contextual_defaults:
            for key in ['suggested_daily_requests', 'suggested_input_tokens', 'suggested_output_tokens']:
                if key in contextual_defaults:
                    base_key = key.replace('suggested_', '')
                    defaults[base_key] = int(contextual_defaults[key])

        # Use extracted metrics or defaults
        requests_per_day = int(metrics.get('daily_requests', defaults["daily_requests"]))
        input_tokens = int(metrics.get('input_tokens', defaults["input_tokens"]))
        output_tokens = int(metrics.get('output_tokens', defaults["output_tokens"]))

        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]

        # Enhanced model selection based on providers and context
        provider_models = {
            "OpenAI API": ["gpt-4-turbo", "gpt-4o"],
            "Anthropic Claude": ["claude-3-opus", "claude-3-haiku"],
            "Google Gemini": ["gemini-pro", "gemini-1.5-pro"]
        }

        for provider in providers:
            if provider in provider_models:
                models.extend(provider_models[provider])

        return self.cost_agent.calculate_llm_costs(
            use_case=use_case,
            requests_per_day=requests_per_day,
            avg_input_tokens=input_tokens,
            avg_output_tokens=output_tokens,
            models=list(set(models))
        )

    def _get_task_analysis(self, query: str, context_analysis: Dict[str, Any]) -> Dict:
        """Enhanced task analysis with rate limiting protection"""
        task_result = self.task_agent.analyze_workflow(query)

        # Enhance with context insights
        if context_analysis and context_analysis.get('insights'):
            try:
                task_result['context_insights'] = [
                    insight for insight in context_analysis['insights']
                    if isinstance(insight, dict) and insight.get('type') in ['pattern', 'evolution']
                ]
            except Exception as e:
                logger.warning(f"Failed to add context insights: {str(e)}")

        return task_result.get("analysis", task_result) if isinstance(task_result, dict) else task_result

    def _get_roi_analysis(self, metrics: Dict, use_case: str, contextual_defaults: Dict[str, Any] = None) -> Dict:
        """Enhanced ROI analysis with rate limiting protection"""
        # Determine budget with context awareness
        budget = metrics.get('budget', metrics.get('monthly_spend', 10000))
        if 'monthly_spend' in metrics:
            budget = metrics['monthly_spend'] * 12
        elif contextual_defaults and 'suggested_budget' in contextual_defaults:
            budget = contextual_defaults['suggested_budget']

        # Enhanced benefit estimation based on context
        hours_saved = metrics.get('hours_saved', 1000)
        if contextual_defaults and 'suggested_hours_saved' in contextual_defaults:
            hours_saved = contextual_defaults['suggested_hours_saved']

        hourly_rate = 50

        annual_benefits = {
            "Labor Cost Savings": hours_saved * hourly_rate,
            "Efficiency Gains": budget * 0.3,
            "Error Reduction": budget * 0.15,
            "Speed Improvements": budget * 0.2
        }

        return self.roi_agent.calculate_roi(
            project_name=f"AI Implementation - {use_case}",
            implementation_cost=budget,
            annual_benefits=annual_benefits
        )

    def _generate_enhanced_recommendations(self, results: Dict[str, Any],
                                         context_analysis: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations with context awareness"""
        recommendations = []

        # Context-aware recommendations
        try:
            if context_analysis and hasattr(self.context_analyzer, 'get_context_based_recommendations'):
                context_recs = self.context_analyzer.get_context_based_recommendations(
                    results["query"], context_analysis
                )
                if context_recs:
                    recommendations.extend(context_recs)
        except Exception as e:
            logger.warning(f"Failed to get context-based recommendations: {str(e)}")

        # Analysis-based recommendations
        if "costs" in results["analysis"] and isinstance(results["analysis"]["costs"], dict):
            cost_data = results["analysis"]["costs"].get("cost_breakdown", {})
            if cost_data:
                try:
                    cheapest = min(cost_data.items(), key=lambda x: x[1].get("monthly_cost", float('inf')))
                    recommendations.append(
                        f"💰 **Most Cost-Effective**: {cheapest[0]} at ${cheapest[1].get('monthly_cost', 0):,.0f}/month"
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate cost recommendation: {str(e)}")

        if "roi" in results["analysis"] and isinstance(results["analysis"]["roi"], dict):
            roi_data = results["analysis"]["roi"].get("basic_metrics", {})
            payback = roi_data.get("payback_period_years", 0)
            roi_pct = roi_data.get("roi_percentage", 0)

            if payback > 0:
                try:
                    if payback < 1:
                        recommendations.append(
                            f"🚀 **Excellent ROI**: {roi_pct:.0f}% return with {payback*12:.1f} month payback"
                        )
                    else:
                        recommendations.append(
                            f"📈 **Strong ROI**: {roi_pct:.0f}% return with {payback:.1f} year payback"
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate ROI recommendation: {str(e)}")

        # Context-specific recommendations
        context_score = results["context"]["score"]
        if context_score > 0.5:
            recommendations.append(
                f"🔗 **Context Continuity**: Building on {results['context']['relevant_queries']} related queries"
            )

        # Intent-based recommendations
        if results["intent"] == "comprehensive":
            recommendations.append("📊 **Next Steps**: Consider a phased implementation starting with highest ROI tasks")
        elif results["confidence"] < 0.5:
            recommendations.append("❓ **Clarification**: Provide more specific requirements for targeted recommendations")

        # Rate limiting awareness
        recommendations.append("⚡ **API Optimization**: Requests are automatically rate-limited for optimal performance")

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:6]  # Limit to top 6 recommendations

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

    # Backward compatibility - keep original method name
    def analyze_request_legacy(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        return self.analyze_request(user_query, session_id)