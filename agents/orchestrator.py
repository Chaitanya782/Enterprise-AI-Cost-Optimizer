"""Multi-Agent Orchestrator for coordinating agent responses"""
from typing import Dict, Any, Optional, List, Tuple
import re
from datetime import datetime
from pathlib import Path
import sys
from functools import lru_cache

sys.path.append(str(Path(__file__).parent.parent))

from agents.task_analyzer import TaskAnalyzerAgent
from agents.cost_calculator import CostCalculatorAgent
from agents.roi_estimator import ROIEstimatorAgent
from utils.logger import logger

class Orchestrator:
    """Orchestrates multiple agents for comprehensive analysis"""

    INTENT_KEYWORDS = {
        "cost_analysis": ["cost", "price", "expensive", "cheap", "budget", "pricing", "spend", "fee",
                         "compare llm", "llm cost", "api cost", "monthly cost", "save money", "reduce cost"],
        "task_analysis": ["automate", "task", "workflow", "process", "implement", "ai for", "streamline",
                         "optimize", "replace", "manual", "repetitive", "efficiency", "productivity"],
        "roi_analysis": ["roi", "return", "investment", "payback", "benefit", "value", "worth", "profit",
                        "savings", "break even", "business case", "justification", "cost benefit"]
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

    USE_CASE_DEFAULTS = {
        "Customer Support": {"input_tokens": 100, "output_tokens": 150, "daily_requests": 500},
        "Content Generation": {"input_tokens": 50, "output_tokens": 400, "daily_requests": 200},
        "Data Analysis": {"input_tokens": 300, "output_tokens": 200, "daily_requests": 100},
        "Code Generation": {"input_tokens": 200, "output_tokens": 500, "daily_requests": 150}
    }

    FINANCIAL_PATTERNS = [
        r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?(?:\s*(?:per\s+)?(?:month|monthly|mo))?',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|usd)\s*([kmb])?',
        r'budget\s*(?:of|is)?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)([kmb])?'
    ]

    PERCENT_PATTERNS = [
        r'(?:reduce|save|cut|decrease)\s*(?:by|up\s*to)?\s*(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|savings|decrease)',
        r'target\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*%'
    ]

    USAGE_PATTERNS = {
        'daily_requests': [r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?)\s*(?:per\s*day|daily)',
                          r'(\d+(?:,\d{3})*)\s*daily\s*(?:requests?|calls?|queries?)'],
        'monthly_requests': [r'(\d+(?:,\d{3})*)\s*(?:requests?|calls?|queries?)\s*(?:per\s*month|monthly)',
                            r'(\d+(?:,\d{3})*)\s*monthly\s*(?:requests?|calls?|queries?)'],
        'users': [r'(\d+(?:,\d{3})*)\s*(?:users?|customers?|employees?)', r'team\s*of\s*(\d+(?:,\d{3})*)'],
        'hours_saved': [r'save\s*(\d+(?:,\d{3})*)\s*hours?', r'(\d+(?:,\d{3})*)\s*hours?\s*(?:saved|reduction)']
    }

    def __init__(self):
        """Initialize all agents"""
        self.task_agent = TaskAnalyzerAgent(agent_key="task")
        self.cost_agent = CostCalculatorAgent(agent_key="cost")
        self.roi_agent = ROIEstimatorAgent(agent_key="roi")

        for agent in [self.task_agent, self.cost_agent, self.roi_agent]:
            if hasattr(agent, 'create_or_get_agent'):
                agent.create_or_get_agent()

        logger.info("Orchestrator initialized with all agents")

    @lru_cache(maxsize=256)
    def _classify_intent(self, user_query: str) -> Tuple[str, float]:
        """Enhanced intent classification with confidence scoring"""
        query_lower = user_query.lower()
        scores = {}

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(len(keyword.split()) for keyword in keywords if keyword in query_lower)
            scores[intent] = score

        total_score = sum(scores.values())
        if total_score == 0:
            return "general", 0.0

        max_intent = max(scores.items(), key=lambda x: x[1])
        confidence = max_intent[1] / total_score

        active_intents = [intent for intent, score in scores.items() if score > 0]
        if len(active_intents) > 1 and confidence < 0.6:
            return "comprehensive", confidence

        return max_intent[0], confidence

    def _extract_financial_metrics(self, query: str) -> Dict[str, Any]:
        """Enhanced financial data extraction"""
        metrics = {}
        query_lower = query.lower()
        multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}

        for pattern in self.FINANCIAL_PATTERNS:
            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                amount = float(match.group(1).replace(',', ''))
                if match.group(2):
                    amount *= multipliers.get(match.group(2).lower(), 1)

                context = match.group(0).lower()
                if any(term in context for term in ['month', 'monthly', 'mo']):
                    metrics.update({'monthly_spend': amount, 'annual_spend': amount * 12})
                elif any(term in context for term in ['year', 'annual', 'yearly']):
                    metrics.update({'annual_spend': amount, 'monthly_spend': amount / 12})
                else:
                    metrics['budget'] = amount

        for pattern in self.PERCENT_PATTERNS:
            matches = re.findall(pattern, query_lower)
            if matches:
                metrics['target_reduction'] = float(matches[0]) / 100
                break

        return metrics

    def _extract_usage_metrics(self, query: str) -> Dict[str, Any]:
        """Extract usage-related metrics"""
        metrics = {}
        query_lower = query.lower()

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

        return metrics

    def _extract_providers(self, query: str) -> List[str]:
        """Enhanced provider extraction"""
        query_lower = query.lower()
        return list(dict.fromkeys(provider for key, provider in self.PROVIDER_MAP.items()
                                if key in query_lower))

    def _determine_use_case(self, query: str) -> str:
        """Determine the primary use case from query"""
        query_lower = query.lower()
        scores = {use_case: sum(1 for keyword in keywords if keyword in query_lower)
                 for use_case, keywords in self.USE_CASE_KEYWORDS.items()}

        scores = {k: v for k, v in scores.items() if v > 0}
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0].replace("_", " ").title()
        return "General AI Implementation"

    def analyze_request(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced request analysis with better coordination"""
        session_id = session_id or f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Orchestrator analyzing: {user_query[:100]}...")

        intent, confidence = self._classify_intent(user_query)
        financial_metrics = self._extract_financial_metrics(user_query)
        usage_metrics = self._extract_usage_metrics(user_query)
        providers = self._extract_providers(user_query)
        use_case = self._determine_use_case(user_query)
        all_metrics = {**financial_metrics, **usage_metrics}

        logger.info(f"Intent: {intent} ({confidence:.2f}), Metrics: {all_metrics}")

        results = {
            "query": user_query, "intent": intent, "confidence": confidence,
            "use_case": use_case, "metrics": all_metrics, "providers": providers,
            "analysis": {}, "recommendations": []
        }

        try:
            if intent in ["cost_analysis", "comprehensive"] or financial_metrics:
                results["analysis"]["costs"] = self._get_cost_analysis(user_query, all_metrics, providers, use_case)

            if intent in ["task_analysis", "comprehensive", "general"] or not financial_metrics:
                results["analysis"]["tasks"] = self._get_task_analysis(user_query)

            if intent in ["roi_analysis", "comprehensive"] or (financial_metrics and usage_metrics):
                results["analysis"]["roi"] = self._get_roi_analysis(all_metrics, use_case)

            results["recommendations"] = self._generate_smart_recommendations(results)

        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            results.update({"error": str(e), "recommendations": ["Please provide more specific details for better analysis."]})

        return results

    def _get_cost_analysis(self, query: str, metrics: Dict, providers: List[str], use_case: str) -> Dict:
        """Get cost analysis with better parameters"""
        defaults = self.USE_CASE_DEFAULTS.get(use_case, {"input_tokens": 150, "output_tokens": 200, "daily_requests": 300})
        requests_per_day = int(metrics.get('daily_requests', defaults["daily_requests"]))
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]

        provider_models = {
            "OpenAI API": ["gpt-4-turbo", "gpt-4o"],
            "Anthropic Claude": ["claude-3-opus", "claude-3-haiku"]
        }

        for provider in providers:
            if provider in provider_models:
                models.extend(provider_models[provider])

        return self.cost_agent.calculate_llm_costs(
            use_case=use_case, requests_per_day=requests_per_day,
            avg_input_tokens=defaults["input_tokens"], avg_output_tokens=defaults["output_tokens"],
            models=list(set(models))
        )

    def _get_task_analysis(self, query: str) -> Dict:
        """Get task analysis"""
        task_result = self.task_agent.analyze_workflow(query)
        return task_result.get("analysis", task_result)

    def _get_roi_analysis(self, metrics: Dict, use_case: str) -> Dict:
        """Get ROI analysis with smart benefit estimation"""
        budget = metrics.get('budget', metrics.get('monthly_spend', 10000))
        if 'monthly_spend' in metrics:
            budget = metrics['monthly_spend'] * 12

        hours_saved = metrics.get('hours_saved', 1000)
        hourly_rate = 50

        annual_benefits = {
            "Labor Cost Savings": hours_saved * hourly_rate,
            "Efficiency Gains": budget * 0.3,
            "Error Reduction": budget * 0.15,
            "Speed Improvements": budget * 0.2
        }

        return self.roi_agent.calculate_roi(
            project_name=f"AI Implementation - {use_case}",
            implementation_cost=budget, annual_benefits=annual_benefits
        )

    def _generate_smart_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate contextual recommendations"""
        recommendations = []

        if "costs" in results["analysis"]:
            cost_data = results["analysis"]["costs"].get("cost_breakdown", {})
            if cost_data:
                cheapest = min(cost_data.items(), key=lambda x: x[1].get("monthly_cost", float('inf')))
                recommendations.append(f"💰 **Most Cost-Effective**: {cheapest[0]} at ${cheapest[1].get('monthly_cost', 0):,.0f}/month")

        if "roi" in results["analysis"]:
            roi_data = results["analysis"]["roi"].get("basic_metrics", {})
            payback = roi_data.get("payback_period_years", 0)
            roi_pct = roi_data.get("roi_percentage", 0)

            if payback > 0:
                if payback < 1:
                    recommendations.append(f"🚀 **Excellent ROI**: {roi_pct:.0f}% return with {payback*12:.1f} month payback")
                else:
                    recommendations.append(f"📈 **Strong ROI**: {roi_pct:.0f}% return with {payback:.1f} year payback")

        if results.get("providers"):
            recommendations.append(f"🎯 **Focus Areas**: Evaluate {', '.join(results['providers'])} for your use case")

        if results["intent"] == "comprehensive":
            recommendations.append("📊 **Next Steps**: Consider a phased implementation starting with highest ROI tasks")
        elif results["confidence"] < 0.5:
            recommendations.append("❓ **Clarification**: Provide more specific requirements for targeted recommendations")

        return recommendations or ["💡 **Get Started**: Define your specific use case and budget for detailed analysis"]

    def get_summary(self, session_id: str) -> str:
        """Get analysis session summary"""
        return f"Analysis session {session_id} completed with multi-agent coordination"