"""Multi-Agent Orchestrator for coordinating agent responses"""
from typing import Dict, Any, Optional, List
import json
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

    # Class-level constants
    INTENT_KEYWORDS = {
        "cost_analysis": ["cost", "price", "expensive", "cheap", "budget", "pricing", "compare llm"],
        "task_analysis": ["automate", "task", "workflow", "process", "implement", "ai for"],
        "roi_analysis": ["roi", "return", "investment", "payback", "benefit", "value", "worth"]
    }

    PROVIDER_MAP = {
        "aws": "AWS Bedrock", "bedrock": "AWS Bedrock", "openai": "OpenAI API",
        "azure": "Azure OpenAI", "anthropic": "Anthropic Claude",
        "gemini": "Google Gemini", "vertex": "Vertex AI"
    }

    def __init__(self):
        """Initialize all agents"""
        self.task_agent = TaskAnalyzerAgent(agent_key="task")
        self.cost_agent = CostCalculatorAgent(agent_key="cost")
        self.roi_agent = ROIEstimatorAgent(agent_key="roi")

        # Initialize agents in Lyzr (if method exists)
        for agent in [self.task_agent, self.cost_agent, self.roi_agent]:
            if hasattr(agent, 'create_or_get_agent'):
                agent.create_or_get_agent()

        logger.info("Orchestrator initialized with all agents")

    @lru_cache(maxsize=256)
    def _classify_intent(self, user_query: str) -> str:
        """Classify user intent from query (cached)"""
        query_lower = user_query.lower()

        scores = {
            intent: sum(1 for kw in keywords if kw in query_lower)
            for intent, keywords in self.INTENT_KEYWORDS.items()
        }

        active_intents = sum(1 for score in scores.values() if score > 0)

        if active_intents == 0:
            return "general"
        elif active_intents > 1:
            return "comprehensive"

        return max(scores.items(), key=lambda x: x[1])[0]

    def _extract_metrics(self, query: str) -> Dict[str, Any]:
        """Extract numerical metrics from query"""
        metrics = {}

        # Dollar amounts with multipliers
        dollar_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d+)?)[kKmM]?'
        for match in re.finditer(dollar_pattern, query):
            amount = float(match.group(1).replace(',', ''))

            # Apply multipliers
            if match.group(0).lower().endswith(('k', 'K')):
                amount *= 1000
            elif match.group(0).lower().endswith(('m', 'M')):
                amount *= 1000000

            # Context-based assignment
            if 'month' in query.lower():
                metrics.update({'monthly_spend': amount, 'annual_spend': amount * 12})
            elif any(term in query.lower() for term in ['year', 'annual']):
                metrics.update({'annual_spend': amount, 'monthly_spend': amount / 12})
            else:
                metrics['budget'] = amount

        # Percentages
        percent_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', query)
        if percent_matches:
            metrics['target_reduction'] = float(percent_matches[0]) / 100

        # Numbers with context
        context_patterns = {
            'daily_requests': [r'(\d+(?:,\d{3})*)\s*(?:requests?|tickets?)'],
            'users': [r'(\d+(?:,\d{3})*)\s*(?:users?|customers?)'],
            'hours': [r'(\d+(?:,\d{3})*)\s*hours?'],
            'minutes': [r'(\d+(?:,\d{3})*)\s*minutes?']
        }

        for metric, patterns in context_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    metrics[metric] = float(matches[0].replace(',', ''))
                    break

        # Special case: "50K/month"
        special_matches = re.findall(r'(\d+)[kK]/month', query)
        if special_matches:
            metrics['monthly_spend'] = float(special_matches[0]) * 1000

        logger.info(f"Extracted metrics: {metrics}")
        return metrics

    def _extract_providers(self, query: str) -> List[str]:
        """Extract provider names from query"""
        return [self.PROVIDER_MAP[key] for key in self.PROVIDER_MAP
                if key in query.lower()]

    def _is_infrastructure_query(self, query: str) -> bool:
        """Check if query is about infrastructure costs"""
        return ("infrastructure" in query.lower() or
                ("spend" in query.lower() and
                 any(provider in query.lower() for provider in self.PROVIDER_MAP)))

    def analyze_request(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze user request and coordinate agent responses"""
        session_id = session_id or f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Orchestrator analyzing request: {user_query[:50]}...")

        intent = self._classify_intent(user_query)
        metrics = self._extract_metrics(user_query)

        logger.info(f"Classified intent: {intent}, Metrics: {metrics}")

        results = {
            "query": user_query,
            "intent": intent,
            "metrics": metrics,
            "analysis": {},
            "recommendations": []
        }

        try:
            # Infrastructure cost analysis
            if self._is_infrastructure_query(user_query):
                logger.info("Detected infrastructure cost analysis request")

                providers = self._extract_providers(user_query) or ["AWS Bedrock", "OpenAI API", "Azure OpenAI"]
                monthly_spend = metrics.get('monthly_spend', metrics.get('budget', 50000))
                target_reduction = metrics.get('target_reduction', 0.3)

                infra_result = self.cost_agent.analyze_infrastructure_costs(
                    current_spend=monthly_spend,
                    providers=providers,
                    target_reduction=target_reduction
                )
                results["analysis"]["infrastructure"] = infra_result

                # General cost comparison
                cost_result = self.cost_agent.calculate_llm_costs(
                    use_case="Enterprise AI Infrastructure",
                    requests_per_day=int(monthly_spend / 30 / 0.01),
                    avg_input_tokens=200,
                    avg_output_tokens=300,
                    models=["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "gemini-pro"]
                )
                results["analysis"]["costs"] = cost_result
            elif intent in ["cost_analysis", "comprehensive"]:
                logger.info("Getting cost analysis...")
                requests_per_day = int(metrics.get('daily_requests', metrics.get('daily_tickets', 1000)))

                cost_result = self.cost_agent.calculate_llm_costs(
                    use_case=user_query[:100],
                    requests_per_day=requests_per_day,
                    avg_input_tokens=150,
                    avg_output_tokens=200,
                    models=["gpt-4", "gpt-3.5-turbo", "gemini-pro", "claude-3-sonnet"]
                )
                results["analysis"]["costs"] = cost_result

            # Route to specific agents based on intent
            if intent in ["task_analysis", "comprehensive", "general"]:
                logger.info("Getting task analysis...")
                task_result = self.task_agent.analyze_workflow(user_query)
                results["analysis"]["tasks"] = task_result["analysis"]

            if intent in ["roi_analysis", "comprehensive"]:
                logger.info("Getting ROI analysis...")
                budget = metrics.get('budget', metrics.get('monthly_spend', 50000))

                roi_result = self.roi_agent.calculate_roi(
                    project_name="AI Implementation",
                    implementation_cost=budget,
                    annual_benefits={
                        "Labor Cost Savings": budget * 2.4,
                        "Efficiency Gains": budget * 0.6,
                        "Error Reduction": budget * 0.3
                    }
                )
                results["analysis"]["roi"] = roi_result

            results["recommendations"] = self._generate_recommendations(results)

        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            results.update({"error": str(e),
                          "recommendations": ["An error occurred. Please try rephrasing your question."]})

        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Infrastructure recommendations
        if "infrastructure" in results["analysis"]:
            infra = results["analysis"]["infrastructure"]
            savings = infra.get("potential_savings", 0)
            reduction = infra.get("reduction_percentage", 0)

            recommendations.extend([
                f"💰 **Cost Reduction Opportunity**: Save ${savings:,.0f}/month ({reduction:.0f}% reduction)",
                f"🎯 **Target Spend**: Reduce from ${infra['current_spend']:,.0f} to ${infra['target_spend']:,.0f}/month"
            ])

        # Cost analysis recommendations
        if "costs" in results["analysis"]:
            cost_breakdown = results["analysis"]["costs"].get("cost_breakdown", {})
            if cost_breakdown:
                cheapest = min(cost_breakdown.items(), key=lambda x: x[1]["monthly_cost"])
                recommendations.append(
                    f"💡 **Most Cost-Effective**: {cheapest[0]} at ${cheapest[1]['monthly_cost']:,.0f}/month"
                )

        # ROI recommendations
        if "roi" in results["analysis"]:
            roi_metrics = results["analysis"]["roi"].get("basic_metrics", {})
            if roi_metrics:
                payback = roi_metrics.get("payback_period_years", 0)
                roi_pct = roi_metrics.get("roi_percentage", 0)
                if payback > 0:
                    recommendations.append(
                        f"📈 **ROI Projection**: {roi_pct:.0f}% return with {payback:.1f} year payback"
                    )

        # Task analysis recommendations
        if "tasks" in results["analysis"]:
            recommendations.append("📋 **Action Plan**: Start with high-impact, low-complexity automations")

        return recommendations or ["💡 **Next Step**: Provide more details for specific recommendations"]

    def get_summary(self, session_id: str) -> str:
        """Get a summary of the analysis session"""
        return "Session summary will be implemented based on conversation history"