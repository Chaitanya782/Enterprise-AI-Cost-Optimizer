"""Cost Calculator Agent - Calculates and compares AI implementation costs"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import json
from functools import lru_cache

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger

class CostCalculatorAgent(BaseAgent):
    """Agent that calculates AI implementation and operational costs"""

    # Class-level pricing data (immutable)
    LLM_PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.0008, "output": 0.004},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "llama-2-70b": {"input": 0.0008, "output": 0.0024},  # Approximate
        "mistral-7b": {"input": 0.0002, "output": 0.0002}  # Approximated for inference
    }

    def __init__(self, agent_key: str = "cost"):
        super().__init__(name="Cost Calculator", description="Calculates automation cost...", agent_key=agent_key)

    def get_system_prompt(self) -> str:
        return """You are an expert AI Cost Calculator specializing in enterprise AI implementations.

                Your role is to:
                1. Calculate detailed costs for AI implementations
                2. Compare different LLM options based on use case requirements
                3. Identify hidden costs (infrastructure, maintenance, training)
                4. Provide cost optimization strategies
                5. Create detailed cost breakdowns and projections
                
                When calculating costs, consider:
                - API costs (per token/request)
                - Infrastructure costs (servers, storage, networking)
                - Development and integration costs
                - Maintenance and monitoring costs
                - Training and support costs
                - Scaling costs
                
                Always provide:
                - Detailed cost breakdowns
                - Monthly and annual projections
                - Cost comparison tables
                - Optimization recommendations
                - ROI calculations
                
                Be precise with numbers and always show your calculations."""

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculate_token_usage",
                "description": "Calculate estimated token usage for a use case",
                "parameters": {
                    "requests_per_day": "number",
                    "avg_input_length": "number",
                    "avg_output_length": "number"
                }
            },
            {
                "name": "compare_llm_costs",
                "description": "Compare costs across different LLMs",
                "parameters": {
                    "use_case": "string",
                    "monthly_requests": "number",
                    "performance_requirements": "string"
                }
            }
        ]

    @lru_cache(maxsize=128)
    def _calculate_model_cost(self, model: str, requests_per_day: int,
                             avg_input_tokens: int, avg_output_tokens: int) -> Dict[str, float]:
        """Calculate costs for a specific model (cached for performance)"""
        if model not in self.LLM_PRICING:
            return {}

        pricing = self.LLM_PRICING[model]
        daily_cost = ((avg_input_tokens * pricing["input"] +
                      avg_output_tokens * pricing["output"]) * requests_per_day / 1000)

        return {
            "daily_cost": round(daily_cost, 2),
            "monthly_cost": round(daily_cost * 30, 2),
            "annual_cost": round(daily_cost * 365, 2),
            "cost_per_request": round(daily_cost / requests_per_day, 4)
        }

    def calculate_llm_costs(self, use_case: str, requests_per_day: int,
                           avg_input_tokens: int, avg_output_tokens: int,
                           models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate costs for different LLM options"""
        logger.info(f"Calculating LLM costs for use case: {use_case}")

        models = models or list(self.LLM_PRICING.keys())
        cost_analysis = {
            model: self._calculate_model_cost(model, requests_per_day,
                                            avg_input_tokens, avg_output_tokens)
            for model in models if model in self.LLM_PRICING
        }

        prompt = f"""Based on the following cost analysis for "{use_case}":

                Cost Analysis:
                {json.dumps(cost_analysis, indent=2)}
                
                Requests per day: {requests_per_day}
                Average input tokens: {avg_input_tokens}
                Average output tokens: {avg_output_tokens}
                
                Provide:
                1. Recommended model based on cost-performance ratio
                2. Potential cost optimization strategies
                3. Hidden costs to consider
                4. Long-term scaling considerations"""

        response = self.chat(prompt)
        return {
            "cost_breakdown": cost_analysis,
            "recommendations": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def estimate_total_implementation_cost(self, project_description: str,
                                         team_size: int, timeline_weeks: int) -> Dict[str, Any]:
        """Estimate total implementation costs including development"""
        logger.info("Estimating total implementation costs")

        prompt = f"""Estimate the total implementation cost for the following AI project:

                    Project Description: {project_description}
                    Team Size: {team_size} developers
                    Timeline: {timeline_weeks} weeks
                    
                    Calculate:
                    1. Development costs (assuming $150/hour average rate)
                    2. Infrastructure setup costs
                    3. LLM API costs for development and testing
                    4. Additional tools and services
                    5. Training and documentation costs
                    6. First-year operational costs
                    
                    Provide a detailed breakdown with assumptions clearly stated."""

        response = self.chat(prompt)
        return {
            "cost_estimate": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def analyze_infrastructure_costs(self, current_spend: float, providers: List[str],
                                   target_reduction: float = 0.3) -> Dict[str, Any]:
        """Analyze current infrastructure costs with specific recommendations"""
        logger.info(f"Analyzing infrastructure costs: ${current_spend}/month")

        context = {
            "current_monthly_spend": current_spend,
            "annual_spend": current_spend * 12,
            "providers": providers,
            "target_reduction_percentage": target_reduction * 100,
            "target_savings": current_spend * target_reduction
        }

        prompt = f"""Analyze this AI infrastructure spending scenario:

        CURRENT SITUATION:
        - Monthly Spend: ${current_spend:,.0f}
        - Annual Spend: ${current_spend * 12:,.0f}
        - Providers: {', '.join(providers)}
        - Target Reduction: {target_reduction * 100}% (${current_spend * target_reduction:,.0f}/month)
        
        Provide a SPECIFIC analysis with:
        
        1. CURRENT COST BREAKDOWN:
           - Estimate percentage split across providers
           - Identify biggest cost drivers (compute, API calls, storage)
           - Calculate per-request costs
        
        2. OPTIMIZATION STRATEGIES (with specific numbers):
           - Strategy 1: [Specific action] → Saves $X/month (Y%)
           - Strategy 2: [Specific action] → Saves $X/month (Y%)
           - Strategy 3: [Specific action] → Saves $X/month (Y%)
        
        3. ALTERNATIVE SOLUTIONS:
           - Option A: [Provider/Solution] - $X/month (Y% savings)
           - Option B: [Provider/Solution] - $X/month (Y% savings)
        
        4. RISK ASSESSMENT:
           - Low Risk: [Actions with minimal impact]
           - Medium Risk: [Actions requiring testing]
           - High Risk: [Actions affecting production]
        
        5. IMPLEMENTATION TIMELINE:
           - Week 1-2: [Quick wins]
           - Week 3-4: [Medium-term changes]
           - Month 2-3: [Long-term optimizations]
        
        6. 12-MONTH ROI PROJECTION:
           - Month 1: Implement [X] → Save $Y
           - Month 3: Complete [X] → Total savings $Y
           - Month 6: Achieve [X] → Total savings $Y
           - Month 12: Full optimization → Total savings $Y
        
        Be specific with numbers and avoid generic advice."""

        response = self.chat(prompt, context=context)
        potential_savings = current_spend * target_reduction

        return {
            "current_spend": current_spend,
            "target_spend": current_spend - potential_savings,
            "potential_savings": potential_savings,
            "annual_savings": potential_savings * 12,
            "reduction_percentage": target_reduction * 100,
            "detailed_analysis": response["content"],
            "providers": providers,
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def calculate_migration_costs(self, from_providers: List[str], to_providers: List[str],
                                monthly_volume: int) -> Dict[str, Any]:
        """Calculate specific migration costs and timeline"""
        prompt = f"""Calculate migration costs for:

                    FROM: {', '.join(from_providers)}
                    TO: {', '.join(to_providers)}
                    Monthly Request Volume: {monthly_volume:,}
                    
                    Provide:
                    1. One-time migration costs (engineering, testing, deployment)
                    2. Monthly cost comparison (before vs after)
                    3. Break-even timeline
                    4. Risk factors and mitigation costs"""

        response = self.chat(prompt)
        return {
            "analysis": response["content"],
            "from_providers": from_providers,
            "to_providers": to_providers,
            "volume": monthly_volume
        }