"""ROI Estimator Agent - Calculates return on investment for AI initiatives"""
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger


class ROIEstimatorAgent(BaseAgent):
    """Agent that estimates ROI for AI implementations"""

    def __init__(self, agent_key: str = "roi"):
        super().__init__(name="ROI Calculator", description="Estimates return on investment...", agent_key=agent_key)

    def get_system_prompt(self) -> str:
        return """Expert AI ROI Analyst specializing in quantifying business value of AI implementations.

                    Calculate detailed ROI considering:
                    - Direct savings (labor, time, resources) & revenue increases
                    - Quality improvements & strategic advantages  
                    - Risk reduction & compliance benefits
                    
                    Always provide: ROI percentage, payback period, monthly/annual projections, 
                    break-even analysis, sensitivity analysis, risk-adjusted returns.
                    Use conservative estimates and industry benchmarks."""

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "calculate_time_savings",
                "description": "Calculate time savings from AI automation",
                "parameters": {
                    "task_hours_per_week": "number",
                    "automation_percentage": "number",
                    "number_of_employees": "number"
                }
            },
            {
                "name": "calculate_error_reduction_value",
                "description": "Calculate value of error reduction",
                "parameters": {
                    "current_error_rate": "number",
                    "expected_error_rate": "number",
                    "cost_per_error": "number",
                    "transactions_per_month": "number"
                }
            }
        ]

    def _format_benefits(self, benefits: Dict[str, float]) -> str:
        """Format benefits dictionary for display"""
        return "\n".join(f"- {k}: ${v:,.2f}" for k, v in benefits.items())

    def _calculate_basic_metrics(self, impl_cost: float, annual_benefit: float, years: int) -> Dict[str, float]:
        """Calculate basic ROI metrics"""
        if annual_benefit <= 0 or impl_cost <= 0:
            return {"payback_period_years": float('inf'), "roi_percentage": 0, "net_benefit": -impl_cost}

        payback = impl_cost / annual_benefit
        total_benefit = annual_benefit * years
        net_benefit = total_benefit - impl_cost
        roi = (net_benefit / impl_cost) * 100

        return {
            "payback_period_years": round(payback, 1),
            "roi_percentage": round(roi, 1),
            "net_benefit": round(net_benefit, 2)
        }

    def calculate_roi(self, project_name: str, implementation_cost: float,
                     annual_benefits: Dict[str, float], timeline_years: int = 3) -> Dict[str, Any]:
        """Calculate comprehensive ROI analysis"""
        logger.info(f"Calculating ROI for project: {project_name}")

        total_annual_benefit = sum(annual_benefits.values())
        basic_metrics = self._calculate_basic_metrics(implementation_cost, total_annual_benefit, timeline_years)

        prompt = f"""Comprehensive ROI analysis for AI project "{project_name}":

                Implementation Cost: ${implementation_cost:,.2f}
                Annual Benefits: {self._format_benefits(annual_benefits)}
                Total Annual: ${total_annual_benefit:,.2f} | Timeline: {timeline_years}y
                
                Basic: Payback {basic_metrics['payback_period_years']}y | ROI {basic_metrics['roi_percentage']}%
                
                Provide: 1) NPV analysis (10% discount) 2) Risk/sensitivity analysis 
                3) Intangible benefits 4) Industry benchmarks 5) ROI optimization recommendations"""

        response = self.chat(prompt)

        return {
            "basic_metrics": {
                "implementation_cost": implementation_cost,
                "annual_benefit": total_annual_benefit,
                **basic_metrics
            },
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def analyze_business_impact(self, use_case: str, current_metrics: Dict[str, Any],
                              expected_improvements: Dict[str, str]) -> Dict[str, Any]:
        """Analyze broader business impact beyond direct ROI"""
        logger.info(f"Analyzing business impact for: {use_case}")

        current_str = "\n".join(f"- {k}: {v}" for k, v in current_metrics.items())
        improvements_str = "\n".join(f"- {k}: {v}" for k, v in expected_improvements.items())

        prompt = f"""Business impact analysis for AI implementation: {use_case}

                    Current: {current_str}
                    Expected: {improvements_str}
                    
                    Analyze: 1) Revenue impact 2) Cost reduction 3) Customer satisfaction 
                    4) Competitive advantage 5) Operational efficiency 6) Risk mitigation 7) Strategic value
                    Quantify where possible with specific examples."""

        response = self.chat(prompt)
        return {
            "impact_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def calculate_optimization_roi(self, current_spend: float, target_savings: float,
                                 implementation_cost: float) -> Dict[str, Any]:
        """Calculate ROI for cost optimization initiatives"""
        monthly_benefit = target_savings
        annual_benefit = monthly_benefit * 12
        payback_months = implementation_cost / monthly_benefit if monthly_benefit > 0 else float('inf')
        first_year_net = annual_benefit - implementation_cost
        roi_percentage = (first_year_net / implementation_cost * 100) if implementation_cost > 0 else 0

        prompt = f"""AI cost optimization ROI analysis:

                    Monthly Spend: ${current_spend:,.0f} | Target Savings: ${target_savings:,.0f}
                    Implementation: ${implementation_cost:,.0f}
                    
                    Provide: 1) 12-month ROI projection 2) Break-even analysis 
                    3) Risk scenarios (conservative/expected/optimistic) 4) Alternative investment comparison 5) KPI tracking"""

        response = self.chat(prompt)

        return {
            "basic_metrics": {
                "implementation_cost": implementation_cost,
                "monthly_savings": monthly_benefit,
                "annual_benefit": annual_benefit,
                "payback_period_months": payback_months,
                "payback_period_years": payback_months / 12,
                "first_year_net_benefit": first_year_net,
                "roi_percentage": roi_percentage
            },
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }