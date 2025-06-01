"""Cost Calculator Agent - Calculates and compares AI implementation costs"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import json
from functools import lru_cache
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger

class CostCalculatorAgent(BaseAgent):
    """Agent that calculates AI implementation and operational costs"""

    # Updated pricing data with latest rates (per 1K tokens)
    LLM_PRICING = {
        # OpenAI Models
        "gpt-4o": {"input": 0.005, "output": 0.015, "provider": "OpenAI", "tier": "premium"},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "provider": "OpenAI", "tier": "efficient"},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03, "provider": "OpenAI", "tier": "premium"},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015, "provider": "OpenAI", "tier": "budget"},

        # Anthropic Models
        "claude-3-opus": {"input": 0.015, "output": 0.075, "provider": "Anthropic", "tier": "premium"},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015, "provider": "Anthropic", "tier": "balanced"},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125, "provider": "Anthropic", "tier": "budget"},

        # Google Models
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005, "provider": "Google", "tier": "balanced"},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003, "provider": "Google", "tier": "budget"},
        "gemini-pro": {"input": 0.0005, "output": 0.0015, "provider": "Google", "tier": "budget"},

        # Open Source (estimated hosting costs)
        "llama-3-70b": {"input": 0.0009, "output": 0.0009, "provider": "Meta", "tier": "efficient"},
        "llama-3-8b": {"input": 0.0002, "output": 0.0002, "provider": "Meta", "tier": "budget"},
        "mistral-large": {"input": 0.008, "output": 0.024, "provider": "Mistral", "tier": "premium"},
        "mistral-7b": {"input": 0.0002, "output": 0.0002, "provider": "Mistral", "tier": "budget"}
    }

    # Infrastructure cost estimates (monthly)
    INFRASTRUCTURE_COSTS = {
        "small": {"compute": 500, "storage": 100, "networking": 50, "monitoring": 100},
        "medium": {"compute": 2000, "storage": 300, "networking": 200, "monitoring": 300},
        "large": {"compute": 8000, "storage": 1000, "networking": 500, "monitoring": 800},
        "enterprise": {"compute": 20000, "storage": 3000, "networking": 1500, "monitoring": 2000}
    }

    # Use case specific token estimates
    USE_CASE_PROFILES = {
        "customer_support": {"input_tokens": 150, "output_tokens": 200, "complexity": "medium"},
        "content_generation": {"input_tokens": 100, "output_tokens": 500, "complexity": "high"},
        "data_analysis": {"input_tokens": 400, "output_tokens": 300, "complexity": "high"},
        "code_generation": {"input_tokens": 200, "output_tokens": 600, "complexity": "high"},
        "document_processing": {"input_tokens": 800, "output_tokens": 200, "complexity": "medium"},
        "chatbot": {"input_tokens": 100, "output_tokens": 150, "complexity": "low"},
        "summarization": {"input_tokens": 1000, "output_tokens": 200, "complexity": "medium"},
        "translation": {"input_tokens": 200, "output_tokens": 200, "complexity": "low"}
    }

    def __init__(self, agent_key: str = "cost"):
        super().__init__(
            name="Cost Calculator",
            description="Expert AI cost analysis and optimization",
            agent_key=agent_key
        )

    def get_system_prompt(self) -> str:
        return """You are an expert AI Cost Calculator specializing in enterprise AI implementations.

Your expertise includes:
- Precise LLM API cost calculations with current pricing
- Infrastructure cost estimation and optimization
- Hidden cost identification (integration, maintenance, scaling)
- ROI analysis and cost-benefit modeling
- Migration cost assessment
- Volume-based pricing tier optimization

When analyzing costs, always:
1. Use specific numbers with clear calculations
2. Consider scaling factors and volume discounts
3. Include implementation and operational costs
4. Identify cost optimization opportunities
5. Provide actionable recommendations with timelines
6. Show break-even analysis and ROI projections

Response format:
- Start with executive summary (cost range, key findings)
- Provide detailed breakdown with assumptions
- Include comparison tables when relevant
- End with specific next steps and recommendations"""

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculate_llm_costs",
                "description": "Calculate detailed LLM API costs with optimization recommendations",
                "parameters": {
                    "use_case": "string",
                    "requests_per_day": "number",
                    "models": "array"
                }
            },
            {
                "name": "analyze_infrastructure_costs",
                "description": "Analyze current infrastructure spending with specific optimization strategies",
                "parameters": {
                    "current_spend": "number",
                    "providers": "array",
                    "target_reduction": "number"
                }
            }
        ]

    @lru_cache(maxsize=256)
    def _calculate_model_cost(self, model: str, requests_per_day: int,
                             avg_input_tokens: int, avg_output_tokens: int) -> Dict[str, Any]:
        """Enhanced model cost calculation with tier classification"""
        if model not in self.LLM_PRICING:
            return {"error": f"Model {model} not found in pricing database"}

        pricing = self.LLM_PRICING[model]

        # Calculate base costs
        daily_input_cost = (avg_input_tokens * pricing["input"] * requests_per_day) / 1000
        daily_output_cost = (avg_output_tokens * pricing["output"] * requests_per_day) / 1000
        daily_total = daily_input_cost + daily_output_cost

        # Apply volume discounts for high usage
        monthly_requests = requests_per_day * 30
        discount_factor = self._get_volume_discount(monthly_requests)

        adjusted_daily_cost = daily_total * discount_factor

        return {
            "provider": pricing["provider"],
            "tier": pricing["tier"],
            "daily_cost": round(adjusted_daily_cost, 2),
            "monthly_cost": round(adjusted_daily_cost * 30, 2),
            "annual_cost": round(adjusted_daily_cost * 365, 2),
            "cost_per_request": round(adjusted_daily_cost / requests_per_day, 4),
            "input_cost_daily": round(daily_input_cost, 2),
            "output_cost_daily": round(daily_output_cost, 2),
            "volume_discount": round((1 - discount_factor) * 100, 1),
            "tokens_per_day": (avg_input_tokens + avg_output_tokens) * requests_per_day
        }

    def _get_volume_discount(self, monthly_requests: int) -> float:
        """Calculate volume discount factor"""
        if monthly_requests > 10000000:  # 10M+ requests
            return 0.85  # 15% discount
        elif monthly_requests > 1000000:  # 1M+ requests
            return 0.90  # 10% discount
        elif monthly_requests > 100000:   # 100K+ requests
            return 0.95  # 5% discount
        return 1.0  # No discount

    def _get_use_case_defaults(self, use_case: str) -> Dict[str, int]:
        """Get default token estimates for use case"""
        use_case_clean = use_case.lower().replace(" ", "_")

        # Find best match
        for profile_key, profile in self.USE_CASE_PROFILES.items():
            if profile_key in use_case_clean or use_case_clean in profile_key:
                return profile

        # Default fallback
        return {"input_tokens": 200, "output_tokens": 250, "complexity": "medium"}

    def calculate_llm_costs(self, use_case: str, requests_per_day: int,
                           avg_input_tokens: Optional[int] = None,
                           avg_output_tokens: Optional[int] = None,
                           models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced LLM cost calculation with smart defaults and optimization"""
        logger.info(f"Calculating LLM costs for: {use_case} ({requests_per_day} req/day)")

        # Use smart defaults if not provided
        if avg_input_tokens is None or avg_output_tokens is None:
            defaults = self._get_use_case_defaults(use_case)
            avg_input_tokens = avg_input_tokens or defaults["input_tokens"]
            avg_output_tokens = avg_output_tokens or defaults["output_tokens"]

        # Smart model selection if not provided
        if not models:
            models = self._recommend_models_for_use_case(use_case, requests_per_day)

        # Calculate costs for all models
        cost_analysis = {}
        for model in models:
            if model in self.LLM_PRICING:
                cost_analysis[model] = self._calculate_model_cost(
                    model, requests_per_day, avg_input_tokens, avg_output_tokens
                )

        # Generate analysis prompt with structured data
        monthly_requests = requests_per_day * 30
        annual_requests = requests_per_day * 365

        cost_summary = self._generate_cost_summary(cost_analysis)

        prompt = f"""Analyze this LLM cost scenario:

**USE CASE**: {use_case}
**VOLUME**: {requests_per_day:,} requests/day ({monthly_requests:,}/month, {annual_requests:,}/year)
**TOKENS**: {avg_input_tokens} input + {avg_output_tokens} output tokens per request

**COST BREAKDOWN**:
{json.dumps(cost_analysis, indent=2)}

**COST SUMMARY**:
{cost_summary}

Provide:
1. **RECOMMENDED SOLUTION**: Best cost-performance model with reasoning
2. **COST OPTIMIZATION**: 3 specific strategies to reduce costs
3. **SCALING CONSIDERATIONS**: How costs change with 2x, 5x, 10x growth
4. **HIDDEN COSTS**: Additional costs to budget for (15-25% of API costs)
5. **BUDGET PLANNING**: Monthly budget ranges for different scenarios"""

        response = self.chat(prompt)

        return {
            "use_case": use_case,
            "volume_metrics": {
                "daily_requests": requests_per_day,
                "monthly_requests": monthly_requests,
                "annual_requests": annual_requests,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens
            },
            "cost_breakdown": cost_analysis,
            "cost_summary": cost_summary,
            "analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def _recommend_models_for_use_case(self, use_case: str, requests_per_day: int) -> List[str]:
        """Recommend appropriate models based on use case and volume"""
        use_case_lower = use_case.lower()

        # High-volume scenarios (>10K requests/day) - focus on cost efficiency
        if requests_per_day > 10000:
            if any(term in use_case_lower for term in ["support", "chat", "simple"]):
                return ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash", "gpt-3.5-turbo"]
            else:
                return ["gpt-4o-mini", "claude-3-sonnet", "gemini-1.5-pro", "claude-3-haiku"]

        # Medium-volume scenarios (1K-10K requests/day) - balanced approach
        elif requests_per_day > 1000:
            if any(term in use_case_lower for term in ["content", "creative", "complex"]):
                return ["gpt-4o", "claude-3-sonnet", "gemini-1.5-pro", "gpt-4-turbo"]
            else:
                return ["gpt-4o-mini", "claude-3-sonnet", "gemini-1.5-pro", "claude-3-haiku"]

        # Low-volume scenarios (<1K requests/day) - quality focused
        else:
            return ["gpt-4o", "claude-3-opus", "gpt-4-turbo", "claude-3-sonnet"]

    def _generate_cost_summary(self, cost_analysis: Dict) -> str:
        """Generate a formatted cost summary table"""
        if not cost_analysis:
            return "No cost data available"

        sorted_models = sorted(
            cost_analysis.items(),
            key=lambda x: x[1].get("monthly_cost", float('inf'))
        )

        summary = "MONTHLY COST COMPARISON:\n"
        for model, data in sorted_models:
            if "error" not in data:
                provider = data.get("provider", "Unknown")
                tier = data.get("tier", "")
                monthly_cost = data.get("monthly_cost", 0)
                cost_per_request = data.get("cost_per_request", 0)
                summary += f"• {model} ({provider} - {tier}): ${monthly_cost:,.0f}/month (${cost_per_request:.4f}/request)\n"

        return summary

    def analyze_infrastructure_costs(self, current_spend: float, providers: List[str],
                                   target_reduction: float = 0.3) -> Dict[str, Any]:
        """Enhanced infrastructure cost analysis with specific recommendations"""
        logger.info(f"Analyzing infrastructure costs: ${current_spend:,.0f}/month")

        # Determine infrastructure scale
        scale = self._determine_infrastructure_scale(current_spend)
        infrastructure_breakdown = self.INFRASTRUCTURE_COSTS[scale]

        potential_savings = current_spend * target_reduction
        target_spend = current_spend - potential_savings

        prompt = f"""Analyze this AI infrastructure cost optimization scenario:

**CURRENT SITUATION**:
- Monthly Spend: ${current_spend:,.0f}
- Annual Spend: ${current_spend * 12:,.0f}
- Scale Category: {scale.title()}
- Providers: {', '.join(providers)}
- Target Reduction: {target_reduction * 100:.0f}% (${potential_savings:,.0f}/month)

**ESTIMATED COST BREAKDOWN** (based on {scale} scale):
- Compute/API Costs: ${infrastructure_breakdown['compute']:,.0f}/month ({infrastructure_breakdown['compute']/current_spend*100:.0f}%)
- Storage: ${infrastructure_breakdown['storage']:,.0f}/month ({infrastructure_breakdown['storage']/current_spend*100:.0f}%)
- Networking: ${infrastructure_breakdown['networking']:,.0f}/month ({infrastructure_breakdown['networking']/current_spend*100:.0f}%)
- Monitoring/Tools: ${infrastructure_breakdown['monitoring']:,.0f}/month ({infrastructure_breakdown['monitoring']/current_spend*100:.0f}%)

**OPTIMIZATION TARGET**: Reduce to ${target_spend:,.0f}/month

Provide a DETAILED ACTION PLAN:

1. **IMMEDIATE WINS (Week 1-2)** - Save ${potential_savings * 0.3:,.0f}/month:
   - Quick optimization actions requiring minimal risk
   - Specific provider settings or configurations to change
   
2. **SHORT-TERM OPTIMIZATIONS (Month 1-2)** - Save ${potential_savings * 0.4:,.0f}/month:
   - Medium-complexity changes requiring testing
   - Provider migrations or tier adjustments
   
3. **STRATEGIC CHANGES (Month 2-6)** - Save ${potential_savings * 0.3:,.0f}/month:
   - Architecture improvements and long-term optimizations
   - Contract renegotiations and volume discounts

4. **PROVIDER-SPECIFIC RECOMMENDATIONS**:
   - Specific advice for each mentioned provider
   - Alternative providers to consider

5. **RISK ASSESSMENT & MITIGATION**:
   - Potential impacts on performance/availability
   - Testing and rollback strategies

6. **12-MONTH FINANCIAL PROJECTION**:
   - Month-by-month savings progression
   - Total cumulative savings
   - ROI on optimization efforts"""

        response = self.chat(prompt)

        return {
            "current_analysis": {
                "monthly_spend": current_spend,
                "annual_spend": current_spend * 12,
                "scale_category": scale,
                "cost_breakdown": infrastructure_breakdown
            },
            "optimization_targets": {
                "target_spend": target_spend,
                "potential_savings": potential_savings,
                "annual_savings": potential_savings * 12,
                "reduction_percentage": target_reduction * 100
            },
            "providers": providers,
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def _determine_infrastructure_scale(self, monthly_spend: float) -> str:
        """Determine infrastructure scale based on spending"""
        if monthly_spend < 1000:
            return "small"
        elif monthly_spend < 5000:
            return "medium"
        elif monthly_spend < 25000:
            return "large"
        else:
            return "enterprise"

    def estimate_total_implementation_cost(self, project_description: str,
                                         team_size: int, timeline_weeks: int,
                                         complexity: str = "medium") -> Dict[str, Any]:
        """Enhanced implementation cost estimation with detailed breakdown"""
        logger.info(f"Estimating implementation costs: {team_size} person team, {timeline_weeks} weeks")

        # Cost factors based on complexity
        complexity_multipliers = {
            "simple": 0.8,
            "medium": 1.0,
            "complex": 1.4,
            "enterprise": 2.0
        }

        base_hourly_rate = 150
        multiplier = complexity_multipliers.get(complexity.lower(), 1.0)
        effective_rate = base_hourly_rate * multiplier

        # Calculate base development cost
        total_hours = team_size * timeline_weeks * 40  # 40 hours/week
        development_cost = total_hours * effective_rate

        prompt = f"""Calculate comprehensive implementation costs for this AI project:

**PROJECT DETAILS**:
- Description: {project_description}
- Team Size: {team_size} developers
- Timeline: {timeline_weeks} weeks ({total_hours:,} total hours)
- Complexity: {complexity}
- Effective Rate: ${effective_rate}/hour

**BASE DEVELOPMENT COST**: ${development_cost:,.0f}

Provide detailed breakdown for:

1. **DEVELOPMENT COSTS** (${development_cost:,.0f} base):
   - Frontend development (% of total)
   - Backend/API development (% of total)
   - AI/ML integration (% of total)
   - Testing and QA (% of total)
   - Project management (% of total)

2. **INFRASTRUCTURE SETUP** (estimate):
   - Cloud environment setup
   - CI/CD pipeline configuration
   - Monitoring and logging setup
   - Security implementation

3. **THIRD-PARTY COSTS** (first year):
   - LLM API costs (development + initial production)
   - Cloud infrastructure (development + production)
   - Tools and licenses
   - External services integration

4. **OPERATIONAL COSTS** (monthly ongoing):
   - Production infrastructure
   - Monitoring and maintenance
   - Support and updates
   - Scaling considerations

5. **TOTAL INVESTMENT SUMMARY**:
   - One-time implementation cost
   - First-year total cost
   - Ongoing monthly operational cost"""

        response = self.chat(prompt)

        return {
            "project_details": {
                "description": project_description,
                "team_size": team_size,
                "timeline_weeks": timeline_weeks,
                "complexity": complexity,
                "total_hours": total_hours
            },
            "base_calculations": {
                "hourly_rate": effective_rate,
                "development_cost": development_cost,
                "complexity_multiplier": multiplier
            },
            "detailed_breakdown": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def calculate_migration_costs(self, from_providers: List[str], to_providers: List[str],
                                monthly_volume: int, current_monthly_cost: Optional[float] = None) -> Dict[str, Any]:
        """Enhanced migration cost calculation with timeline and risk assessment"""

        prompt = f"""Calculate detailed migration costs and timeline:

**MIGRATION SCENARIO**:
- FROM: {', '.join(from_providers)}
- TO: {', '.join(to_providers)}
- Monthly Volume: {monthly_volume:,} requests
- Current Monthly Cost: ${current_monthly_cost or 'Unknown'}

Provide comprehensive analysis:

1. **MIGRATION COSTS** (one-time):
   - Engineering effort (development hours × rate)
   - Testing and validation costs
   - Deployment and rollout costs
   - Training and documentation
   - Contingency buffer (15-20%)

2. **ONGOING COST COMPARISON**:
   - Before migration: Monthly cost breakdown
   - After migration: Projected monthly cost
   - Net monthly savings/increase
   - Break-even timeline

3. **MIGRATION TIMELINE** (detailed):
   - Week 1-2: Planning and setup
   - Week 3-6: Development and integration
   - Week 7-8: Testing and validation
   - Week 9-10: Deployment and monitoring
   - Total timeline: X weeks

4. **RISK ASSESSMENT**:
   - Technical risks and mitigation strategies
   - Performance/quality risks
   - Cost overrun risks
   - Timeline risks

5. **ROI ANALYSIS**:
   - Total migration investment
   - Monthly savings after migration
   - Payback period
   - 12-month and 24-month ROI"""

        response = self.chat(prompt)

        return {
            "migration_details": {
                "from_providers": from_providers,
                "to_providers": to_providers,
                "monthly_volume": monthly_volume,
                "current_monthly_cost": current_monthly_cost
            },
            "analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }