"""ROI Estimator Agent - Calculates return on investment for AI initiatives"""
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys
from functools import lru_cache
from datetime import datetime, timedelta
import math

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger


class ROIEstimatorAgent(BaseAgent):
    """Agent that estimates ROI for AI implementations with comprehensive financial modeling"""

    # Industry benchmark data for AI implementations
    INDUSTRY_BENCHMARKS = {
        "customer_service": {
            "typical_roi": {"low": 150, "medium": 250, "high": 400},
            "payback_months": {"fast": 6, "typical": 12, "slow": 18},
            "automation_potential": 0.7,
            "cost_reduction": 0.3
        },
        "content_generation": {
            "typical_roi": {"low": 200, "medium": 350, "high": 500},
            "payback_months": {"fast": 4, "typical": 8, "slow": 12},
            "automation_potential": 0.8,
            "cost_reduction": 0.4
        },
        "data_analysis": {
            "typical_roi": {"low": 180, "medium": 300, "high": 450},
            "payback_months": {"fast": 8, "typical": 15, "slow": 24},
            "automation_potential": 0.6,
            "cost_reduction": 0.25
        },
        "document_processing": {
            "typical_roi": {"low": 300, "medium": 500, "high": 800},
            "payback_months": {"fast": 3, "typical": 6, "slow": 12},
            "automation_potential": 0.9,
            "cost_reduction": 0.5
        },
        "code_generation": {
            "typical_roi": {"low": 250, "medium": 400, "high": 600},
            "payback_months": {"fast": 6, "typical": 12, "slow": 18},
            "automation_potential": 0.5,
            "cost_reduction": 0.35
        }
    }

    # Standard financial parameters
    FINANCIAL_DEFAULTS = {
        "discount_rate": 0.10,  # 10% cost of capital
        "risk_free_rate": 0.04,  # 4% risk-free rate
        "inflation_rate": 0.03,  # 3% inflation
        "corporate_tax_rate": 0.25,  # 25% corporate tax
        "analysis_years": 5
    }

    # Cost factors for different business impacts
    IMPACT_MULTIPLIERS = {
        "revenue_increase": 1.0,
        "cost_reduction": 0.8,  # More predictable
        "efficiency_gains": 0.7,  # Harder to measure
        "quality_improvements": 0.6,  # Indirect benefits
        "risk_reduction": 0.5,  # Hard to quantify
        "strategic_value": 0.3  # Very intangible
    }

    def __init__(self, agent_key: str = "roi"):
        super().__init__(
            name="ROI Calculator",
            description="Expert ROI analysis and financial modeling for AI implementations",
            agent_key=agent_key
        )

    def get_system_prompt(self) -> str:
        return """You are an expert AI ROI Analyst specializing in comprehensive financial modeling for AI implementations.

Your expertise includes:
- Advanced ROI calculations (NPV, IRR, payback period, risk-adjusted returns)
- Industry benchmark analysis and competitive positioning
- Sensitivity analysis and scenario modeling
- Business case development with conservative estimates
- Risk assessment and mitigation strategies
- Strategic value quantification

Always provide:
1. **Executive Summary**: Key ROI metrics and recommendation
2. **Financial Analysis**: Detailed calculations with assumptions
3. **Scenario Analysis**: Conservative, expected, and optimistic projections
4. **Risk Assessment**: Potential risks and mitigation strategies
5. **Industry Benchmarks**: Comparison to typical AI ROI outcomes
6. **Implementation Roadmap**: Timeline and milestones for value realization

Use conservative estimates, show all calculations, and provide actionable insights for decision-makers."""

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculate_roi",
                "description": "Comprehensive ROI analysis with NPV, IRR, and sensitivity analysis",
                "parameters": {
                    "project_name": "string",
                    "implementation_cost": "number",
                    "annual_benefits": "object",
                    "timeline_years": "number"
                }
            },
            {
                "name": "analyze_automation_roi",
                "description": "Calculate ROI for process automation initiatives",
                "parameters": {
                    "process_description": "string",
                    "current_cost": "number",
                    "automation_percentage": "number",
                    "implementation_cost": "number"
                }
            },
            {
                "name": "calculate_optimization_roi",
                "description": "ROI analysis for cost optimization and efficiency improvements",
                "parameters": {
                    "current_spend": "number",
                    "target_savings": "number",
                    "implementation_cost": "number"
                }
            }
        ]

    @lru_cache(maxsize=128)
    def _calculate_npv(self, implementation_cost: float, annual_benefit: float,
                       years: int, discount_rate: float = 0.10) -> float:
        """Calculate Net Present Value with caching"""
        if annual_benefit <= 0:
            return -implementation_cost

        npv = -implementation_cost
        for year in range(1, years + 1):
            npv += annual_benefit / ((1 + discount_rate) ** year)

        return npv

    def _calculate_irr(self, implementation_cost: float, annual_benefit: float, years: int) -> float:
        """Calculate Internal Rate of Return using iterative method"""
        if annual_benefit <= 0:
            return -1.0

        # Simple approximation for IRR
        if implementation_cost <= 0:
            return float('inf')

        # Newton's method approximation
        rate = 0.1  # Initial guess
        for _ in range(100):  # Max iterations
            npv = -implementation_cost + sum(annual_benefit / ((1 + rate) ** year) for year in range(1, years + 1))
            if abs(npv) < 0.01:
                break

            # Derivative of NPV with respect to rate
            dnpv = -sum(year * annual_benefit / ((1 + rate) ** (year + 1)) for year in range(1, years + 1))
            if abs(dnpv) < 1e-10:
                break

            rate = rate - npv / dnpv

            if rate < -0.99:  # Prevent negative rates below -99%
                rate = -0.99

        return rate

    def _get_industry_benchmark(self, use_case: str) -> Dict[str, Any]:
        """Get relevant industry benchmark data"""
        use_case_clean = use_case.lower().replace(" ", "_")

        # Find best match
        for benchmark_key, benchmark_data in self.INDUSTRY_BENCHMARKS.items():
            if benchmark_key in use_case_clean or any(word in use_case_clean for word in benchmark_key.split("_")):
                return {**benchmark_data, "category": benchmark_key}

        # Return average benchmark if no match
        return {
            "typical_roi": {"low": 200, "medium": 300, "high": 450},
            "payback_months": {"fast": 8, "typical": 12, "slow": 18},
            "automation_potential": 0.6,
            "cost_reduction": 0.3,
            "category": "general"
        }

    def _calculate_comprehensive_metrics(self, implementation_cost: float, annual_benefit: float,
                                       years: int) -> Dict[str, Any]:
        """Calculate comprehensive financial metrics"""
        if annual_benefit <= 0 or implementation_cost <= 0:
            return {
                "payback_period_years": float('inf'),
                "payback_period_months": float('inf'),
                "roi_percentage": -100,
                "net_benefit": -implementation_cost,
                "npv": -implementation_cost,
                "irr": -1.0,
                "profitability_index": 0
            }

        # Basic metrics
        payback_years = implementation_cost / annual_benefit
        payback_months = payback_years * 12
        total_benefit = annual_benefit * years
        net_benefit = total_benefit - implementation_cost
        roi_percentage = (net_benefit / implementation_cost) * 100

        # Advanced metrics
        npv = self._calculate_npv(implementation_cost, annual_benefit, years)
        irr = self._calculate_irr(implementation_cost, annual_benefit, years)
        profitability_index = (npv + implementation_cost) / implementation_cost if implementation_cost > 0 else 0

        return {
            "payback_period_years": round(payback_years, 1),
            "payback_period_months": round(payback_months, 1),
            "roi_percentage": round(roi_percentage, 1),
            "net_benefit": round(net_benefit, 2),
            "npv": round(npv, 2),
            "irr": round(irr * 100, 1),  # Convert to percentage
            "profitability_index": round(profitability_index, 2)
        }

    def _generate_scenario_analysis(self, base_benefit: float, implementation_cost: float,
                                  years: int) -> Dict[str, Any]:
        """Generate conservative, expected, and optimistic scenarios"""
        scenarios = {}

        scenario_multipliers = {
            "conservative": 0.7,
            "expected": 1.0,
            "optimistic": 1.4
        }

        for scenario, multiplier in scenario_multipliers.items():
            adjusted_benefit = base_benefit * multiplier
            scenarios[scenario] = self._calculate_comprehensive_metrics(
                implementation_cost, adjusted_benefit, years
            )
            scenarios[scenario]["annual_benefit"] = adjusted_benefit

        return scenarios

    def calculate_roi(self, project_name: str, implementation_cost: float,
                     annual_benefits: Dict[str, float], timeline_years: int = 5) -> Dict[str, Any]:
        """Enhanced ROI calculation with comprehensive financial analysis"""
        logger.info(f"Calculating comprehensive ROI for: {project_name}")

        # Calculate total annual benefit with impact weighting
        weighted_benefits = {}
        total_annual_benefit = 0

        for benefit_type, amount in annual_benefits.items():
            # Apply impact multiplier based on benefit type
            benefit_key = benefit_type.lower().replace(" ", "_")
            multiplier = 1.0

            for impact_type, impact_multiplier in self.IMPACT_MULTIPLIERS.items():
                if impact_type in benefit_key:
                    multiplier = impact_multiplier
                    break

            weighted_amount = amount * multiplier
            weighted_benefits[benefit_type] = {
                "original": amount,
                "weighted": weighted_amount,
                "confidence": multiplier
            }
            total_annual_benefit += weighted_amount

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(implementation_cost, total_annual_benefit, timeline_years)

        # Generate scenario analysis
        scenarios = self._generate_scenario_analysis(total_annual_benefit, implementation_cost, timeline_years)

        # Get industry benchmarks
        benchmark = self._get_industry_benchmark(project_name)

        # Format benefits for analysis
        benefits_summary = "\n".join([
            f"- {benefit}: ${details['original']:,.0f} (weighted: ${details['weighted']:,.0f}, confidence: {details['confidence']*100:.0f}%)"
            for benefit, details in weighted_benefits.items()
        ])

        prompt = f"""Comprehensive ROI Analysis for "{project_name}":

**FINANCIAL SUMMARY**:
- Implementation Cost: ${implementation_cost:,.0f}
- Total Annual Benefits: ${sum(b['original'] for b in weighted_benefits.values()):,.0f} (raw) / ${total_annual_benefit:,.0f} (risk-adjusted)
- Analysis Period: {timeline_years} years

**BENEFIT BREAKDOWN**:
{benefits_summary}

**KEY METRICS**:
- ROI: {metrics['roi_percentage']}%
- NPV: ${metrics['npv']:,.0f}
- IRR: {metrics['irr']}%
- Payback: {metrics['payback_period_months']:.1f} months
- Profitability Index: {metrics['profitability_index']}

**SCENARIO ANALYSIS**:
- Conservative: {scenarios['conservative']['roi_percentage']}% ROI, {scenarios['conservative']['payback_period_months']:.1f} month payback
- Expected: {scenarios['expected']['roi_percentage']}% ROI, {scenarios['expected']['payback_period_months']:.1f} month payback  
- Optimistic: {scenarios['optimistic']['roi_percentage']}% ROI, {scenarios['optimistic']['payback_period_months']:.1f} month payback

**INDUSTRY BENCHMARK** ({benchmark['category']}):
- Typical ROI Range: {benchmark['typical_roi']['low']}-{benchmark['typical_roi']['high']}%
- Typical Payback: {benchmark['payback_months']['typical']} months

Provide comprehensive analysis covering:
1. **Investment Recommendation**: Clear go/no-go recommendation with reasoning
2. **Financial Viability**: NPV, IRR, and payback analysis interpretation
3. **Risk Assessment**: Key risks and their potential impact on returns
4. **Benchmark Comparison**: How this project compares to industry standards
5. **Value Realization Timeline**: When benefits will be realized
6. **Optimization Opportunities**: Ways to improve ROI
7. **Success Metrics**: KPIs to track ROI achievement"""

        response = self.chat(prompt)

        return {
            "project_name": project_name,
            "financial_summary": {
                "implementation_cost": implementation_cost,
                "total_annual_benefit_raw": sum(b['original'] for b in weighted_benefits.values()),
                "total_annual_benefit_adjusted": total_annual_benefit,
                "timeline_years": timeline_years
            },
            "weighted_benefits": weighted_benefits,
            "key_metrics": metrics,
            "scenario_analysis": scenarios,
            "industry_benchmark": benchmark,
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def analyze_automation_roi(self, process_description: str, current_annual_cost: float,
                             automation_percentage: float, implementation_cost: float,
                             timeline_years: int = 3) -> Dict[str, Any]:
        """Calculate ROI for process automation with detailed analysis"""
        logger.info(f"Analyzing automation ROI for: {process_description}")

        # Calculate automation benefits
        annual_savings = current_annual_cost * (automation_percentage / 100)
        remaining_cost = current_annual_cost - annual_savings

        # Additional benefits (quality, speed, etc.)
        quality_improvement_benefit = annual_savings * 0.15  # 15% additional value from quality
        efficiency_benefit = annual_savings * 0.10  # 10% from efficiency gains

        total_annual_benefit = annual_savings + quality_improvement_benefit + efficiency_benefit

        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(implementation_cost, total_annual_benefit, timeline_years)

        # Industry benchmark for automation
        benchmark = self._get_industry_benchmark(process_description)
        expected_automation = benchmark.get('automation_potential', 0.6) * 100

        prompt = f"""Process Automation ROI Analysis:

**PROCESS DETAILS**:
- Description: {process_description}
- Current Annual Cost: ${current_annual_cost:,.0f}
- Target Automation: {automation_percentage}% (Industry typical: {expected_automation:.0f}%)
- Implementation Cost: ${implementation_cost:,.0f}

**BENEFIT ANALYSIS**:
- Direct Labor Savings: ${annual_savings:,.0f}/year ({automation_percentage}% of ${current_annual_cost:,.0f})
- Quality Improvement Value: ${quality_improvement_benefit:,.0f}/year (15% bonus)
- Efficiency Gains: ${efficiency_benefit:,.0f}/year (10% bonus)
- **Total Annual Benefit**: ${total_annual_benefit:,.0f}
- Remaining Manual Cost: ${remaining_cost:,.0f}/year

**FINANCIAL METRICS**:
- ROI: {metrics['roi_percentage']}%
- Payback: {metrics['payback_period_months']:.1f} months
- NPV: ${metrics['npv']:,.0f}
- IRR: {metrics['irr']}%

Provide detailed analysis covering:
1. **Automation Feasibility**: Technical and organizational readiness
2. **Process Impact Assessment**: How automation will change operations
3. **Risk Factors**: Implementation risks and mitigation strategies
4. **Phased Implementation**: Suggested rollout approach
5. **Change Management**: Training and adoption considerations
6. **Success Metrics**: KPIs to measure automation success
7. **Alternative Approaches**: Other automation options to consider"""

        response = self.chat(prompt)

        return {
            "process_analysis": {
                "description": process_description,
                "current_annual_cost": current_annual_cost,
                "automation_percentage": automation_percentage,
                "implementation_cost": implementation_cost
            },
            "benefit_breakdown": {
                "direct_savings": annual_savings,
                "quality_improvements": quality_improvement_benefit,
                "efficiency_gains": efficiency_benefit,
                "total_annual_benefit": total_annual_benefit
            },
            "financial_metrics": metrics,
            "industry_comparison": {
                "target_automation": automation_percentage,
                "typical_automation": expected_automation,
                "benchmark_category": benchmark.get('category', 'general')
            },
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def calculate_optimization_roi(self, current_spend: float, target_savings: float,
                                 implementation_cost: float, timeline_years: int = 2) -> Dict[str, Any]:
        """Enhanced ROI analysis for cost optimization initiatives"""
        logger.info(f"Calculating optimization ROI: ${target_savings:,.0f} monthly savings")

        # Convert monthly to annual
        annual_savings = target_savings * 12
        annual_spend = current_spend * 12
        savings_percentage = (target_savings / current_spend) * 100

        # Calculate risk-adjusted savings (optimization projects often achieve 70-80% of targets)
        risk_adjustment = 0.75
        conservative_annual_savings = annual_savings * risk_adjustment

        # Calculate metrics for both scenarios
        optimistic_metrics = self._calculate_comprehensive_metrics(implementation_cost, annual_savings, timeline_years)
        conservative_metrics = self._calculate_comprehensive_metrics(implementation_cost, conservative_annual_savings, timeline_years)

        prompt = f"""Cost Optimization ROI Analysis:

**OPTIMIZATION SCENARIO**:
- Current Monthly Spend: ${current_spend:,.0f}
- Target Monthly Savings: ${target_savings:,.0f} ({savings_percentage:.1f}% reduction)
- Implementation Cost: ${implementation_cost:,.0f}
- Analysis Period: {timeline_years} years

**FINANCIAL PROJECTIONS**:

*Optimistic Scenario* (100% target achievement):
- Annual Savings: ${annual_savings:,.0f}
- ROI: {optimistic_metrics['roi_percentage']}%
- Payback: {optimistic_metrics['payback_period_months']:.1f} months
- NPV: ${optimistic_metrics['npv']:,.0f}

*Conservative Scenario* (75% target achievement):
- Annual Savings: ${conservative_annual_savings:,.0f}
- ROI: {conservative_metrics['roi_percentage']}%
- Payback: {conservative_metrics['payback_period_months']:.1f} months
- NPV: ${conservative_metrics['npv']:,.0f}

**VALUE DRIVERS**:
- Immediate cost reduction
- Process efficiency improvements
- Technology consolidation benefits
- Vendor negotiation leverage

Provide comprehensive analysis including:
1. **Optimization Strategy Assessment**: Feasibility of achieving target savings
2. **Implementation Roadmap**: Phase-by-phase cost reduction plan
3. **Risk Analysis**: Factors that could impact savings achievement
4. **Monitoring Plan**: KPIs and tracking mechanisms
5. **Sustainability**: Long-term maintenance of cost reductions
6. **Alternative Scenarios**: Different savings targets and their ROI
7. **Quick Wins**: Immediate optimization opportunities"""

        response = self.chat(prompt)

        return {
            "optimization_details": {
                "current_monthly_spend": current_spend,
                "target_monthly_savings": target_savings,
                "savings_percentage": savings_percentage,
                "implementation_cost": implementation_cost,
                "timeline_years": timeline_years
            },
            "scenario_comparison": {
                "optimistic": {
                    "annual_savings": annual_savings,
                    "achievement_rate": 100,
                    **optimistic_metrics
                },
                "conservative": {
                    "annual_savings": conservative_annual_savings,
                    "achievement_rate": 75,
                    **conservative_metrics
                }
            },
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def analyze_business_impact(self, use_case: str, current_metrics: Dict[str, Any],
                              expected_improvements: Dict[str, Any],
                              revenue_context: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Enhanced business impact analysis with quantified benefits"""
        logger.info(f"Analyzing business impact for: {use_case}")

        # Format current state and improvements
        current_str = "\n".join(f"- {k}: {v}" for k, v in current_metrics.items())
        improvements_str = "\n".join(f"- {k}: {v}" for k, v in expected_improvements.items())

        # Add revenue context if provided
        revenue_context_str = ""
        if revenue_context:
            revenue_context_str = f"\n**REVENUE CONTEXT**:\n" + "\n".join(f"- {k}: ${v:,.0f}" for k, v in revenue_context.items())

        # Get industry benchmark
        benchmark = self._get_industry_benchmark(use_case)

        prompt = f"""Comprehensive Business Impact Analysis for AI Implementation:

**USE CASE**: {use_case}

**CURRENT STATE**:
{current_str}

**EXPECTED IMPROVEMENTS**:
{improvements_str}
{revenue_context_str}

**INDUSTRY BENCHMARK** ({benchmark['category']}):
- Typical ROI: {benchmark['typical_roi']['medium']}%
- Automation Potential: {benchmark['automation_potential']*100:.0f}%
- Cost Reduction Potential: {benchmark['cost_reduction']*100:.0f}%

Provide quantified business impact analysis covering:

1. **REVENUE IMPACT** (quantify where possible):
   - Direct revenue increases from improvements
   - Revenue protection from risk mitigation
   - New revenue opportunities enabled

2. **COST IMPACT** (with specific calculations):
   - Direct cost reductions from automation/efficiency
   - Indirect cost savings from quality improvements
   - Avoided costs from risk mitigation

3. **OPERATIONAL EXCELLENCE**:
   - Process efficiency gains (time, quality, consistency)
   - Resource optimization opportunities
   - Scalability improvements

4. **STRATEGIC VALUE**:
   - Competitive advantages gained
   - Market positioning improvements
   - Innovation enablement

5. **RISK MITIGATION**:
   - Compliance risk reduction
   - Operational risk mitigation
   - Financial risk management

6. **CUSTOMER IMPACT**:
   - Customer satisfaction improvements
   - Customer retention benefits
   - Customer acquisition advantages

7. **QUANTIFIED BENEFITS SUMMARY**:
   - Annual financial impact estimate
   - Percentage improvement in key metrics
   - Long-term strategic value

Include specific examples and industry comparisons where relevant."""

        response = self.chat(prompt)

        return {
            "use_case": use_case,
            "current_state": current_metrics,
            "expected_improvements": expected_improvements,
            "revenue_context": revenue_context,
            "industry_benchmark": benchmark,
            "detailed_impact_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }