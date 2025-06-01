"""Task Analyzer Agent - Identifies AI automation opportunities"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
from functools import lru_cache
import re

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger

class TaskAnalyzerAgent(BaseAgent):
    """Agent that identifies and analyzes AI automation opportunities with detailed implementation guidance"""

    # Task categorization patterns for automated classification
    TASK_PATTERNS = {
        "data_processing": {
            "keywords": ["data entry", "spreadsheet", "extract", "parse", "import", "export", "csv", "excel", "database"],
            "ai_solutions": ["OCR", "Document AI", "NLP parsing", "Automated data extraction"],
            "complexity": "Low",
            "automation_potential": 0.9
        },
        "document_analysis": {
            "keywords": ["review", "analyze", "summarize", "extract information", "pdf", "contract", "invoice", "report"],
            "ai_solutions": ["Document AI", "LLM summarization", "Entity extraction", "Classification"],
            "complexity": "Medium",
            "automation_potential": 0.8
        },
        "customer_service": {
            "keywords": ["customer", "support", "ticket", "inquiry", "chat", "email response", "help desk"],
            "ai_solutions": ["Chatbot", "Automated responses", "Sentiment analysis", "Ticket routing"],
            "complexity": "Medium",
            "automation_potential": 0.7
        },
        "content_generation": {
            "keywords": ["write", "create", "generate", "draft", "content", "blog", "marketing", "copywriting"],
            "ai_solutions": ["LLM content generation", "Template automation", "Personalization"],
            "complexity": "Low",
            "automation_potential": 0.8
        },
        "scheduling_coordination": {
            "keywords": ["schedule", "calendar", "meeting", "appointment", "coordinate", "booking"],
            "ai_solutions": ["Calendar AI", "Automated scheduling", "Meeting optimization"],
            "complexity": "Medium",
            "automation_potential": 0.6
        },
        "quality_assurance": {
            "keywords": ["review", "check", "validate", "approve", "audit", "compliance", "quality"],
            "ai_solutions": ["Automated QA", "Compliance checking", "Anomaly detection"],
            "complexity": "High",
            "automation_potential": 0.6
        },
        "research_analysis": {
            "keywords": ["research", "analyze", "investigate", "compare", "benchmark", "market research"],
            "ai_solutions": ["Web scraping + AI", "Competitive analysis", "Trend analysis"],
            "complexity": "High",
            "automation_potential": 0.5
        },
        "reporting": {
            "keywords": ["report", "dashboard", "metrics", "kpi", "analytics", "status update"],
            "ai_solutions": ["Automated reporting", "Data visualization", "Insight generation"],
            "complexity": "Medium",
            "automation_potential": 0.7
        }
    }

    # Complexity assessment framework
    COMPLEXITY_FACTORS = {
        "data_availability": {"high": 0.2, "medium": 0.5, "low": 0.8},
        "process_standardization": {"high": 0.2, "medium": 0.5, "low": 0.9},
        "human_judgment_required": {"low": 0.2, "medium": 0.6, "high": 1.0},
        "integration_complexity": {"simple": 0.3, "moderate": 0.6, "complex": 1.0},
        "stakeholder_count": {"few": 0.2, "moderate": 0.5, "many": 0.8}
    }

    # ROI calculation parameters
    ROI_PARAMETERS = {
        "average_hourly_rate": 50,  # Default knowledge worker rate
        "implementation_multipliers": {
            "low": 2,      # 2x weekly hours for implementation
            "medium": 4,   # 4x weekly hours for implementation
            "high": 8      # 8x weekly hours for implementation
        },
        "automation_efficiency": {
            "low": 0.6,    # 60% of manual time saved
            "medium": 0.75, # 75% of manual time saved
            "high": 0.9    # 90% of manual time saved
        }
    }

    def __init__(self, agent_key: str = "task"):
        super().__init__(
            name="Task Analyzer",
            description="Expert AI automation opportunity identification and implementation planning",
            agent_key=agent_key
        )

    def get_system_prompt(self) -> str:
        return """You are an expert AI Task Analyzer specializing in identifying and prioritizing enterprise automation opportunities.

Your expertise includes:
- Process decomposition and automation opportunity identification
- AI solution mapping (LLMs, ML models, RPA, document AI)
- Implementation complexity assessment and timeline estimation
- ROI calculation for automation initiatives
- Change management and adoption strategy development

When analyzing tasks and workflows:
1. **Identify specific automatable components** within larger processes
2. **Map appropriate AI solutions** to each component (be technology-specific)
3. **Assess implementation complexity** considering data, integration, and change management
4. **Calculate time savings and ROI** with realistic estimates
5. **Prioritize by impact vs. effort** using a structured framework
6. **Provide implementation roadmaps** with phases and milestones

Always provide:
- **Executive Summary**: Top automation opportunities with impact estimates
- **Detailed Task Analysis**: Break down of each automation opportunity
- **Implementation Roadmap**: Phased approach with timelines
- **Resource Requirements**: Team, tools, and budget needed
- **Risk Assessment**: Technical and business risks with mitigation
- **Success Metrics**: KPIs to measure automation success

Be specific about AI technologies, realistic about timelines, and focus on high-impact opportunities."""

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "analyze_workflow",
                "description": "Comprehensive workflow analysis for AI automation opportunities",
                "parameters": {
                    "workflow_description": "string"
                }
            },
            {
                "name": "assess_task_complexity",
                "description": "Detailed complexity assessment for specific automation tasks",
                "parameters": {
                    "task_description": "string",
                    "current_process": "string",
                    "constraints": "string"
                }
            },
            {
                "name": "prioritize_automation_opportunities",
                "description": "Prioritize multiple automation opportunities by ROI and feasibility",
                "parameters": {
                    "opportunities": "array"
                }
            }
        ]

    @lru_cache(maxsize=128)
    def _classify_task_type(self, task_description: str) -> Tuple[str, Dict[str, Any]]:
        """Classify task type based on description keywords"""
        task_lower = task_description.lower()

        # Score each task category
        scores = {}
        for category, pattern_data in self.TASK_PATTERNS.items():
            score = sum(1 for keyword in pattern_data["keywords"] if keyword in task_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return "general", {
                "ai_solutions": ["Custom AI solution"],
                "complexity": "Medium",
                "automation_potential": 0.5
            }

        # Return highest scoring category
        best_category = max(scores.items(), key=lambda x: x[1])[0]
        return best_category, self.TASK_PATTERNS[best_category]

    def _extract_time_metrics(self, description: str) -> Dict[str, float]:
        """Extract time-related metrics from description"""
        metrics = {}

        # Time patterns
        time_patterns = {
            'hours_per_day': [r'(\d+(?:\.\d+)?)\s*hours?\s*(?:per\s*day|daily)', r'(\d+(?:\.\d+)?)\s*hrs?\s*/\s*day'],
            'hours_per_week': [r'(\d+(?:\.\d+)?)\s*hours?\s*(?:per\s*week|weekly)', r'(\d+(?:\.\d+)?)\s*hrs?\s*/\s*week'],
            'minutes_per_task': [r'(\d+)\s*minutes?\s*(?:per\s*task|each)', r'(\d+)\s*mins?\s*/\s*task'],
            'tasks_per_day': [r'(\d+)\s*tasks?\s*(?:per\s*day|daily)', r'(\d+)\s*(?:requests?|tickets?)\s*(?:per\s*day|daily)']
        }

        description_lower = description.lower()

        for metric, patterns in time_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, description_lower)
                if matches:
                    metrics[metric] = float(matches[0])
                    break

        # Calculate derived metrics
        if 'hours_per_day' in metrics and 'hours_per_week' not in metrics:
            metrics['hours_per_week'] = metrics['hours_per_day'] * 5
        elif 'hours_per_week' in metrics and 'hours_per_day' not in metrics:
            metrics['hours_per_day'] = metrics['hours_per_week'] / 5

        if 'minutes_per_task' in metrics and 'tasks_per_day' in metrics:
            daily_hours = (metrics['minutes_per_task'] * metrics['tasks_per_day']) / 60
            metrics['hours_per_day'] = daily_hours
            metrics['hours_per_week'] = daily_hours * 5

        return metrics

    def _calculate_automation_score(self, task_category: str, time_metrics: Dict[str, float],
                                   complexity_factors: Dict[str, str]) -> Dict[str, Any]:
        """Calculate automation potential score and ROI estimate"""
        pattern_data = self.TASK_PATTERNS.get(task_category, {})
        base_potential = pattern_data.get("automation_potential", 0.5)

        # Adjust based on complexity factors
        complexity_penalty = 0
        for factor, value in complexity_factors.items():
            if factor in self.COMPLEXITY_FACTORS:
                complexity_penalty += self.COMPLEXITY_FACTORS[factor].get(value, 0.5)

        # Normalize complexity penalty (0-1 scale)
        avg_complexity_penalty = complexity_penalty / len(complexity_factors) if complexity_factors else 0.5

        # Calculate adjusted automation potential
        adjusted_potential = base_potential * (1 - avg_complexity_penalty * 0.3)  # Max 30% reduction

        # Calculate time savings
        weekly_hours = time_metrics.get('hours_per_week', 10)  # Default 10 hours
        complexity_level = pattern_data.get("complexity", "Medium").lower()
        efficiency = self.ROI_PARAMETERS["automation_efficiency"].get(complexity_level, 0.75)

        weekly_time_saved = weekly_hours * adjusted_potential * efficiency
        annual_time_saved = weekly_time_saved * 50  # 50 working weeks

        # Calculate cost savings
        hourly_rate = self.ROI_PARAMETERS["average_hourly_rate"]
        annual_savings = annual_time_saved * hourly_rate

        # Estimate implementation cost
        impl_multiplier = self.ROI_PARAMETERS["implementation_multipliers"].get(complexity_level, 4)
        implementation_cost = weekly_hours * impl_multiplier * hourly_rate

        return {
            "automation_potential": round(adjusted_potential, 2),
            "weekly_time_saved": round(weekly_time_saved, 1),
            "annual_time_saved": round(annual_time_saved, 0),
            "annual_cost_savings": round(annual_savings, 0),
            "estimated_implementation_cost": round(implementation_cost, 0),
            "payback_months": round((implementation_cost / (annual_savings / 12)), 1) if annual_savings > 0 else float('inf'),
            "complexity_score": round(avg_complexity_penalty, 2)
        }

    def analyze_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """Enhanced workflow analysis with detailed automation opportunities"""
        logger.info(f"Analyzing workflow: {workflow_description[:100]}...")

        # Extract time metrics
        time_metrics = self._extract_time_metrics(workflow_description)

        # Classify main task type
        task_category, pattern_data = self._classify_task_type(workflow_description)

        # Estimate complexity factors (can be enhanced with more sophisticated NLP)
        complexity_factors = self._estimate_complexity_factors(workflow_description)

        # Calculate automation potential
        automation_analysis = self._calculate_automation_score(task_category, time_metrics, complexity_factors)

        prompt = f"""Comprehensive workflow automation analysis:

**WORKFLOW DESCRIPTION**:
{workflow_description}

**AUTOMATED ANALYSIS RESULTS**:
- **Task Category**: {task_category.replace('_', ' ').title()}
- **Time Metrics**: {time_metrics}
- **Suggested AI Solutions**: {', '.join(pattern_data.get('ai_solutions', ['Custom solution']))}
- **Automation Potential**: {automation_analysis['automation_potential']*100:.0f}%
- **Estimated Annual Savings**: ${automation_analysis['annual_cost_savings']:,.0f}
- **Implementation Cost**: ${automation_analysis['estimated_implementation_cost']:,.0f}
- **Payback Period**: {automation_analysis['payback_months']:.1f} months

**COMPLEXITY ASSESSMENT**:
{complexity_factors}

Provide detailed analysis covering:

1. **AUTOMATION OPPORTUNITIES** (identify 3-5 specific opportunities):
   - Opportunity 1: [Specific task] → [AI solution] → [Time savings] → [Priority: High/Med/Low]
   - Opportunity 2: [Specific task] → [AI solution] → [Time savings] → [Priority: High/Med/Low]
   - Include specific AI technologies (GPT-4, Claude, custom ML, RPA, etc.)

2. **IMPLEMENTATION ROADMAP** (3-phase approach):
   - **Phase 1 (Months 1-2)**: Quick wins with highest ROI
   - **Phase 2 (Months 3-4)**: Medium complexity implementations
   - **Phase 3 (Months 5-6)**: Complex integrations and optimizations

3. **TECHNICAL REQUIREMENTS**:
   - Required AI technologies and platforms
   - Data requirements and preparation needed
   - Integration points and API requirements
   - Infrastructure and scaling considerations

4. **RESOURCE PLANNING**:
   - Team composition (developers, data scientists, business analysts)
   - Estimated development time per phase
   - Training and change management requirements
   - Budget breakdown by category

5. **RISK ASSESSMENT & MITIGATION**:
   - Technical risks (data quality, integration complexity)
   - Business risks (user adoption, process changes)
   - Mitigation strategies for each identified risk

6. **SUCCESS METRICS & KPIs**:
   - Quantitative metrics (time savings, error reduction, cost savings)
   - Qualitative metrics (user satisfaction, process improvement)
   - Measurement timeline and reporting strategy

7. **NEXT STEPS**:
   - Immediate actions to get started
   - Proof of concept recommendations
   - Pilot project suggestions

Focus on actionable, specific recommendations with realistic timelines and cost estimates."""

        response = self.chat(prompt)

        return {
            "workflow_description": workflow_description,
            "automated_analysis": {
                "task_category": task_category,
                "time_metrics": time_metrics,
                "suggested_solutions": pattern_data.get('ai_solutions', []),
                "automation_potential": automation_analysis,
                "complexity_factors": complexity_factors
            },
            "detailed_analysis": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def _estimate_complexity_factors(self, description: str) -> Dict[str, str]:
        """Estimate complexity factors from description"""
        desc_lower = description.lower()

        factors = {}

        # Data availability assessment
        if any(term in desc_lower for term in ["structured data", "database", "api", "excel", "csv"]):
            factors["data_availability"] = "high"
        elif any(term in desc_lower for term in ["manual entry", "paper", "unstructured"]):
            factors["data_availability"] = "low"
        else:
            factors["data_availability"] = "medium"

        # Process standardization
        if any(term in desc_lower for term in ["standard", "routine", "template", "checklist"]):
            factors["process_standardization"] = "high"
        elif any(term in desc_lower for term in ["varies", "different", "custom", "case-by-case"]):
            factors["process_standardization"] = "low"
        else:
            factors["process_standardization"] = "medium"

        # Human judgment requirement
        if any(term in desc_lower for term in ["decision", "judgment", "creative", "strategic", "complex analysis"]):
            factors["human_judgment_required"] = "high"
        elif any(term in desc_lower for term in ["rule-based", "simple", "straightforward", "routine"]):
            factors["human_judgment_required"] = "low"
        else:
            factors["human_judgment_required"] = "medium"

        return factors

    def assess_task_complexity(self, task_description: str, current_process: str,
                             constraints: Optional[str] = None) -> Dict[str, Any]:
        """Detailed complexity assessment for specific tasks"""
        logger.info(f"Assessing task complexity: {task_description[:50]}...")

        # Classify and analyze task
        task_category, pattern_data = self._classify_task_type(task_description)
        time_metrics = self._extract_time_metrics(f"{task_description} {current_process}")
        complexity_factors = self._estimate_complexity_factors(f"{task_description} {current_process} {constraints or ''}")

        automation_analysis = self._calculate_automation_score(task_category, time_metrics, complexity_factors)

        prompt = f"""Detailed complexity assessment for automation task:

**TASK**: {task_description}

**CURRENT PROCESS**: {current_process}

**CONSTRAINTS**: {constraints or 'None specified'}

**AUTOMATED ASSESSMENT**:
- Task Category: {task_category.replace('_', ' ').title()}
- Automation Potential: {automation_analysis['automation_potential']*100:.0f}%
- Complexity Score: {automation_analysis['complexity_score']:.2f}
- Estimated Implementation: ${automation_analysis['estimated_implementation_cost']:,.0f}

Provide comprehensive complexity analysis:

1. **IMPLEMENTATION COMPLEXITY BREAKDOWN**:
   - **Technical Complexity** (1-10): Data integration, AI model selection, accuracy requirements
   - **Business Complexity** (1-10): Process changes, stakeholder alignment, change management
   - **Integration Complexity** (1-10): System integrations, API requirements, infrastructure

2. **DETAILED REQUIREMENTS ANALYSIS**:
   - Data requirements (sources, quality, volume, format)
   - AI/ML model requirements (accuracy, latency, explainability)
   - Integration requirements (systems, APIs, workflows)
   - User interface requirements (dashboards, notifications, approvals)

3. **IMPLEMENTATION APPROACH**:
   - Recommended AI technologies and platforms
   - Development methodology (Agile, waterfall, POC-first)
   - Testing and validation strategy
   - Deployment and rollout plan

4. **EFFORT ESTIMATION**:
   - Development time breakdown by component
   - Resource requirements (FTEs by role)
   - Timeline with major milestones
   - Budget estimate with contingencies

5. **RISK ASSESSMENT**:
   - **High Risk**: Factors that could derail the project
   - **Medium Risk**: Factors requiring careful management
   - **Low Risk**: Minor considerations
   - Mitigation strategies for each risk level

6. **SUCCESS FACTORS**:
   - Critical requirements for project success
   - Key stakeholder buy-in needed
   - Technical prerequisites
   - Change management considerations"""

        response = self.chat(prompt)

        return {
            "task_details": {
                "description": task_description,
                "current_process": current_process,
                "constraints": constraints
            },
            "complexity_assessment": {
                "task_category": task_category,
                "automation_analysis": automation_analysis,
                "complexity_factors": complexity_factors
            },
            "detailed_assessment": response["content"],
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }

    def prioritize_automation_opportunities(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prioritize multiple automation opportunities using impact vs effort analysis"""
        logger.info(f"Prioritizing {len(opportunities)} automation opportunities")

        # Calculate priority scores for each opportunity
        scored_opportunities = []
        for i, opp in enumerate(opportunities):
            # Extract or estimate key metrics
            time_savings = opp.get('annual_time_saved', opp.get('hours_per_week', 10) * 50)
            cost_savings = opp.get('annual_cost_savings', time_savings * 50)
            implementation_cost = opp.get('implementation_cost', cost_savings * 0.5)
            complexity = opp.get('complexity', 'medium').lower()

            # Calculate impact score (0-100)
            impact_score = min(100, (cost_savings / 10000) * 20)  # Scale based on $10k = 20 points

            # Calculate effort score (0-100, lower is better)
            complexity_multiplier = {"low": 20, "medium": 50, "high": 80}.get(complexity, 50)
            effort_score = min(100, (implementation_cost / 5000) * 10 + complexity_multiplier)

            # Calculate priority score (impact/effort ratio)
            priority_score = impact_score / max(effort_score, 1) * 100

            scored_opportunities.append({
                **opp,
                "opportunity_id": i + 1,
                "impact_score": round(impact_score, 1),
                "effort_score": round(effort_score, 1),
                "priority_score": round(priority_score, 1),
                "estimated_cost_savings": cost_savings,
                "estimated_implementation_cost": implementation_cost
            })

        # Sort by priority score
        scored_opportunities.sort(key=lambda x: x['priority_score'], reverse=True)

        # Format for analysis
        opportunities_summary = ""
        for i, opp in enumerate(scored_opportunities[:10]):  # Top 10
            opportunities_summary += f"""
Opportunity #{opp['opportunity_id']}: {opp.get('name', opp.get('description', f'Task {i+1}'))}
- Impact Score: {opp['impact_score']}/100 (${opp['estimated_cost_savings']:,.0f} annual savings)
- Effort Score: {opp['effort_score']}/100 (${opp['estimated_implementation_cost']:,.0f} implementation)
- Priority Score: {opp['priority_score']:.1f}
- Complexity: {opp.get('complexity', 'Medium')}
"""

        prompt = f"""Prioritization analysis for {len(opportunities)} automation opportunities:

**OPPORTUNITY RANKINGS** (by priority score):
{opportunities_summary}

**PRIORITIZATION FRAMEWORK**:
- **Impact Score**: Based on annual cost savings potential (revenue impact, cost reduction, efficiency gains)
- **Effort Score**: Based on implementation cost and complexity (technical, business, integration)
- **Priority Score**: Impact/Effort ratio (higher = better ROI and feasibility)

Provide strategic prioritization recommendations:

1. **TOP 3 IMMEDIATE PRIORITIES** (Quick Wins):
   - Highest priority opportunities for immediate implementation
   - Justification for each selection
   - Expected timeline and resources for each

2. **PORTFOLIO APPROACH** (recommended implementation sequence):
   - **Wave 1 (Months 1-3)**: 2-3 quick wins to build momentum
   - **Wave 2 (Months 4-8)**: 2-3 medium complexity, high impact projects
   - **Wave 3 (Months 9-12)**: 1-2 complex, strategic initiatives

3. **RESOURCE ALLOCATION STRATEGY**:
   - Team capacity planning across waves
   - Budget allocation recommendations
   - Skill development priorities

4. **RISK MANAGEMENT**:
   - Portfolio-level risks and dependencies
   - Contingency planning for high-risk initiatives
   - Success criteria and exit strategies

5. **STRATEGIC CONSIDERATIONS**:
   - Alignment with business objectives
   - Technology platform consolidation opportunities
   - Long-term automation roadmap

6. **SUCCESS METRICS**:
   - Portfolio-level KPIs and tracking
   - Individual project success criteria
   - ROI measurement and reporting plan

Include specific recommendations for which opportunities to pursue first and why."""

        response = self.chat(prompt)

        return {
            "total_opportunities": len(opportunities),
            "scored_opportunities": scored_opportunities,
            "prioritization_analysis": response["content"],
            "portfolio_metrics": {
                "total_potential_savings": sum(opp['estimated_cost_savings'] for opp in scored_opportunities),
                "total_implementation_cost": sum(opp['estimated_implementation_cost'] for opp in scored_opportunities),
                "portfolio_roi": round(sum(opp['estimated_cost_savings'] for opp in scored_opportunities) /
                                     max(sum(opp['estimated_implementation_cost'] for opp in scored_opportunities), 1) * 100, 1),
                "average_priority_score": round(sum(opp['priority_score'] for opp in scored_opportunities) / len(scored_opportunities), 1)
            },
            "credits_used": response.get("credits_used", 0),
            "agent": self.name
        }