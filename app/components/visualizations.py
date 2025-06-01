"""
Optimized data visualization components for AI Cost Optimizer
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime


def generate_unique_key(prefix: str = "chart") -> str:
    """Generate unique key for Streamlit components"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_cost_comparison_chart(cost_breakdown: Dict[str, Dict[str, float]]) -> Optional[go.Figure]:
    """Create interactive cost comparison chart with time period toggle"""
    if not cost_breakdown:
        return None

    try:
        periods = ['daily_cost', 'monthly_cost', 'annual_cost']
        labels = ['Daily', 'Monthly', 'Annual']
        colors = ['#3498db', '#2980b9', '#1f4e79']

        # Filter valid models
        valid_models = [
            model for model in cost_breakdown.keys()
            if isinstance(cost_breakdown[model], dict) and
            all(period in cost_breakdown[model] for period in periods)
        ]

        if not valid_models:
            return None

        traces = []
        for i, (period, label, color) in enumerate(zip(periods, labels, colors)):
            model_costs = [(model, cost_breakdown[model][period])
                          for model in valid_models
                          if cost_breakdown[model].get(period, 0) > 0]

            if not model_costs:
                continue

            models, costs = zip(*model_costs)
            text_format = '${:.2f}' if period == 'daily_cost' else '${:,.0f}'

            traces.append(go.Bar(
                name=label,
                x=models,
                y=costs,
                text=[text_format.format(c) for c in costs],
                textposition='auto',
                marker_color=color,
                visible=(i == 1),  # Show monthly by default
                hovertemplate=f'<b>%{{x}}</b><br>{label}: $%{{y:,.2f}}<extra></extra>'
            ))

        if not traces:
            return None

        fig = go.Figure(data=traces)
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "direction": "right",
                "active": 1,
                "x": 0.5,
                "y": 1.15,
                "xanchor": "center",
                "buttons": [
                    {"label": label, "method": "update",
                     "args": [{"visible": [j == i for j in range(len(traces))]},
                             {"title": f"{label} AI Model Cost Comparison"}]}
                    for i, label in enumerate(labels) if i < len(traces)
                ]
            }],
            title="Monthly AI Model Cost Comparison",
            xaxis_title="AI Model",
            yaxis_title="Cost (USD)",
            height=450,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(tickangle=45)
        return fig

    except Exception as e:
        st.error(f"Error creating cost chart: {e}")
        return None


def create_roi_timeline(roi_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create ROI timeline with break-even analysis"""
    try:
        metrics = (roi_data.get("basic_metrics") or
                  roi_data.get("key_metrics") or
                  roi_data.get("financial_metrics", {}))

        if not metrics:
            return None

        impl_cost = metrics.get("implementation_cost", 50000)
        annual_benefit = metrics.get("annual_benefit", metrics.get("total_annual_benefit", 120000))
        payback_years = min(metrics.get("payback_period_years", 1.5), 5)  # Cap at 5 years

        years = list(range(6))
        cum_cost = [impl_cost] * 6
        cum_benefit = [annual_benefit * i for i in years]
        net_benefit = [cb - impl_cost for cb in cum_benefit]

        fig = go.Figure()

        # Add data traces
        traces = [
            ("Implementation Cost", cum_cost, '#e74c3c', 'lines+markers'),
            ("Cumulative Benefits", cum_benefit, '#27ae60', 'lines+markers'),
            ("Net Benefit", net_benefit, '#3498db', 'lines+markers')
        ]

        for name, data, color, mode in traces:
            fig.add_trace(go.Scatter(
                x=years, y=data, mode=mode, name=name,
                line=dict(color=color, width=3, dash='dash' if name == 'Net Benefit' else 'solid'),
                marker=dict(size=8),
                fill='tonexty' if name == 'Net Benefit' else None,
                fillcolor='rgba(52,152,219,0.1)' if name == 'Net Benefit' else None,
                hovertemplate=f'Year %{{x}}<br>{name}: $%{{y:,.0f}}<extra></extra>'
            ))

        # Add break-even line and annotation
        if 0 < payback_years <= 5:
            fig.add_vline(x=payback_years, line_width=2, line_dash="dash", line_color="orange",
                          annotation_text=f"Break-even: {payback_years:.1f}y")

        final_roi = ((cum_benefit[-1] - impl_cost) / impl_cost * 100) if impl_cost > 0 else 0
        fig.add_annotation(
            x=4.5, y=max(net_benefit[-1], 0),
            text=f"5-Year ROI: {final_roi:.0f}%",
            showarrow=True, arrowhead=2,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#3498db",
            font=dict(size=12)
        )

        fig.update_layout(
            title="ROI Timeline & Break-Even Analysis",
            xaxis_title="Years",
            yaxis_title="Amount (USD)",
            height=450,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    except Exception as e:
        st.error(f"Error creating ROI timeline: {e}")
        return None


def create_task_priority_matrix(tasks_data: Any) -> go.Figure:
    """Create task prioritization matrix with quadrant analysis"""
    try:
        # Extract tasks from different data formats
        if isinstance(tasks_data, dict) and "automated_analysis" in tasks_data:
            auto_analysis = tasks_data["automated_analysis"]
            if "automation_analysis" in auto_analysis:
                aa = auto_analysis["automation_analysis"]
                tasks = [{
                    "name": "Primary Task",
                    "complexity": int(aa.get("complexity_score", 0.5) * 10),
                    "impact": int(aa.get("automation_potential", 0.5) * 10),
                    "size": aa.get("annual_cost_savings", 10000) / 1000
                }]
            else:
                tasks = []
        elif isinstance(tasks_data, list):
            tasks = tasks_data
        else:
            tasks = []

        # Default tasks if no data
        if not tasks:
            tasks = [
                {"name": "Customer Support", "complexity": 3, "impact": 9, "size": 120},
                {"name": "Content Generation", "complexity": 2, "impact": 8, "size": 100},
                {"name": "Document Processing", "complexity": 4, "impact": 8, "size": 90},
                {"name": "Data Entry", "complexity": 1, "impact": 6, "size": 80},
                {"name": "Report Generation", "complexity": 5, "impact": 7, "size": 70},
                {"name": "Email Routing", "complexity": 2, "impact": 5, "size": 60}
            ]

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=[t["complexity"] for t in tasks],
            y=[t["impact"] for t in tasks],
            mode='markers+text',
            marker=dict(
                size=[max(20, min(80, t.get("size", 50))) for t in tasks],
                color=[t["impact"] for t in tasks],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Impact Score", x=1.02),
                line=dict(width=2, color='white')
            ),
            text=[t["name"] for t in tasks],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertemplate='<b>%{text}</b><br>Complexity: %{x}<br>Impact: %{y}<extra></extra>'
        ))

        # Add quadrant lines and labels
        fig.add_hline(y=6.5, line_dash="dash", line_color="gray", opacity=0.7, line_width=1)
        fig.add_vline(x=3, line_dash="dash", line_color="gray", opacity=0.7, line_width=1)

        quadrants = [
            (1.5, 9.2, "🚀 Quick Wins", "#27ae60"),
            (4.5, 9.2, "📈 Strategic Projects", "#3498db"),
            (1.5, 2.8, "⏳ Fill-ins", "#95a5a6"),
            (4.5, 2.8, "❓ Questionable", "#e74c3c")
        ]

        for x, y, text, color in quadrants:
            fig.add_annotation(
                x=x, y=y, text=text, showarrow=False,
                font=dict(size=12, color=color, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color, borderwidth=1
            )

        fig.update_layout(
            title="Task Prioritization Matrix",
            xaxis_title="Implementation Complexity →",
            yaxis_title="Business Impact →",
            height=500,
            xaxis=dict(range=[0, 6], gridcolor='lightgray'),
            yaxis=dict(range=[0, 10], gridcolor='lightgray'),
            showlegend=False,
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    except Exception as e:
        st.error(f"Error creating task matrix: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Task data not available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Task Prioritization Matrix")
        return fig


def create_savings_breakdown_chart(savings_data: Dict[str, float]) -> go.Figure:
    """Create pie chart for savings breakdown"""
    try:
        if not savings_data:
            savings_data = {
                "Labor Cost Reduction": 60000,
                "Efficiency Gains": 25000,
                "Error Reduction": 15000,
                "Process Optimization": 10000
            }

        fig = go.Figure(data=[go.Pie(
            labels=list(savings_data.keys()),
            values=list(savings_data.values()),
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(colors=['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
        )])

        fig.update_layout(
            title="Annual Savings Breakdown",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
        )

        return fig

    except Exception as e:
        st.error(f"Error creating savings chart: {e}")
        return go.Figure()


def _get_key_metrics(analysis: Dict[str, Any]) -> tuple:
    """Extract key metrics from analysis data"""
    roi_data = analysis.get("analysis", {}).get("roi", {})
    cost_data = analysis.get("analysis", {}).get("costs", {})
    infra_data = analysis.get("analysis", {}).get("infrastructure", {})

    roi_metrics = roi_data.get("basic_metrics") or roi_data.get("key_metrics", {})

    return roi_data, cost_data, infra_data, roi_metrics


def display_enhanced_analysis(analysis: Dict[str, Any]):
    """Enhanced analysis display with horizontal tabs"""
    st.markdown("### 📊 Executive Summary")

    # Extract metrics
    roi_data, cost_data, infra_data, roi_metrics = _get_key_metrics(analysis)

    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi_pct = roi_metrics.get("roi_percentage", 0)
        st.metric("ROI", f"{roi_pct:.0f}%", delta="5-year projection")

    with col2:
        if infra_data:
            monthly_savings = infra_data.get("potential_savings", 0)
            st.metric("Monthly Savings", f"${monthly_savings:,.0f}", delta="Infrastructure optimization")
        elif cost_data.get("cost_breakdown"):
            cheapest_cost = min(
                (costs.get("monthly_cost", float('inf'))
                 for costs in cost_data["cost_breakdown"].values()
                 if isinstance(costs, dict)),
                default=float('inf')
            )
            if cheapest_cost < float('inf'):
                st.metric("Lowest Monthly Cost", f"${cheapest_cost:,.0f}", delta="Best LLM option")
            else:
                st.metric("Cost Analysis", "Available", delta="Multiple models compared")
        else:
            st.metric("Cost Analysis", "Available", delta="See detailed breakdown")

    with col3:
        payback = roi_metrics.get("payback_period_years", 0)
        if payback and payback != float('inf'):
            if payback < 1:
                payback_months = roi_metrics.get("payback_period_months", payback * 12)
                st.metric("Payback", f"{payback_months:.1f} months", delta="Quick ROI")
            else:
                st.metric("Payback", f"{payback:.1f} years", delta="Long-term value")
        else:
            st.metric("Implementation", "Ready", delta="Analysis complete")

    with col4:
        confidence = analysis.get("confidence", 0)
        intent = analysis.get("intent", "general")
        st.metric("Analysis Confidence", f"{confidence:.0%}", delta=f"{intent.title()} focus")

    st.divider()

    # Build tabs
    tab_data = []
    if cost_data or infra_data:
        tab_data.append(("💰 Cost Analysis", "cost", cost_data, infra_data))
    if roi_data:
        tab_data.append(("📊 ROI Analysis", "roi", roi_data, None))
    if analysis.get("analysis", {}).get("tasks"):
        tab_data.append(("📋 Task Analysis", "tasks", analysis.get("analysis", {}).get("tasks"), None))
    if analysis.get("recommendations"):
        tab_data.append(("🎯 Recommendations", "recommendations", analysis.get("recommendations"), None))

    if tab_data:
        tab_names, section_keys, primary_data, secondary_data = zip(*tab_data)
        tabs = st.tabs(tab_names)

        for tab, section_key, primary, secondary in zip(tabs, section_keys, primary_data, secondary_data):
            with tab:
                if section_key == "cost":
                    render_cost_analysis_section(primary, secondary, analysis)
                elif section_key == "roi":
                    render_roi_analysis_section(primary, analysis)
                elif section_key == "tasks":
                    render_task_analysis_section(primary, analysis)
                elif section_key == "recommendations":
                    render_recommendations_section(primary, analysis)

    st.divider()
    render_export_section(analysis)


def render_cost_analysis_section(cost_data: Dict[str, Any], infra_data: Dict[str, Any], analysis: Dict[str, Any]):
    """Render the cost analysis section with charts"""
    # Infrastructure analysis
    if infra_data:
        st.markdown("#### 🏗️ Infrastructure Cost Optimization")

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Current Spend", infra_data.get('current_spend', 0), "/mo"),
            ("Target Spend", infra_data.get('target_spend', 0), "/mo"),
            ("Reduction", infra_data.get('reduction_percentage', 0), "%"),
            ("Annual Savings", infra_data.get('annual_savings', 0), "")
        ]

        for col, (label, value, suffix) in zip([col1, col2, col3, col4], metrics):
            with col:
                if "%" in suffix:
                    st.metric(label, f"{value:.0f}{suffix}")
                else:
                    st.metric(label, f"${value:,.0f}{suffix}")

        if detailed := infra_data.get("detailed_analysis"):
            st.markdown("**📋 Optimization Strategy:**")
            st.write(detailed)

    # LLM cost comparison
    if cost_data and cost_data.get("cost_breakdown"):
        st.markdown("#### 💵 LLM Cost Comparison")

        if fig := create_cost_comparison_chart(cost_data["cost_breakdown"]):
            st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("cost_chart"))

        # Cost breakdown table
        st.markdown("**📊 Detailed Cost Breakdown:**")
        breakdown_df = pd.DataFrame.from_dict(cost_data["cost_breakdown"], orient='index')

        if not breakdown_df.empty:
            display_cols = ['monthly_cost', 'cost_per_request', 'provider', 'tier']
            available_cols = [col for col in display_cols if col in breakdown_df.columns]

            if available_cols:
                display_df = breakdown_df[available_cols].copy()

                # Format currency columns
                if 'monthly_cost' in display_df.columns:
                    display_df['monthly_cost'] = display_df['monthly_cost'].apply(lambda x: f"${x:,.2f}")
                if 'cost_per_request' in display_df.columns:
                    display_df['cost_per_request'] = display_df['cost_per_request'].apply(lambda x: f"${x:.4f}")

                display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
                st.dataframe(display_df, use_container_width=True)

        # Volume metrics
        if vm := cost_data.get("volume_metrics"):
            st.markdown("**📈 Usage Metrics:**")
            col1, col2, col3 = st.columns(3)
            metrics = [
                ("Daily Requests", vm.get('daily_requests', 0)),
                ("Monthly Requests", vm.get('monthly_requests', 0)),
                ("Avg Tokens/Request", vm.get('avg_input_tokens', 0) + vm.get('avg_output_tokens', 0))
            ]

            for col, (label, value) in zip([col1, col2, col3], metrics):
                with col:
                    st.info(f"**{label}:** {value:,}")

    if analysis_text := cost_data.get("analysis"):
        st.markdown("**🔍 Detailed Cost Analysis:**")
        st.write(analysis_text)


def render_roi_analysis_section(roi_data: Dict[str, Any], analysis: Dict[str, Any]):
    """Render the ROI analysis section with timeline chart"""
    if fig := create_roi_timeline(roi_data):
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("roi_chart"))

    # Financial metrics
    if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics")):
        st.markdown("#### 💰 Financial Summary")
        col1, col2, col3 = st.columns(3)

        financial_metrics = [
            ("Implementation Cost", metrics.get("implementation_cost", 0)),
            ("Annual Benefit", metrics.get("annual_benefit", 0)),
            ("5-Year Net Benefit", metrics.get("net_benefit", 0))
        ]

        for col, (label, value) in zip([col1, col2, col3], financial_metrics):
            with col:
                st.metric(label, f"${value:,.0f}")

    # Scenario analysis
    if scenarios := roi_data.get("scenario_analysis"):
        st.markdown("#### 📈 Scenario Analysis")
        scenario_cols = st.columns(3)

        for i, (scenario, data) in enumerate(scenarios.items()):
            if i < 3:  # Limit to 3 scenarios
                with scenario_cols[i]:
                    roi_pct = data.get("roi_percentage", 0)
                    payback = data.get("payback_period_months", 0)
                    st.metric(f"{scenario.title()} Scenario", f"{roi_pct}% ROI", f"{payback:.1f}mo payback")

    if detailed := roi_data.get("detailed_analysis"):
        st.markdown("**🔍 Detailed ROI Analysis:**")
        st.write(detailed)


def render_task_analysis_section(tasks_data: Any, analysis: Dict[str, Any]):
    """Render the task analysis section with priority matrix"""
    fig = create_task_priority_matrix(tasks_data)
    st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("task_chart"))

    if isinstance(tasks_data, dict):
        if detailed := tasks_data.get("detailed_analysis"):
            st.markdown("**🔍 Detailed Task Analysis:**")
            st.write(detailed)
        elif auto_analysis := tasks_data.get("automated_analysis"):
            if aa := auto_analysis.get("automation_analysis"):
                st.markdown("#### 🤖 Automation Potential")

                col1, col2, col3 = st.columns(3)
                automation_metrics = [
                    ("Automation Potential", aa.get("automation_potential", 0), "%"),
                    ("Weekly Time Saved", aa.get("weekly_time_saved", 0), " hours"),
                    ("Annual Savings", aa.get("annual_cost_savings", 0), "")
                ]

                for col, (label, value, suffix) in zip([col1, col2, col3], automation_metrics):
                    with col:
                        if "%" in suffix:
                            st.metric(label, f"{value:.0%}")
                        elif "$" not in suffix and suffix:
                            st.metric(label, f"{value:.1f}{suffix}")
                        else:
                            st.metric(label, f"${value:,.0f}")

            if detailed := auto_analysis.get("detailed_analysis"):
                st.markdown("**📋 Implementation Analysis:**")
                st.write(detailed)
    elif isinstance(tasks_data, str):
        st.markdown("**📋 Task Analysis:**")
        st.write(tasks_data)


def render_recommendations_section(recommendations: List[str], analysis: Dict[str, Any]):
    """Render the recommendations section"""
    st.markdown("#### 🎯 Key Recommendations")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("#### 📅 Implementation Roadmap")

    # Generate roadmap based on analysis intent
    intent = analysis.get("intent", "general")

    roadmaps = {
        "comprehensive": {
            "Phase 1 (Month 1-2)": [
                "🎯 Implement quick wins identified in cost analysis",
                "🏗️ Set up infrastructure optimizations",
                "📊 Establish baseline metrics and KPIs"
            ],
            "Phase 2 (Month 3-4)": [
                "🤖 Deploy high-priority automation tasks",
                "💰 Monitor cost savings and ROI metrics",
                "🔄 Iterate based on initial results"
            ],
            "Phase 3 (Month 5-6)": [
                "📈 Scale successful implementations",
                "🎨 Optimize and fine-tune systems",
                "📋 Plan next phase of automation"
            ]
        },
        "cost_analysis": {
            "Week 1-2": [
                "🔍 Audit current LLM usage patterns",
                "💰 Implement cost monitoring tools",
                "⚡ Deploy quick optimization wins"
            ],
            "Week 3-4": [
                "🔄 A/B test alternative models",
                "📊 Monitor performance vs cost trade-offs",
                "🎯 Implement recommended cost optimizations"
            ],
            "Month 2": [
                "📈 Scale optimizations across all use cases",
                "📋 Establish ongoing cost governance",
                "🎨 Fine-tune for optimal cost-performance"
            ]
        }
    }

    roadmap = roadmaps.get(intent, {
        "Phase 1": [
            "📊 Complete detailed requirements analysis",
            "🎯 Prioritize implementation opportunities",
            "📋 Prepare project plan and resources"
        ],
        "Phase 2": [
            "🚀 Begin implementation of top priorities",
            "📈 Monitor progress and metrics",
            "🔄 Adjust approach based on results"
        ],
        "Phase 3": [
            "📊 Measure and report on outcomes",
            "🎨 Optimize and scale successful initiatives",
            "📅 Plan next iteration"
        ]
    })

    for phase, tasks in roadmap.items():
        st.markdown(f"**📅 {phase}**")
        for task in tasks:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {task}")
        st.markdown("")


def render_export_section(analysis: Dict[str, Any]):
    """Render export options"""
    st.markdown("### 📥 Export Analysis Results")

    col1, col2, col3 = st.columns(3)

    export_buttons = [
        ("📊 Export to CSV", "export_csv", "CSV export functionality ready"),
        ("📋 Generate Report", "export_report", "Report generation functionality ready"),
        ("📧 Email Summary", "export_email", "Email functionality ready for implementation")
    ]

    for col, (label, key, message) in zip([col1, col2, col3], export_buttons):
        with col:
            if st.button(label, key=key):
                st.success(message)

    # Raw data toggle
    if st.checkbox("🔧 Show Raw Analysis Data", key="show_raw_data"):
        st.markdown("**Raw Analysis JSON:**")
        st.json(analysis)