"""
Optimized data visualization components for AI Cost Optimizer
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from app.components.export import add_export_buttons
import pandas as pd
from typing import Dict, Any, List


def create_cost_comparison_chart(cost_breakdown: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create interactive cost comparison chart with time period toggle"""
    models = list(cost_breakdown.keys())
    periods = ['daily_cost', 'monthly_cost', 'annual_cost']
    labels = ['Daily', 'Monthly', 'Annual']
    colors = ['lightblue', 'blue', 'darkblue']

    traces = []
    for i, period in enumerate(periods):
        costs = [cost_breakdown[model][period] for model in models]
        traces.append(go.Bar(
            name=labels[i],
            x=models,
            y=costs,
            text=[f'${c:.2f}' if period == 'daily_cost' else f'${c:.0f}' for c in costs],
            textposition='auto',
            marker_color=colors[i],
            visible=(i == 0)
        ))

    fig = go.Figure(data=traces)

    # Add toggle buttons
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "active": 0,
            "x": 0.5,
            "y": 1.15,
            "xanchor": "center",
            "buttons": [
                {"label": label, "method": "update",
                 "args": [{"visible": [j == i for j in range(3)]},
                         {"title": f"{label} AI Cost Comparison"}]}
                for i, label in enumerate(labels)
            ]
        }],
        title="Daily AI Cost Comparison",
        xaxis_title="AI Model",
        yaxis_title="Cost (USD)",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )

    return fig


def create_roi_timeline(roi_data: Dict[str, Any]) -> go.Figure:
    """Create ROI timeline with break-even analysis"""
    metrics = roi_data.get("basic_metrics", {})
    impl_cost = metrics.get("implementation_cost", 50000)
    annual_benefit = metrics.get("annual_benefit", 120000)
    payback_years = metrics.get("payback_period_years", 1.5)
    roi_pct = metrics.get("roi_percentage", 0)

    years = list(range(6))
    cum_cost = [impl_cost] * 6
    cum_benefit = [annual_benefit * i for i in years]
    net_benefit = [cb - impl_cost for cb in cum_benefit]

    traces = [
        go.Scatter(x=years, y=cum_cost, mode='lines+markers', name='Cumulative Cost',
                  line=dict(color='red', width=3), marker=dict(size=8)),
        go.Scatter(x=years, y=cum_benefit, mode='lines+markers', name='Cumulative Benefit',
                  line=dict(color='green', width=3), marker=dict(size=8)),
        go.Scatter(x=years, y=net_benefit, mode='lines+markers', name='Net Benefit',
                  line=dict(color='blue', width=3, dash='dash'), marker=dict(size=8),
                  fill='tozeroy', fillcolor='rgba(0,100,255,0.1)')
    ]

    fig = go.Figure(data=traces)
    fig.add_vline(x=payback_years, line_width=2, line_dash="dash", line_color="gray",
                  annotation_text=f"Break-even: {payback_years:.1f} years")
    fig.add_annotation(x=5, y=net_benefit[-1], text=f"5-Year ROI: {roi_pct:.0f}%",
                      showarrow=True, arrowhead=2, bgcolor="rgba(255,255,255,0.8)", bordercolor="blue")

    fig.update_layout(
        title="ROI Timeline Analysis",
        xaxis_title="Years",
        yaxis_title="Amount (USD)",
        height=400,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def create_task_priority_matrix(tasks: List[Dict[str, Any]]) -> go.Figure:
    """Create task prioritization matrix with quadrant analysis"""
    if not tasks or not isinstance(tasks[0], dict):
        tasks = [
            {"name": "FAQ Automation", "complexity": 2, "impact": 9, "size": 100},
            {"name": "Ticket Routing", "complexity": 4, "impact": 7, "size": 80},
            {"name": "Sentiment Analysis", "complexity": 3, "impact": 6, "size": 60},
            {"name": "Report Generation", "complexity": 5, "impact": 8, "size": 70},
            {"name": "Data Entry", "complexity": 1, "impact": 5, "size": 90}
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t["complexity"] for t in tasks],
        y=[t["impact"] for t in tasks],
        mode='markers+text',
        marker=dict(
            size=[t.get("size", 50) for t in tasks],
            color=[t["impact"] for t in tasks],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Impact Score")
        ),
        text=[t["name"] for t in tasks],
        textposition="top center"
    ))

    # Add quadrant lines and labels
    fig.add_hline(y=6.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=3, line_dash="dash", line_color="gray", opacity=0.5)

    quadrants = [
        (1.5, 9, "Quick Wins", "green"),
        (4.5, 9, "Strategic Projects", "blue"),
        (1.5, 3, "Fill-ins", "gray"),
        (4.5, 3, "Questionable", "red")
    ]

    for x, y, text, color in quadrants:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=14, color=color))

    fig.update_layout(
        title="Task Prioritization Matrix",
        xaxis_title="Implementation Complexity →",
        yaxis_title="Business Impact →",
        height=500,
        xaxis=dict(range=[0, 6]),
        yaxis=dict(range=[0, 10]),
        showlegend=False
    )

    return fig


def create_savings_dashboard(analysis_data: Dict[str, Any]) -> None:
    """Create comprehensive savings metrics dashboard"""
    roi_metrics = analysis_data.get("analysis", {}).get("roi", {}).get("basic_metrics", {})

    metrics = [
        ("Total Savings", f"${roi_metrics.get('net_benefit', 0):,.0f}",
         f"{roi_metrics.get('roi_percentage', 0):.0f}% ROI"),
        ("Time Saved", "2,000 hrs/year", "↑ Efficiency"),
        ("Accuracy", "95%", "↑ vs Manual"),
        ("Payback Period", f"{roi_metrics.get('payback_period_years', 0):.1f} years",
         "Quick ROI" if roi_metrics.get('payback_period_years', 0) < 2 else "Long-term")
    ]

    cols = st.columns(4)
    for i, (label, value, delta) in enumerate(metrics):
        with cols[i]:
            delta_color = "normal" if i < 3 else ("normal" if "Quick" in delta else "inverse")
            st.metric(label, value, delta, delta_color=delta_color)


def render_infrastructure_analysis(infra: Dict[str, Any], analysis: Dict[str, Any]):
    """Render infrastructure cost optimization analysis"""
    st.subheader("🏗️ AI Infrastructure Cost Optimization Analysis")

    # Key metrics
    metrics = [
        ("Current Spend", f"${infra['current_spend']:,.0f}/mo", f"${infra['current_spend'] * 12:,.0f}/year"),
        ("Target Spend", f"${infra['target_spend']:,.0f}/mo", f"-{infra['reduction_percentage']:.0f}%"),
        ("Monthly Savings", f"${infra['potential_savings']:,.0f}", "Immediate impact"),
        ("Annual Savings", f"${infra['annual_savings']:,.0f}", "12-month projection")
    ]

    cols = st.columns(4)
    for i, (label, value, delta) in enumerate(metrics):
        with cols[i]:
            delta_color = "inverse" if i == 1 else "normal"
            st.metric(label, value, delta, delta_color=delta_color)

    # Analysis tabs
    tabs = st.tabs(["📋 Strategy", "⚠️ Risks", "📅 Timeline", "💰 ROI"])

    with tabs[0]:
        st.markdown("### Cost Optimization Strategy")
        st.write(infra.get("detailed_analysis", "Analysis pending..."))

    with tabs[1]:
        st.markdown("### Risk Assessment")
        st.warning("Risk levels are categorized based on potential impact to operations")

    with tabs[2]:
        st.markdown("### Implementation Timeline")
        timeline = {
            "Week 1-2": ["Audit current usage", "Identify quick wins", "Test alternatives"],
            "Week 3-4": ["Implement caching", "Optimize prompts", "A/B test models"],
            "Month 2": ["Migrate non-critical workloads", "Monitor performance"],
            "Month 3": ["Full migration", "Performance optimization", "Cost tracking"]
        }
        for phase, tasks in timeline.items():
            with st.expander(f"📅 {phase}"):
                for task in tasks:
                    st.write(f"• {task}")

    with tabs[3]:
        st.markdown("### 12-Month ROI Projection")
        if viz_data := analysis.get("visualizations", {}).get("savings_timeline"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=viz_data["months"],
                y=viz_data["cumulative_savings"],
                mode='lines+markers',
                name='Cumulative Savings',
                line=dict(color='green', width=3)
            ))
            fig.update_layout(
                title="Projected Savings Timeline",
                xaxis_title="Month",
                yaxis_title="Cumulative Savings ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def display_enhanced_analysis(analysis: Dict[str, Any]):
    """Enhanced analysis display with comprehensive visualizations"""
    create_savings_dashboard(analysis)

    tabs = st.tabs(["💰 Cost Analysis", "📊 ROI Analysis", "📋 Task Analysis", "🎯 Recommendations"])

    with tabs[0]:
        if cost_data := analysis.get("analysis", {}).get("costs"):
            if "cost_breakdown" in cost_data:
                fig = create_cost_comparison_chart(cost_data["cost_breakdown"])

                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{tabs[0]}")

                st.subheader("Detailed Cost Breakdown")
                cost_df = pd.DataFrame.from_dict(cost_data["cost_breakdown"], orient='index').round(2)
                st.dataframe(
                    cost_df.style.format({
                        'daily_cost': '${:.2f}',
                        'monthly_cost': '${:.0f}',
                        'annual_cost': '${:.0f}',
                        'cost_per_request': '${:.4f}'
                    }).highlight_min(axis=0, color='lightgreen'),
                    use_container_width=True
                )

            if "recommendations" in cost_data:
                st.info(f"**💡 Cost Optimization Tips:** {cost_data['recommendations']}")

            if infra := analysis.get("analysis", {}).get("infrastructure"):
                render_infrastructure_analysis(infra, analysis)

    with tabs[1]:
        if roi_data := analysis.get("analysis", {}).get("roi"):
            fig = create_roi_timeline(roi_data)
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{tabs[1]}")

            if metrics := roi_data.get("basic_metrics"):
                cols = st.columns(3)
                metric_data = [
                    ("Implementation Cost", metrics['implementation_cost']),
                    ("Annual Benefit", metrics['annual_benefit']),
                    ("5-Year Net Benefit", metrics['net_benefit'])
                ]
                for col, (label, value) in zip(cols, metric_data):
                    with col:
                        st.metric(label, f"${value:,.0f}")

            if "detailed_analysis" in roi_data:
                with st.expander("📈 Detailed Financial Analysis"):
                    st.write(roi_data["detailed_analysis"])

    with tabs[2]:
        if "tasks" in analysis.get("analysis", {}):
            st.subheader("Task Automation Opportunities")
            fig = create_task_priority_matrix([])
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{tabs[2]}")
            st.write(analysis["analysis"]["tasks"])

    with tabs[3]:
        if recommendations := analysis.get("recommendations"):
            st.subheader("🎯 Action Plan")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

            st.subheader("📅 Implementation Roadmap")
            roadmap = {
                "Phase 1 (Month 1-2)": ["Select pilot use case", "Set up AI infrastructure", "Train initial models"],
                "Phase 2 (Month 3-4)": ["Deploy pilot project", "Gather user feedback", "Optimize performance"],
                "Phase 3 (Month 5-6)": ["Scale to full deployment", "Monitor ROI metrics", "Plan next use cases"]
            }
            for phase, tasks in roadmap.items():
                with st.expander(phase):
                    for task in tasks:
                        st.write(f"• {task}")

    st.divider()
    st.subheader("📥 Export Results")
    add_export_buttons(analysis)