"""
Enhanced data visualization components with improved spacing and design
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


def clean_text_for_display(text: str) -> str:
    """FIXED: Clean text to prevent LaTeX and formatting issues"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove all problematic formatting that causes LaTeX issues
    text = text.replace("$", "\\$")      # Escape dollar signs
    text = text.replace("**", "")        # Remove bold markdown
    text = text.replace("*", "")         # Remove italic markdown
    text = text.replace("#", "")         # Remove header markdown
    text = text.replace("`", "")         # Remove code markdown
    text = text.replace("_", "\\_")      # Escape underscores
    text = text.replace("^", "\\^")      # Escape carets
    text = text.replace("{", "\\{")      # Escape braces
    text = text.replace("}", "\\}")      # Escape braces
    text = text.replace("\\n", " ")      # Replace newlines with spaces
    text = text.replace("\\r", " ")      # Replace carriage returns
    
    return text


def create_cost_comparison_chart(cost_breakdown: Dict[str, Dict[str, float]]) -> Optional[go.Figure]:
    """Create interactive cost comparison chart"""
    if not cost_breakdown:
        return None

    try:
        periods = ['daily_cost', 'monthly_cost', 'annual_cost']
        labels = ['Daily', 'Monthly', 'Annual']
        colors = ['#667eea', '#764ba2', '#f093fb']

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
                hovertemplate=f'<b>%{{x}}</b><br>{label}: $%{{y:,.2f}}<extra></extra>',
                marker=dict(
                    line=dict(width=0),
                    opacity=0.8
                )
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
            title={
                "text": "Monthly AI Model Cost Comparison",
                "x": 0.5,
                "font": {"size": 20, "family": "Arial, sans-serif"}
            },
            xaxis_title="AI Model",
            yaxis_title="Cost (USD)",
            height=500,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12),
            margin=dict(l=60, r=60, t=100, b=60)
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
        fig.update_yaxes(tickfont=dict(size=11))
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
        payback_years = min(metrics.get("payback_period_years", 1.5), 5)

        years = list(range(6))
        cum_cost = [impl_cost] * 6
        cum_benefit = [annual_benefit * i for i in years]
        net_benefit = [cb - impl_cost for cb in cum_benefit]

        fig = go.Figure()

        # Add data traces with improved styling
        traces = [
            ("Implementation Cost", cum_cost, '#e74c3c', 'lines+markers'),
            ("Cumulative Benefits", cum_benefit, '#27ae60', 'lines+markers'),
            ("Net Benefit", net_benefit, '#3498db', 'lines+markers')
        ]

        for name, data, color, mode in traces:
            fig.add_trace(go.Scatter(
                x=years, y=data, mode=mode, name=name,
                line=dict(color=color, width=4, dash='dash' if name == 'Net Benefit' else 'solid'),
                marker=dict(size=10, line=dict(width=2, color='white')),
                fill='tonexty' if name == 'Net Benefit' else None,
                fillcolor='rgba(52,152,219,0.1)' if name == 'Net Benefit' else None,
                hovertemplate=f'Year %{{x}}<br>{name}: $%{{y:,.0f}}<extra></extra>'
            ))

        # Add break-even line
        if 0 < payback_years <= 5:
            fig.add_vline(
                x=payback_years, 
                line_width=3, 
                line_dash="dash", 
                line_color="#f39c12",
                annotation_text=f"Break-even: {payback_years:.1f}y",
                annotation_position="top"
            )

        fig.update_layout(
            title={
                "text": "ROI Timeline & Break-Even Analysis",
                "x": 0.5,
                "font": {"size": 20, "family": "Arial, sans-serif"}
            },
            xaxis_title="Years",
            yaxis_title="Amount (USD)",
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top", 
                y=0.99, 
                xanchor="left", 
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12),
            margin=dict(l=60, r=60, t=100, b=60)
        )

        return fig

    except Exception as e:
        st.error(f"Error creating ROI timeline: {e}")
        return None


def display_enhanced_analysis(analysis: Dict[str, Any]):
    """Enhanced analysis display with improved design"""
    
    # Compact executive summary with cards
    st.markdown("### 📊 Executive Summary")

    # Extract metrics
    roi_data = analysis.get("analysis", {}).get("roi", {})
    cost_data = analysis.get("analysis", {}).get("costs", {})
    infra_data = analysis.get("analysis", {}).get("infrastructure", {})
    tasks_data = analysis.get("analysis", {}).get("tasks", {})
    roi_metrics = roi_data.get("basic_metrics") or roi_data.get("key_metrics", {})

    # Compact metrics dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi_pct = roi_metrics.get("roi_percentage", 0)
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="margin: 0; color: #667eea;">ROI</h4>
            <h2 style="margin: 0.2rem 0; color: #1f2937;">{roi_pct:.0f}%</h2>
            <small style="color: #6b7280;">5-year projection</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if infra_data:
            monthly_savings = infra_data.get("potential_savings", 0)
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin: 0; color: #27ae60;">Monthly Savings</h4>
                <h2 style="margin: 0.2rem 0; color: #1f2937;">${monthly_savings:,.0f}</h2>
                <small style="color: #6b7280;">Infrastructure optimization</small>
            </div>
            """, unsafe_allow_html=True)
        elif cost_data.get("cost_breakdown"):
            cheapest_cost = min(
                (costs.get("monthly_cost", float('inf'))
                 for costs in cost_data["cost_breakdown"].values()
                 if isinstance(costs, dict)),
                default=float('inf')
            )
            if cheapest_cost < float('inf'):
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="margin: 0; color: #f093fb;">Lowest Cost</h4>
                    <h2 style="margin: 0.2rem 0; color: #1f2937;">${cheapest_cost:,.0f}</h2>
                    <small style="color: #6b7280;">Best LLM option</small>
                </div>
                """, unsafe_allow_html=True)

    with col3:
        payback = roi_metrics.get("payback_period_years", 0)
        if payback and payback != float('inf'):
            if payback < 1:
                payback_months = roi_metrics.get("payback_period_months", payback * 12)
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="margin: 0; color: #4facfe;">Payback</h4>
                    <h2 style="margin: 0.2rem 0; color: #1f2937;">{payback_months:.1f}mo</h2>
                    <small style="color: #6b7280;">Quick ROI</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="margin: 0; color: #4facfe;">Payback</h4>
                    <h2 style="margin: 0.2rem 0; color: #1f2937;">{payback:.1f}yr</h2>
                    <small style="color: #6b7280;">Long-term value</small>
                </div>
                """, unsafe_allow_html=True)

    with col4:
        confidence = analysis.get("confidence", 0)
        intent = analysis.get("intent", "general")
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="margin: 0; color: #43e97b;">Confidence</h4>
            <h2 style="margin: 0.2rem 0; color: #1f2937;">{confidence:.0%}</h2>
            <small style="color: #6b7280;">{intent.title()} focus</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Build tabs - ENHANCED to include ALL sections
    tab_data = []
    if cost_data or infra_data:
        tab_data.append(("💰 Cost Analysis", "cost", cost_data, infra_data))
    if roi_data:
        tab_data.append(("📊 ROI Analysis", "roi", roi_data, None))
    if tasks_data:
        tab_data.append(("📋 Task Analysis", "tasks", tasks_data, None))
    if analysis.get("recommendations"):
        tab_data.append(("🎯 Recommendations", "recommendations", analysis.get("recommendations"), None))

    if tab_data:
        tab_names, section_keys, primary_data, secondary_data = zip(*tab_data)
        tabs = st.tabs(tab_names)

        for tab, section_key, primary, secondary in zip(tabs, section_keys, primary_data, secondary_data):
            with tab:
                if section_key == "cost":
                    render_cost_analysis_section(primary, secondary)
                elif section_key == "roi":
                    render_roi_analysis_section(primary)
                elif section_key == "tasks":
                    render_task_analysis_section(primary)
                elif section_key == "recommendations":
                    render_recommendations_section(primary, analysis)

    # Export section at the bottom - INTEGRATED
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📥 Export Analysis Results")
    
    # Import and use the comprehensive export functionality
    from app.components.export import add_export_buttons
    add_export_buttons(analysis)


def render_cost_analysis_section(cost_data: Dict[str, Any], infra_data: Dict[str, Any]):
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
            # FIXED: Clean text for display
            clean_text = clean_text_for_display(detailed)
            st.write(clean_text)

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
        # FIXED: Clean text for display
        clean_text = clean_text_for_display(analysis_text)
        st.write(clean_text)


def render_roi_analysis_section(roi_data: Dict[str, Any]):
    """Render the ROI analysis section with timeline chart"""
    if fig := create_roi_timeline(roi_data):
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("roi_chart"))

    # Financial metrics
    if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics") or roi_data.get("financial_metrics")):
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
        # FIXED: Clean text for display
        clean_text = clean_text_for_display(detailed)
        st.write(clean_text)


def render_task_analysis_section(tasks_data: Any):
    """Render the task analysis section"""
    if isinstance(tasks_data, dict):
        if detailed := tasks_data.get("detailed_analysis"):
            st.markdown("**🔍 Detailed Task Analysis:**")
            # FIXED: Clean text for display
            clean_text = clean_text_for_display(detailed)
            st.write(clean_text)
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
                # FIXED: Clean text for display
                clean_text = clean_text_for_display(detailed)
                st.write(clean_text)
    elif isinstance(tasks_data, str):
        st.markdown("**📋 Task Analysis:**")
        # FIXED: Clean text for display
        clean_text = clean_text_for_display(tasks_data)
        st.write(clean_text)


def render_recommendations_section(recommendations: List[str], analysis: Dict[str, Any]):
    """Render the recommendations section"""
    st.markdown("#### 🎯 Key Recommendations")
    for i, rec in enumerate(recommendations, 1):
        # FIXED: Clean recommendation text
        clean_rec = clean_text_for_display(str(rec))
        st.markdown(f"**{i}.** {clean_rec}")

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