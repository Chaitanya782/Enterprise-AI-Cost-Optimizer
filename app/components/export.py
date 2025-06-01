"""
Export functionality for analysis results
"""
import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime


def add_export_buttons(analysis: Dict[str, Any]):
    """Add export buttons for analysis results"""
    st.markdown("### 📥 Export Analysis Results")

    col1, col2, col3 = st.columns(3)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    export_configs = [
        (col1, "📊 Download CSV", "export_csv_btn", convert_to_csv, "csv", "text/csv"),
        (col2, "📋 Download JSON", "export_json_btn", lambda x: json.dumps(x, indent=2, default=str), "json", "application/json"),
        (col3, "📧 Generate Summary", "export_summary_btn", generate_summary, None, None)
    ]

    for col, label, key, converter, file_ext, mime_type in export_configs:
        with col:
            if st.button(label, key=key):
                data = converter(analysis)

                if file_ext:  # Download button
                    st.download_button(
                        label=f"💾 Download {file_ext.upper()} File",
                        data=data,
                        file_name=f"ai_analysis_{timestamp}.{file_ext}",
                        mime=mime_type
                    )
                else:  # Text area for summary
                    st.text_area("📝 Analysis Summary", data, height=200)


def _extract_cost_data(analysis: Dict[str, Any]) -> List[List[str]]:
    """Extract cost data for CSV export"""
    rows = []
    cost_data = analysis.get("analysis", {}).get("costs", {}).get("cost_breakdown")

    if not cost_data:
        return rows

    rows.extend([
        ["Cost Analysis", ""],
        ["Model", "Monthly Cost", "Daily Cost", "Cost per Request", "Provider"]
    ])

    for model, costs in cost_data.items():
        if isinstance(costs, dict):
            rows.append([
                model,
                f"${costs.get('monthly_cost', 0):.2f}",
                f"${costs.get('daily_cost', 0):.2f}",
                f"${costs.get('cost_per_request', 0):.4f}",
                costs.get('provider', 'Unknown')
            ])

    rows.append(["", ""])
    return rows


def _extract_roi_data(analysis: Dict[str, Any]) -> List[List[str]]:
    """Extract ROI data for CSV export"""
    rows = []
    roi_data = analysis.get("analysis", {}).get("roi", {}).get("basic_metrics")

    if not roi_data:
        return rows

    rows.extend([
        ["ROI Analysis", ""],
        ["Metric", "Value"],
        ["ROI Percentage", f"{roi_data.get('roi_percentage', 0)}%"],
        ["Payback Period", f"{roi_data.get('payback_period_years', 0):.1f} years"],
        ["Net Benefit", f"${roi_data.get('net_benefit', 0):,.0f}"],
        ["", ""]
    ])

    return rows


def _extract_recommendations(analysis: Dict[str, Any]) -> List[List[str]]:
    """Extract recommendations for CSV export"""
    rows = []
    recommendations = analysis.get("recommendations")

    if not recommendations:
        return rows

    rows.append(["Recommendations", ""])
    rows.extend([f"Recommendation {i}", rec] for i, rec in enumerate(recommendations, 1))

    return rows


def convert_to_csv(analysis: Dict[str, Any]) -> str:
    """Convert analysis to CSV format"""
    try:
        rows = [
            ["Analysis Type", analysis.get("intent", "general")],
            ["Confidence", f"{analysis.get('confidence', 0):.1%}"],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""]  # Empty row
        ]

        # Add all data sections
        for extractor in [_extract_cost_data, _extract_roi_data, _extract_recommendations]:
            rows.extend(extractor(analysis))

        return pd.DataFrame(rows).to_csv(index=False, header=False)

    except Exception as e:
        st.error(f"Error generating CSV: {e}")
        return "Error generating CSV data"


def _get_cheapest_model(breakdown: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Find the cheapest model from cost breakdown"""
    return min(
        breakdown.items(),
        key=lambda x: x[1].get("monthly_cost", float('inf')) if isinstance(x[1], dict) else float('inf')
    )


def _add_section(parts: List[str], title: str, content: List[str]):
    """Add a section to summary parts"""
    parts.extend([
        f"{title}:",
        "-" * len(title),
        *content,
        ""
    ])


def generate_summary(analysis: Dict[str, Any]) -> str:
    """Generate a text summary of the analysis"""
    try:
        summary_parts = [
            "AI COST OPTIMIZATION ANALYSIS SUMMARY",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Type: {analysis.get('intent', 'general').title()}",
            f"Confidence: {analysis.get('confidence', 0):.1%}",
            ""
        ]

        # Key recommendations
        if recommendations := analysis.get("recommendations"):
            _add_section(summary_parts, "KEY RECOMMENDATIONS",
                        [f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])

        analysis_data = analysis.get("analysis", {})

        # Cost analysis
        if cost_data := analysis_data.get("costs"):
            content = []
            if breakdown := cost_data.get("cost_breakdown"):
                cheapest_model, cheapest_costs = _get_cheapest_model(breakdown)
                if isinstance(cheapest_costs, dict):
                    content.extend([
                        f"Most cost-effective model: {cheapest_model}",
                        f"Monthly cost: ${cheapest_costs.get('monthly_cost', 0):,.2f}"
                    ])
            _add_section(summary_parts, "COST ANALYSIS", content)

        # ROI analysis
        if roi_data := analysis_data.get("roi"):
            if metrics := roi_data.get("basic_metrics", {}):
                content = [
                    f"ROI: {metrics.get('roi_percentage', 0)}%",
                    f"Payback: {metrics.get('payback_period_years', 0):.1f} years",
                    f"Net Benefit: ${metrics.get('net_benefit', 0):,.0f}"
                ]
                _add_section(summary_parts, "ROI ANALYSIS", content)

        # Infrastructure savings
        if infra_data := analysis_data.get("infrastructure"):
            content = [
                f"Current spend: ${infra_data.get('current_spend', 0):,.0f}/month",
                f"Potential savings: ${infra_data.get('potential_savings', 0):,.0f}/month",
                f"Annual savings: ${infra_data.get('annual_savings', 0):,.0f}"
            ]
            _add_section(summary_parts, "INFRASTRUCTURE OPTIMIZATION", content)

        # Next steps
        _add_section(summary_parts, "NEXT STEPS", [
            "1. Review detailed analysis in each section",
            "2. Prioritize recommendations based on your constraints",
            "3. Begin with quick wins for immediate impact",
            "4. Plan phased implementation approach",
            "5. Set up monitoring and measurement systems"
        ])

        return "\n".join(summary_parts)

    except Exception as e:
        return f"Error generating summary: {e}"