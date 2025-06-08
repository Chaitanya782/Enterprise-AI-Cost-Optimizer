"""
Fixed export functionality with Streamlit compatibility and proper text handling
"""
import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime
import io
import base64


def safe_convert_to_csv(analysis: Dict[str, Any]) -> str:
    """Safely convert analysis to CSV format with ALL sections included"""
    try:
        rows = [
            ["Analysis Type", analysis.get("intent", "general")],
            ["Confidence", f"{analysis.get('confidence', 0):.1%}"],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""]  # Empty row
        ]

        # Add recommendations
        if recommendations := analysis.get("recommendations"):
            rows.append(["RECOMMENDATIONS", ""])
            for i, rec in enumerate(recommendations, 1):
                # Clean recommendation text
                clean_rec = str(rec).replace('\n', ' ').replace('\r', ' ').replace('*', '').replace('#', '')
                rows.append([f"Recommendation {i}", clean_rec])
            rows.append(["", ""])

        # Add ALL analysis sections
        analysis_data = analysis.get("analysis", {})
        
        # Cost analysis section
        if cost_data := analysis_data.get("costs"):
            rows.append(["COST ANALYSIS", ""])
            
            # Volume metrics
            if vm := cost_data.get("volume_metrics"):
                rows.append(["Volume Metrics", ""])
                rows.append(["Daily Requests", f"{vm.get('daily_requests', 0):,}"])
                rows.append(["Monthly Requests", f"{vm.get('monthly_requests', 0):,}"])
                rows.append(["Input Tokens", f"{vm.get('avg_input_tokens', 0):,}"])
                rows.append(["Output Tokens", f"{vm.get('avg_output_tokens', 0):,}"])
                rows.append(["", ""])
            
            # Cost breakdown
            if breakdown := cost_data.get("cost_breakdown"):
                rows.append(["Model Cost Comparison", ""])
                rows.append(["Model", "Monthly Cost", "Daily Cost", "Cost per Request", "Provider", "Tier"])
                for model, costs in breakdown.items():
                    if isinstance(costs, dict) and "monthly_cost" in costs:
                        rows.append([
                            str(model),
                            f"${costs.get('monthly_cost', 0):.2f}",
                            f"${costs.get('daily_cost', 0):.2f}",
                            f"${costs.get('cost_per_request', 0):.4f}",
                            str(costs.get('provider', 'Unknown')),
                            str(costs.get('tier', 'Unknown'))
                        ])
                rows.append(["", ""])
            
            # Cost analysis text
            if cost_analysis := cost_data.get("analysis"):
                rows.append(["Cost Analysis Details", clean_text_for_csv(cost_analysis)])
                rows.append(["", ""])

        # ROI analysis section
        if roi_data := analysis_data.get("roi"):
            rows.append(["ROI ANALYSIS", ""])
            
            # Financial metrics
            if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics") or roi_data.get("financial_metrics")):
                rows.append(["Financial Metrics", ""])
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if "percentage" in key:
                            rows.append([key.replace("_", " ").title(), f"{value}%"])
                        elif any(term in key for term in ["cost", "benefit", "npv"]):
                            rows.append([key.replace("_", " ").title(), f"${value:,.0f}"])
                        else:
                            rows.append([key.replace("_", " ").title(), str(value)])
                rows.append(["", ""])
            
            # ROI analysis text
            if roi_analysis := roi_data.get("detailed_analysis"):
                rows.append(["ROI Analysis Details", clean_text_for_csv(roi_analysis)])
                rows.append(["", ""])

        # Task analysis section
        if tasks_data := analysis_data.get("tasks"):
            rows.append(["TASK ANALYSIS", ""])
            if isinstance(tasks_data, dict):
                if detailed := tasks_data.get("detailed_analysis"):
                    rows.append(["Task Analysis Details", clean_text_for_csv(detailed)])
                elif auto_analysis := tasks_data.get("automated_analysis"):
                    if aa := auto_analysis.get("automation_analysis"):
                        rows.append(["Automation Potential", f"{aa.get('automation_potential', 0):.0%}"])
                        rows.append(["Weekly Time Saved", f"{aa.get('weekly_time_saved', 0):.1f} hours"])
                        rows.append(["Annual Savings", f"${aa.get('annual_cost_savings', 0):,.0f}"])
            elif isinstance(tasks_data, str):
                rows.append(["Task Analysis Details", clean_text_for_csv(tasks_data)])
            rows.append(["", ""])

        # Infrastructure analysis section
        if infra_data := analysis_data.get("infrastructure"):
            rows.append(["INFRASTRUCTURE ANALYSIS", ""])
            rows.append(["Current Monthly Spend", f"${infra_data.get('current_spend', 0):,.0f}"])
            rows.append(["Target Monthly Spend", f"${infra_data.get('target_spend', 0):,.0f}"])
            rows.append(["Potential Monthly Savings", f"${infra_data.get('potential_savings', 0):,.0f}"])
            rows.append(["Annual Savings", f"${infra_data.get('annual_savings', 0):,.0f}"])
            if detailed := infra_data.get("detailed_analysis"):
                rows.append(["Infrastructure Analysis Details", clean_text_for_csv(detailed)])
            rows.append(["", ""])

        # Extracted metrics
        if metrics := analysis.get("metrics"):
            rows.append(["EXTRACTED METRICS", ""])
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if any(term in key for term in ["spend", "budget", "cost"]):
                        rows.append([key.replace("_", " ").title(), f"${value:,.0f}"])
                    elif "percentage" in key or "reduction" in key:
                        rows.append([key.replace("_", " ").title(), f"{value:.1%}"])
                    else:
                        rows.append([key.replace("_", " ").title(), f"{value:,}"])
                else:
                    rows.append([key.replace("_", " ").title(), str(value)])
            rows.append(["", ""])

        # Convert to DataFrame and then CSV
        df = pd.DataFrame(rows)
        return df.to_csv(index=False, header=False)

    except Exception as e:
        st.error(f"Error generating CSV: {e}")
        # Return basic CSV with error info
        error_df = pd.DataFrame([
            ["Error", "Failed to generate full CSV"],
            ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Analysis Type", analysis.get("intent", "unknown")]
        ])
        return error_df.to_csv(index=False, header=False)


def clean_text_for_csv(text: str) -> str:
    """Clean text for CSV export by removing problematic characters"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove markdown formatting
    text = text.replace('**', '')  # Bold
    text = text.replace('*', '')   # Italic
    text = text.replace('#', '')   # Headers
    text = text.replace('`', '')   # Code
    text = text.replace('\n', ' ') # Newlines
    text = text.replace('\r', ' ') # Carriage returns
    text = text.replace('"', "'")  # Quotes
    
    # Limit length for CSV
    return text[:500] + "..." if len(text) > 500 else text


def safe_generate_summary(analysis: Dict[str, Any]) -> str:
    """Generate comprehensive text summary with ALL sections"""
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
            summary_parts.extend([
                "KEY RECOMMENDATIONS:",
                "-" * 20
            ])
            for i, rec in enumerate(recommendations, 1):
                clean_rec = clean_text_for_csv(rec)[:200]
                summary_parts.append(f"{i}. {clean_rec}")
            summary_parts.append("")

        # Analysis sections
        analysis_data = analysis.get("analysis", {})

        # Cost analysis
        if cost_data := analysis_data.get("costs"):
            summary_parts.extend([
                "COST ANALYSIS:",
                "-" * 15
            ])
            
            if vm := cost_data.get("volume_metrics"):
                summary_parts.append(f"Daily Requests: {vm.get('daily_requests', 0):,}")
                summary_parts.append(f"Monthly Requests: {vm.get('monthly_requests', 0):,}")
            
            if breakdown := cost_data.get("cost_breakdown"):
                cheapest_model = min(
                    breakdown.items(),
                    key=lambda x: x[1].get("monthly_cost", float('inf')) if isinstance(x[1], dict) else float('inf')
                )
                if isinstance(cheapest_model[1], dict):
                    summary_parts.append(f"Most cost-effective: {cheapest_model[0]} at ${cheapest_model[1].get('monthly_cost', 0):,.2f}/month")
            summary_parts.append("")

        # ROI analysis
        if roi_data := analysis_data.get("roi"):
            summary_parts.extend([
                "ROI ANALYSIS:",
                "-" * 13
            ])
            if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics") or roi_data.get("financial_metrics")):
                roi_pct = metrics.get('roi_percentage', 0)
                payback = metrics.get('payback_period_years', 0)
                net_benefit = metrics.get('net_benefit', 0)
                summary_parts.extend([
                    f"ROI: {roi_pct}%",
                    f"Payback: {payback:.1f} years",
                    f"Net Benefit: ${net_benefit:,.0f}"
                ])
            summary_parts.append("")

        # Task analysis
        if tasks_data := analysis_data.get("tasks"):
            summary_parts.extend([
                "TASK ANALYSIS:",
                "-" * 14
            ])
            if isinstance(tasks_data, dict) and tasks_data.get("automated_analysis"):
                aa = tasks_data["automated_analysis"].get("automation_analysis", {})
                summary_parts.extend([
                    f"Automation Potential: {aa.get('automation_potential', 0):.0%}",
                    f"Weekly Time Saved: {aa.get('weekly_time_saved', 0):.1f} hours",
                    f"Annual Savings: ${aa.get('annual_cost_savings', 0):,.0f}"
                ])
            summary_parts.append("")

        # Infrastructure analysis
        if infra_data := analysis_data.get("infrastructure"):
            summary_parts.extend([
                "INFRASTRUCTURE OPTIMIZATION:",
                "-" * 26,
                f"Current spend: ${infra_data.get('current_spend', 0):,.0f}/month",
                f"Potential savings: ${infra_data.get('potential_savings', 0):,.0f}/month",
                f"Annual savings: ${infra_data.get('annual_savings', 0):,.0f}",
                ""
            ])

        # Next steps
        summary_parts.extend([
            "NEXT STEPS:",
            "-" * 11,
            "1. Review detailed analysis in each section",
            "2. Prioritize recommendations based on your constraints",
            "3. Begin with quick wins for immediate impact",
            "4. Plan phased implementation approach",
            "5. Set up monitoring and measurement systems",
            ""
        ])

        return "\n".join(summary_parts)

    except Exception as e:
        return f"Error generating summary: {e}\n\nBasic Info:\nAnalysis Type: {analysis.get('intent', 'unknown')}\nGenerated: {datetime.now()}"


def add_export_buttons(analysis: Dict[str, Any]):
    """FIXED: Add comprehensive export buttons without problematic Streamlit features"""
    st.markdown("### 📥 Export Analysis Results")

    if not analysis:
        st.warning("No analysis data available for export.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    col1, col2, col3 = st.columns(3)

    # CSV Export
    with col1:
        if st.button("📊 Export CSV", key="export_csv_btn", use_container_width=True):
            try:
                csv_data = safe_convert_to_csv(analysis)
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_data,
                    file_name=f"ai_analysis_{timestamp}.csv",
                    mime="text/csv",
                    key="csv_download"
                )
                st.success("✅ CSV export ready! Includes ALL analysis sections.")
            except Exception as e:
                st.error(f"CSV export failed: {e}")

    # JSON Export
    with col2:
        if st.button("📋 Export JSON", key="export_json_btn", use_container_width=True):
            try:
                # Clean the analysis data for JSON serialization
                clean_analysis = clean_for_json(analysis)
                json_data = json.dumps(clean_analysis, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON File",
                    data=json_data,
                    file_name=f"ai_analysis_{timestamp}.json",
                    mime="application/json",
                    key="json_download"
                )
                st.success("✅ JSON export ready! Complete analysis data.")
            except Exception as e:
                st.error(f"JSON export failed: {e}")

    # Summary Export
    with col3:
        if st.button("📧 Generate Summary", key="export_summary_btn", use_container_width=True):
            try:
                summary_data = safe_generate_summary(analysis)
                st.download_button(
                    label="💾 Download Summary",
                    data=summary_data,
                    file_name=f"ai_summary_{timestamp}.txt",
                    mime="text/plain",
                    key="summary_download"
                )
                st.success("✅ Summary export ready! Comprehensive overview.")
            except Exception as e:
                st.error(f"Summary export failed: {e}")

    # FIXED: Show preview without problematic expander key
    if st.checkbox("👀 Show Export Preview", key="show_preview_checkbox"):
        tab1, tab2, tab3 = st.tabs(["📊 CSV Preview", "📋 JSON Preview", "📧 Summary Preview"])
        
        with tab1:
            try:
                csv_preview = safe_convert_to_csv(analysis)
                st.text_area("CSV Data", csv_preview[:1000] + "..." if len(csv_preview) > 1000 else csv_preview, height=200, key="csv_preview")
            except Exception as e:
                st.error(f"CSV preview error: {e}")
        
        with tab2:
            try:
                clean_analysis = clean_for_json(analysis)
                json_preview = json.dumps(clean_analysis, indent=2, default=str)
                st.text_area("JSON Data", json_preview[:1000] + "..." if len(json_preview) > 1000 else json_preview, height=200, key="json_preview")
            except Exception as e:
                st.error(f"JSON preview error: {e}")
        
        with tab3:
            try:
                summary_preview = safe_generate_summary(analysis)
                st.text_area("Summary", summary_preview[:1000] + "..." if len(summary_preview) > 1000 else summary_preview, height=200, key="summary_preview")
            except Exception as e:
                st.error(f"Summary preview error: {e}")


def clean_for_json(obj: Any) -> Any:
    """Recursively clean object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return clean_for_json(obj.__dict__)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)