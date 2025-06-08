"""
Fixed export functionality for analysis results
"""
import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime
import io
import base64


def safe_convert_to_csv(analysis: Dict[str, Any]) -> str:
    """Safely convert analysis to CSV format with error handling"""
    try:
        rows = [
            ["Analysis Type", analysis.get("intent", "general")],
            ["Confidence", f"{analysis.get('confidence', 0):.1%}"],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""]  # Empty row
        ]

        # Add recommendations
        if recommendations := analysis.get("recommendations"):
            rows.append(["Recommendations", ""])
            for i, rec in enumerate(recommendations, 1):
                # Clean recommendation text
                clean_rec = str(rec).replace('\n', ' ').replace('\r', ' ')
                rows.append([f"Recommendation {i}", clean_rec])
            rows.append(["", ""])

        # Add cost data if available
        analysis_data = analysis.get("analysis", {})
        if cost_data := analysis_data.get("costs"):
            rows.append(["Cost Analysis", ""])
            if breakdown := cost_data.get("cost_breakdown"):
                rows.append(["Model", "Monthly Cost", "Daily Cost", "Provider"])
                for model, costs in breakdown.items():
                    if isinstance(costs, dict) and "monthly_cost" in costs:
                        rows.append([
                            str(model),
                            f"${costs.get('monthly_cost', 0):.2f}",
                            f"${costs.get('daily_cost', 0):.2f}",
                            str(costs.get('provider', 'Unknown'))
                        ])
            rows.append(["", ""])

        # Add ROI data if available
        if roi_data := analysis_data.get("roi"):
            rows.append(["ROI Analysis", ""])
            if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics")):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if "percentage" in key:
                            rows.append([key.replace("_", " ").title(), f"{value}%"])
                        elif "cost" in key or "benefit" in key:
                            rows.append([key.replace("_", " ").title(), f"${value:,.0f}"])
                        else:
                            rows.append([key.replace("_", " ").title(), str(value)])

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


def safe_generate_summary(analysis: Dict[str, Any]) -> str:
    """Safely generate a text summary with error handling"""
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
                clean_rec = str(rec).replace('\n', ' ').replace('\r', ' ')[:200]
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
            if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics")):
                roi_pct = metrics.get('roi_percentage', 0)
                payback = metrics.get('payback_period_years', 0)
                net_benefit = metrics.get('net_benefit', 0)
                summary_parts.extend([
                    f"ROI: {roi_pct}%",
                    f"Payback: {payback:.1f} years",
                    f"Net Benefit: ${net_benefit:,.0f}"
                ])
            summary_parts.append("")

        # Next steps
        summary_parts.extend([
            "NEXT STEPS:",
            "-" * 11,
            "1. Review detailed analysis sections",
            "2. Prioritize quick wins for immediate impact",
            "3. Plan phased implementation approach",
            "4. Set up monitoring and measurement systems",
            ""
        ])

        return "\n".join(summary_parts)

    except Exception as e:
        return f"Error generating summary: {e}\n\nBasic Info:\nAnalysis Type: {analysis.get('intent', 'unknown')}\nGenerated: {datetime.now()}"


def create_download_link(data: str, filename: str, mime_type: str) -> str:
    """Create a download link for data"""
    try:
        b64 = base64.b64encode(data.encode()).decode()
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    except Exception:
        return f"Error creating download link for {filename}"


def add_export_buttons(analysis: Dict[str, Any]):
    """Add export buttons for analysis results with improved error handling"""
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
                st.success("✅ CSV export ready!")
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
                st.success("✅ JSON export ready!")
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
                st.success("✅ Summary export ready!")
            except Exception as e:
                st.error(f"Summary export failed: {e}")

    # Show preview in expander
    with st.expander("👀 Preview Export Data"):
        tab1, tab2, tab3 = st.tabs(["📊 CSV Preview", "📋 JSON Preview", "📧 Summary Preview"])
        
        with tab1:
            try:
                csv_preview = safe_convert_to_csv(analysis)
                st.text_area("CSV Data", csv_preview[:1000] + "..." if len(csv_preview) > 1000 else csv_preview, height=200)
            except Exception as e:
                st.error(f"CSV preview error: {e}")
        
        with tab2:
            try:
                clean_analysis = clean_for_json(analysis)
                json_preview = json.dumps(clean_analysis, indent=2, default=str)
                st.text_area("JSON Data", json_preview[:1000] + "..." if len(json_preview) > 1000 else json_preview, height=200)
            except Exception as e:
                st.error(f"JSON preview error: {e}")
        
        with tab3:
            try:
                summary_preview = safe_generate_summary(analysis)
                st.text_area("Summary", summary_preview[:1000] + "..." if len(summary_preview) > 1000 else summary_preview, height=200)
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