# """Export functionality for analysis results"""
# import streamlit as st
# import json
# import pandas as pd
# from typing import Dict, Any
# from datetime import datetime
#
#
# def add_export_buttons(analysis: Dict[str, Any]):
#     """Add export buttons for analysis results"""
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         st.download_button(
#             label="📄 Download JSON",
#             data=json.dumps(analysis, indent=2),
#             file_name=f"ai_cost_analysis_{timestamp}.json",
#             mime="application/json"
#         )
#
#     with col2:
#         # Export CSV if cost data exists
#         if cost_breakdown := analysis.get("analysis", {}).get("costs", {}).get("cost_breakdown"):
#             csv = pd.DataFrame.from_dict(cost_breakdown, orient='index').to_csv()
#             st.download_button(
#                 label="📊 Download Cost CSV",
#                 data=csv,
#                 file_name=f"ai_cost_comparison_{timestamp}.csv",
#                 mime="text/csv"
#             )
#
#     with col3:
#         st.download_button(
#             label="📝 Download Report",
#             data=generate_text_report(analysis),
#             file_name=f"ai_cost_report_{timestamp}.txt",
#             mime="text/plain"
#         )
#
#
# def generate_text_report(analysis: Dict[str, Any]) -> str:
#     """Generate a text report from analysis"""
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
#     sections = [
#         "ENTERPRISE AI COST OPTIMIZATION REPORT",
#         "=" * 50,
#         f"Generated: {timestamp}",
#         f"Query: {analysis.get('query', 'N/A')}",
#         ""
#     ]
#
#     # Recommendations
#     if recs := analysis.get("recommendations"):
#         sections.extend([
#             "KEY RECOMMENDATIONS:",
#             "-" * 30,
#             *[f"• {rec}" for rec in recs],
#             ""
#         ])
#
#     # Cost Analysis
#     if cost_data := analysis.get("analysis", {}).get("costs"):
#         sections.extend(["COST ANALYSIS:", "-" * 30])
#
#         if breakdown := cost_data.get("cost_breakdown"):
#             for model, costs in breakdown.items():
#                 sections.extend([
#                     f"\n{model}:",
#                     f"  Daily Cost: ${costs['daily_cost']:.2f}",
#                     f"  Monthly Cost: ${costs['monthly_cost']:.0f}",
#                     f"  Annual Cost: ${costs['annual_cost']:.0f}"
#                 ])
#         sections.append("")
#
#     # ROI Analysis
#     if roi_data := analysis.get("analysis", {}).get("roi"):
#         if metrics := roi_data.get("basic_metrics", {}):
#             sections.extend([
#                 "ROI ANALYSIS:",
#                 "-" * 30,
#                 f"Implementation Cost: ${metrics.get('implementation_cost', 0):,.0f}",
#                 f"Annual Benefit: ${metrics.get('annual_benefit', 0):,.0f}",
#                 f"ROI Percentage: {metrics.get('roi_percentage', 0):.1f}%",
#                 f"Payback Period: {metrics.get('payback_period_years', 0):.1f} years",
#                 ""
#             ])
#
#     return "\n".join(sections)

"""
Export functionality for analysis results
"""
import streamlit as st
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import io


def add_export_buttons(analysis: Dict[str, Any]):
    """Add export buttons for analysis results"""

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export as JSON (complete data)
        json_str = json.dumps(analysis, indent=2, default=str)
        st.download_button(
            label="📄 Download Complete JSON",
            data=json_str,
            file_name=f"ai_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        # Export as comprehensive CSV
        csv_data = generate_comprehensive_csv(analysis)
        if csv_data:
            st.download_button(
                label="📊 Download Analysis CSV",
                data=csv_data,
                file_name=f"ai_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col3:
        # Export as detailed text report
        report = generate_detailed_report(analysis)
        st.download_button(
            label="📝 Download Full Report",
            data=report,
            file_name=f"ai_cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def generate_comprehensive_csv(analysis: Dict[str, Any]) -> str:
    """Generate comprehensive CSV with all analysis data"""

    all_data = []

    # Extract cost data
    if "costs" in analysis.get("analysis", {}) and "cost_breakdown" in analysis["analysis"]["costs"]:
        cost_data = analysis["analysis"]["costs"]["cost_breakdown"]
        for model, costs in cost_data.items():
            all_data.append({
                "Category": "Cost Analysis",
                "Item": model,
                "Daily Cost": costs.get("daily_cost", 0),
                "Monthly Cost": costs.get("monthly_cost", 0),
                "Annual Cost": costs.get("annual_cost", 0),
                "Cost per Request": costs.get("cost_per_request", 0)
            })

    # Extract infrastructure data
    if "infrastructure" in analysis.get("analysis", {}):
        infra = analysis["analysis"]["infrastructure"]
        all_data.append({
            "Category": "Infrastructure",
            "Item": "Current Spend",
            "Monthly Cost": infra.get("current_spend", 0),
            "Annual Cost": infra.get("current_spend", 0) * 12
        })
        all_data.append({
            "Category": "Infrastructure",
            "Item": "Target Spend",
            "Monthly Cost": infra.get("target_spend", 0),
            "Annual Cost": infra.get("target_spend", 0) * 12
        })
        all_data.append({
            "Category": "Infrastructure",
            "Item": "Potential Savings",
            "Monthly Cost": infra.get("potential_savings", 0),
            "Annual Cost": infra.get("annual_savings", 0)
        })

    # Extract ROI data
    if "roi" in analysis.get("analysis", {}):
        roi_metrics = analysis["analysis"]["roi"].get("basic_metrics", {})
        all_data.append({
            "Category": "ROI Analysis",
            "Item": "Implementation Cost",
            "Value": roi_metrics.get("implementation_cost", 0)
        })
        all_data.append({
            "Category": "ROI Analysis",
            "Item": "Annual Benefit",
            "Value": roi_metrics.get("annual_benefit", 0)
        })
        all_data.append({
            "Category": "ROI Analysis",
            "Item": "ROI Percentage",
            "Value": f"{roi_metrics.get('roi_percentage', 0):.1f}%"
        })
        all_data.append({
            "Category": "ROI Analysis",
            "Item": "Payback Period",
            "Value": f"{roi_metrics.get('payback_period_years', 0):.1f} years"
        })

    if all_data:
        df = pd.DataFrame(all_data)
        return df.to_csv(index=False)
    return ""


def generate_detailed_report(analysis: Dict[str, Any]) -> str:
    """Generate a detailed text report from analysis"""

    report = []
    report.append("=" * 80)
    report.append("ENTERPRISE AI COST OPTIMIZATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Query: {analysis.get('query', 'N/A')}")
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    if "recommendations" in analysis:
        for i, rec in enumerate(analysis["recommendations"], 1):
            # Clean up the recommendation text
            clean_rec = rec.replace("**", "").replace("*", "")
            report.append(f"{i}. {clean_rec}")
    report.append("")

    # Infrastructure Analysis
    if "infrastructure" in analysis.get("analysis", {}):
        report.append("INFRASTRUCTURE COST ANALYSIS")
        report.append("-" * 40)
        infra = analysis["analysis"]["infrastructure"]

        report.append(f"Current Monthly Spend: ${infra.get('current_spend', 0):,.0f}")
        report.append(f"Target Monthly Spend: ${infra.get('target_spend', 0):,.0f}")
        report.append(f"Potential Monthly Savings: ${infra.get('potential_savings', 0):,.0f}")
        report.append(f"Annual Savings Opportunity: ${infra.get('annual_savings', 0):,.0f}")
        report.append(f"Reduction Percentage: {infra.get('reduction_percentage', 0):.0f}%")
        report.append("")

        if "providers" in infra:
            report.append("Providers Analyzed:")
            for provider in infra["providers"]:
                report.append(f"  • {provider}")
            report.append("")

        if "detailed_analysis" in infra:
            report.append("Detailed Analysis:")
            report.append("-" * 20)
            # Split the analysis into lines and format
            analysis_lines = infra["detailed_analysis"].split('\n')
            for line in analysis_lines:
                if line.strip():
                    report.append(line)
            report.append("")

    # Cost Analysis
    if "costs" in analysis.get("analysis", {}):
        report.append("LLM COST COMPARISON")
        report.append("-" * 40)
        cost_data = analysis["analysis"]["costs"]

        if "cost_breakdown" in cost_data:
            # Create a simple table
            report.append(f"{'Model':<20} {'Daily':<12} {'Monthly':<12} {'Annual':<12} {'Per Request':<12}")
            report.append("-" * 68)

            for model, costs in cost_data["cost_breakdown"].items():
                report.append(
                    f"{model:<20} "
                    f"${costs['daily_cost']:<11.2f} "
                    f"${costs['monthly_cost']:<11,.0f} "
                    f"${costs['annual_cost']:<11,.0f} "
                    f"${costs['cost_per_request']:<11.4f}"
                )
            report.append("")

        if "recommendations" in cost_data:
            report.append("Cost Optimization Recommendations:")
            report.append("-" * 20)
            recommendations_text = cost_data["recommendations"]
            for line in recommendations_text.split('\n'):
                if line.strip():
                    report.append(line)
            report.append("")

    # ROI Analysis
    if "roi" in analysis.get("analysis", {}):
        report.append("ROI ANALYSIS")
        report.append("-" * 40)
        roi_data = analysis["analysis"]["roi"]

        if "basic_metrics" in roi_data:
            metrics = roi_data["basic_metrics"]
            report.append(f"Implementation Cost: ${metrics.get('implementation_cost', 0):,.0f}")
            report.append(f"Annual Benefit: ${metrics.get('annual_benefit', 0):,.0f}")
            report.append(f"Net Benefit: ${metrics.get('net_benefit', 0):,.0f}")
            report.append(f"ROI Percentage: {metrics.get('roi_percentage', 0):.1f}%")
            report.append(f"Payback Period: {metrics.get('payback_period_years', 0):.1f} years")
            report.append("")

        if "detailed_analysis" in roi_data:
            report.append("Detailed ROI Analysis:")
            report.append("-" * 20)
            analysis_text = roi_data["detailed_analysis"]
            for line in analysis_text.split('\n'):
                if line.strip():
                    report.append(line)
            report.append("")

    # Task Analysis
    if "tasks" in analysis.get("analysis", {}):
        report.append("TASK AUTOMATION ANALYSIS")
        report.append("-" * 40)

        task_data = analysis["analysis"]["tasks"]
        if isinstance(task_data, dict) and "analysis" in task_data:
            task_text = task_data["analysis"]
        else:
            task_text = str(task_data)

        for line in task_text.split('\n'):
            if line.strip():
                report.append(line)
        report.append("")

    # Migration Analysis
    if "migration" in analysis.get("analysis", {}):
        report.append("MIGRATION ANALYSIS")
        report.append("-" * 40)

        migration_data = analysis["analysis"]["migration"]
        if "analysis" in migration_data:
            migration_text = migration_data["analysis"]
            for line in migration_text.split('\n'):
                if line.strip():
                    report.append(line)
        report.append("")

    # Implementation Timeline
    report.append("IMPLEMENTATION TIMELINE")
    report.append("-" * 40)
    report.append("Week 1-2: Quick Wins")
    report.append("  • Audit current usage patterns")
    report.append("  • Implement caching and optimization")
    report.append("  • Set up monitoring dashboards")
    report.append("")
    report.append("Week 3-4: Optimization Phase")
    report.append("  • Deploy request batching")
    report.append("  • Test alternative models")
    report.append("  • Implement smart routing")
    report.append("")
    report.append("Month 2: Migration")
    report.append("  • Migrate low-risk workloads")
    report.append("  • A/B test performance")
    report.append("  • Monitor quality metrics")
    report.append("")
    report.append("Month 3: Scale and Optimize")
    report.append("  • Complete migration")
    report.append("  • Document best practices")
    report.append("  • Plan next optimization cycle")
    report.append("")

    # Risk Assessment
    report.append("RISK ASSESSMENT")
    report.append("-" * 40)
    report.append("Low Risk Actions:")
    report.append("  • Implement caching")
    report.append("  • Optimize prompts")
    report.append("  • Enable batching")
    report.append("")
    report.append("Medium Risk Actions:")
    report.append("  • Switch models for simple tasks")
    report.append("  • Implement fallback strategies")
    report.append("  • Migrate non-critical workloads")
    report.append("")
    report.append("High Risk Actions:")
    report.append("  • Complete provider migration")
    report.append("  • Change core architecture")
    report.append("  • Modify production workflows")
    report.append("")

    # Metrics and KPIs
    report.append("KEY PERFORMANCE INDICATORS")
    report.append("-" * 40)
    report.append("• Cost per request")
    report.append("• Response time (latency)")
    report.append("• Error rate")
    report.append("• User satisfaction score")
    report.append("• Token efficiency")
    report.append("• Cache hit rate")
    report.append("")

    # Next Steps
    report.append("NEXT STEPS")
    report.append("-" * 40)
    report.append("1. Review and approve optimization strategy")
    report.append("2. Allocate resources for implementation")
    report.append("3. Set up monitoring and tracking")
    report.append("4. Begin with quick wins")
    report.append("5. Schedule regular review meetings")
    report.append("")

    # Footer
    report.append("=" * 80)
    report.append("End of Report")
    report.append(f"Generated by Enterprise AI Cost Optimizer")
    report.append("=" * 80)

    return "\n".join(report)


def export_to_markdown(analysis: Dict[str, Any]) -> str:
    """Export analysis as markdown format"""
    md = []

    md.append("# Enterprise AI Cost Optimization Report")
    md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"\n**Query:** {analysis.get('query', 'N/A')}\n")

    # Executive Summary
    if "recommendations" in analysis:
        md.append("## Executive Summary\n")
        for rec in analysis["recommendations"]:
            md.append(f"- {rec}")
        md.append("")

    # Add sections based on available analysis
    if "infrastructure" in analysis.get("analysis", {}):
        md.append("## Infrastructure Analysis\n")
        infra = analysis["analysis"]["infrastructure"]
        md.append(f"- **Current Spend:** ${infra.get('current_spend', 0):,.0f}/month")
        md.append(f"- **Target Spend:** ${infra.get('target_spend', 0):,.0f}/month")
        md.append(f"- **Potential Savings:** ${infra.get('potential_savings', 0):,.0f}/month")
        md.append("")

    return "\n".join(md)