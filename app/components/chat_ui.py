"""Chat UI components for Streamlit"""
import streamlit as st
from typing import Dict, Any
from datetime import datetime
from app.components.visualizations import display_enhanced_analysis


def display_message(message: Dict[str, str]):
    """Display a single chat message"""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "analysis" in message:
            display_enhanced_analysis(message["analysis"])
        else:
            st.write(message["content"])


def display_analysis(analysis: Dict[str, Any]):
    """Display structured analysis results"""

    # Main recommendations
    if "recommendations" in analysis:
        st.info("### 🎯 Key Recommendations")
        for rec in analysis["recommendations"]:
            st.write(rec)

    # Cost analysis
    if cost_data := analysis.get("analysis", {}).get("costs"):
        with st.expander("💰 Detailed Cost Analysis", expanded=True):
            # Cost breakdown table
            if breakdown := cost_data.get("cost_breakdown"):
                st.write("**Monthly Cost Comparison:**")
                cost_df = [{
                    "Model": model,
                    "Daily Cost": f"${costs['daily_cost']:.2f}",
                    "Monthly Cost": f"${costs['monthly_cost']:.2f}",
                    "Annual Cost": f"${costs['annual_cost']:.2f}",
                    "Per Request": f"${costs['cost_per_request']:.4f}"
                } for model, costs in breakdown.items()]
                st.table(cost_df)

            # Recommendations
            if recs := cost_data.get("recommendations"):
                st.write("**Detailed Recommendations:**")
                st.write(recs)

    # Task analysis
    if tasks := analysis.get("analysis", {}).get("tasks"):
        with st.expander("📋 Task Automation Analysis"):
            st.write(tasks)

    # ROI analysis
    if roi_data := analysis.get("analysis", {}).get("roi"):
        with st.expander("📊 ROI Analysis"):
            if metrics := roi_data.get("basic_metrics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROI", f"{metrics['roi_percentage']}%")
                with col2:
                    st.metric("Payback Period", f"{metrics['payback_period_years']} years")
                with col3:
                    st.metric("Net Benefit", f"${metrics['net_benefit']:,.0f}")

            if detailed := roi_data.get("detailed_analysis"):
                st.write("**Detailed Analysis:**")
                st.write(detailed)


def render_chat_interface(orchestrator):
    """Render the main chat interface"""

    # Display chat history
    for message in st.session_state.messages:
        display_message(message)

    # Chat input
    if prompt := st.chat_input("Ask about AI cost optimization..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                try:
                    # Generate session ID if not exists
                    if "session_id" not in st.session_state:
                        st.session_state.session_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Get analysis from orchestrator
                    response = orchestrator.analyze_request(prompt, session_id=st.session_state.session_id)

                    # Add assistant message and display
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here's my analysis:",
                        "analysis": response
                    })

                    display_enhanced_analysis(response)
                    st.session_state.total_cost += 0.01

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            st.rerun()