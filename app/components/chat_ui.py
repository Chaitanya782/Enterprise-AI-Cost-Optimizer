"""Chat UI components for Streamlit"""
import streamlit as st
from typing import Dict, Any
from datetime import datetime
import traceback


def display_message(message: Dict[str, str]):
    """Display a single chat message with proper formatting"""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "analysis" in message:
            from app.components.visualizations import display_enhanced_analysis
            display_enhanced_analysis(message["analysis"])
        else:
            # Fixed: Escape dollar signs to prevent LaTeX rendering issues
            content = message["content"].replace("$", r"\$")
            st.markdown(content)


def _display_cost_metrics(vm: dict):
    """Display volume metrics in columns"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Requests", f"{vm.get('daily_requests', 0):,}")
    with col2:
        st.metric("Monthly Requests", f"{vm.get('monthly_requests', 0):,}")
    with col3:
        avg_tokens = vm.get('avg_input_tokens', 0) + vm.get('avg_output_tokens', 0)
        st.metric("Avg Tokens/Request", f"{avg_tokens}")


def _display_cost_breakdown(breakdown: dict):
    """Display cost breakdown table"""
    st.write("**💵 Model Cost Comparison:**")
    
    cost_data_list = [
        {
            "Model": model,
            "Provider": costs.get("provider", "Unknown"),
            "Tier": costs.get("tier", "Unknown"),
            "Monthly Cost": f"${costs.get('monthly_cost', 0):,.2f}",
            "Daily Cost": f"${costs.get('daily_cost', 0):.2f}",
            "Per Request": f"${costs.get('cost_per_request', 0):.4f}",
            "Volume Discount": f"{costs.get('volume_discount', 0)}%"
        }
        for model, costs in breakdown.items()
        if isinstance(costs, dict) and "monthly_cost" in costs
    ]
    
    if cost_data_list:
        st.dataframe(cost_data_list, use_container_width=True)
    else:
        st.warning("No valid cost data found in breakdown")


def _display_roi_metrics(metrics: dict):
    """Display ROI metrics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROI", f"{metrics.get('roi_percentage', 0)}%")
    
    with col2:
        payback = metrics.get("payback_period_years", 0)
        if payback < 2:
            payback_months = metrics.get("payback_period_months", payback * 12)
            st.metric("Payback", f"{payback_months:.1f} months")
        else:
            st.metric("Payback", f"{payback:.1f} years")
    
    with col3:
        st.metric("Net Benefit", f"${metrics.get('net_benefit', 0):,.0f}")
    
    with col4:
        if npv := metrics.get("npv", 0):
            st.metric("NPV", f"${npv:,.0f}")


def _display_extracted_metrics(metrics: dict):
    """Display extracted metrics in organized columns"""
    cols = st.columns(min(3, len(metrics)))
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % 3]:
            key_title = key.replace("_", " ").title()
            
            if isinstance(value, (int, float)):
                if any(term in key for term in ["spend", "budget", "cost"]):
                    st.metric(key_title, f"${value:,.0f}")
                elif any(term in key for term in ["percentage", "reduction"]):
                    st.metric(key_title, f"{value:.1%}")
                else:
                    st.metric(key_title, f"{value:,}")
            else:
                st.write(f"**{key_title}**: {value}")


def display_analysis(analysis: Dict[str, Any]):
    """Display structured analysis results"""
    try:
        # Main recommendations
        if recommendations := analysis.get("recommendations"):
            st.success("### 🎯 Key Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

        # Intent and confidence
        if intent := analysis.get("intent"):
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Intent Detected**: {intent.title()}")
            with col2:
                confidence = analysis.get("confidence", 0)
                st.info(f"**Confidence**: {confidence:.1%}")

        analysis_data = analysis.get("analysis", {})

        # Cost analysis
        if cost_data := analysis_data.get("costs"):
            with st.expander("💰 Cost Analysis", expanded=True):
                if vm := cost_data.get("volume_metrics"):
                    _display_cost_metrics(vm)
                
                if breakdown := cost_data.get("cost_breakdown"):
                    _display_cost_breakdown(breakdown)
                
                for key in ["cost_summary", "analysis"]:
                    if content := cost_data.get(key):
                        st.write(f"**{'📊 Cost Summary' if key == 'cost_summary' else '🔍 Detailed Analysis'}:**")
                        st.write(content)

        # Infrastructure analysis
        if infra_data := analysis_data.get("infrastructure"):
            with st.expander("🏗️ Infrastructure Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Spend", f"${infra_data.get('current_spend', 0):,.0f}/month")
                with col2:
                    st.metric("Target Spend", f"${infra_data.get('target_spend', 0):,.0f}/month")
                with col3:
                    st.metric("Potential Savings", f"${infra_data.get('potential_savings', 0):,.0f}/month")
                
                if detailed := infra_data.get("detailed_analysis"):
                    st.write("**📋 Optimization Plan:**")
                    st.write(detailed)

        # Task analysis
        if tasks_data := analysis_data.get("tasks"):
            with st.expander("📋 Task Automation Analysis", expanded=True):
                if isinstance(tasks_data, str):
                    st.write(tasks_data)
                elif isinstance(tasks_data, dict):
                    content = (tasks_data.get("detailed_analysis") or 
                              tasks_data.get("analysis") or 
                              str(tasks_data))
                    st.write(content)

        # ROI analysis
        if roi_data := analysis_data.get("roi"):
            with st.expander("📊 ROI Analysis", expanded=True):
                if metrics := (roi_data.get("basic_metrics") or roi_data.get("key_metrics")):
                    _display_roi_metrics(metrics)
                
                if detailed := (roi_data.get("detailed_analysis") or roi_data.get("analysis")):
                    st.write("**🔍 Detailed ROI Analysis:**")
                    st.write(detailed)

        # Extracted metrics
        if metrics := analysis.get("metrics"):
            with st.expander("📈 Extracted Metrics", expanded=False):
                _display_extracted_metrics(metrics)

    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        st.write("**Raw analysis data:**")
        st.json(analysis)
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def render_chat_interface(orchestrator):
    """Render the main chat interface"""
    # Display chat history
    for message in st.session_state.messages:
        display_message(message)

    # Chat input
    if prompt := st.chat_input("Describe your AI use case, costs, or automation needs..."):
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        
        # Display user message (fixed: removed custom CSS)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing your request..."):
                try:
                    # Generate session ID if not exists
                    if "session_id" not in st.session_state:
                        st.session_state.session_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    response = orchestrator.analyze_request(prompt, session_id=st.session_state.session_id)

                    # Debug mode
                    if st.session_state.get("show_debug", False):
                        with st.expander("🔧 Debug: Response Structure", expanded=False):
                            st.json(response)

                    if not isinstance(response, dict):
                        raise ValueError(f"Expected dict response, got {type(response)}")

                    # Add assistant message and display analysis
                    assistant_msg = {
                        "role": "assistant",
                        "content": "Here's my comprehensive analysis:",
                        "analysis": response
                    }
                    st.session_state.messages.append(assistant_msg)

                    from app.components.visualizations import display_enhanced_analysis
                    display_enhanced_analysis(response)

                    st.session_state.total_cost += 0.01

                except Exception as e:
                    error_msg = f"❌ Analysis Error: {str(e)}"
                    st.error(error_msg)
                    
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

        st.rerun()
