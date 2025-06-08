"""
FIXED: Enhanced form-based input component with proper processing
"""
import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class AIAnalysisForm:
    """Structured form for AI cost analysis input"""

    def __init__(self):
        self.form_data = {}

    def render_form(self) -> Optional[Dict[str, Any]]:
        """Render the comprehensive analysis form with improved design"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;">
            <h2 style="color: white; margin-bottom: 0.5rem;">📋 Structured Analysis Form</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1rem;">
                Get a comprehensive AI cost and ROI analysis tailored to your needs
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("ai_analysis_form", clear_on_submit=True):
            # Basic Information
            st.markdown("### 🏢 Basic Information")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                company_size = st.selectbox(
                    "Company Size",
                    ["Startup (1-50)", "SMB (51-200)", "Mid-market (201-1000)", "Enterprise (1000+)"],
                    help="Select your company size category"
                )

                industry = st.selectbox(
                    "Industry",
                    ["Technology", "Healthcare", "Finance", "E-commerce", "Manufacturing",
                     "Education", "Legal", "Other"],
                    help="Select your primary industry"
                )

            with col2:
                use_case = st.selectbox(
                    "Primary AI Use Case",
                    ["Customer Support", "Content Generation", "Data Analysis",
                     "Document Processing", "Code Generation", "Automation", "Other"],
                    help="What's your main AI application?"
                )

                urgency = st.selectbox(
                    "Implementation Urgency",
                    ["Immediate (1-3 months)", "Short-term (3-6 months)",
                     "Medium-term (6-12 months)", "Long-term (12+ months)"],
                    help="When do you need to implement this?"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Current State
            st.markdown("### 📊 Current State")
            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                current_spend = st.number_input(
                    "Current Monthly AI Spend ($)",
                    min_value=0,
                    value=0,
                    step=1000,
                    help="Total monthly spending on AI/LLM services"
                )

                team_size = st.number_input(
                    "Team Size",
                    min_value=1,
                    value=5,
                    step=1,
                    help="Number of people who would use AI tools"
                )

            with col2:
                daily_requests = st.number_input(
                    "Daily AI Requests",
                    min_value=0,
                    value=100,
                    step=50,
                    help="Estimated daily API calls or AI interactions"
                )

                hours_per_week = st.number_input(
                    "Manual Hours/Week",
                    min_value=0,
                    value=40,
                    step=5,
                    help="Hours spent weekly on tasks that could be automated"
                )

            with col3:
                avg_hourly_rate = st.number_input(
                    "Average Hourly Rate ($)",
                    min_value=0,
                    value=50,
                    step=5,
                    help="Average hourly cost of your team members"
                )

                error_rate = st.slider(
                    "Current Error Rate (%)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    help="Percentage of errors in current manual processes"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Technical Details
            st.markdown("### ⚙️ Technical Requirements")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                current_providers = st.multiselect(
                    "Current AI Providers",
                    ["OpenAI", "Anthropic Claude", "Google Gemini", "AWS Bedrock",
                     "Azure OpenAI", "Cohere", "Hugging Face", "None"],
                    help="Which AI providers are you currently using?"
                )

                integration_complexity = st.selectbox(
                    "Integration Complexity",
                    ["Simple (API only)", "Medium (Some custom logic)",
                     "Complex (Deep integration)", "Very Complex (Custom ML)"],
                    help="How complex will the integration be?"
                )

            with col2:
                data_sensitivity = st.selectbox(
                    "Data Sensitivity",
                    ["Public", "Internal", "Confidential", "Highly Sensitive"],
                    help="What's the sensitivity level of your data?"
                )

                compliance_requirements = st.multiselect(
                    "Compliance Requirements",
                    ["GDPR", "HIPAA", "SOC 2", "ISO 27001", "PCI DSS", "None"],
                    help="What compliance standards do you need to meet?"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Goals and Constraints
            st.markdown("### 🎯 Goals & Constraints")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                budget_range = st.selectbox(
                    "Implementation Budget",
                    ["< $10K", "$10K - $50K", "$50K - $100K", "$100K - $500K", "> $500K"],
                    help="What's your budget for AI implementation?"
                )

                target_savings = st.slider(
                    "Target Cost Reduction (%)",
                    min_value=0,
                    max_value=80,
                    value=30,
                    help="What percentage cost reduction are you targeting?"
                )

            with col2:
                roi_timeline = st.selectbox(
                    "Expected ROI Timeline",
                    ["3 months", "6 months", "1 year", "2 years", "3+ years"],
                    help="When do you expect to see ROI?"
                )

                success_metrics = st.multiselect(
                    "Key Success Metrics",
                    ["Cost Reduction", "Time Savings", "Quality Improvement",
                     "Customer Satisfaction", "Revenue Growth", "Error Reduction"],
                    help="What metrics will define success for you?"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Additional Context
            st.markdown("### 📝 Additional Context")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                specific_challenges = st.text_area(
                    "Specific Challenges",
                    placeholder="Describe any specific challenges or pain points you're facing...",
                    height=120
                )

            with col2:
                additional_requirements = st.text_area(
                    "Additional Requirements",
                    placeholder="Any other requirements, constraints, or goals...",
                    height=120
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Submit button with improved styling
            submitted = st.form_submit_button(
                "🚀 Generate Comprehensive Analysis",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                # FIXED: Properly validate and compile form data
                form_data = {
                    "basic_info": {
                        "company_size": company_size,
                        "industry": industry,
                        "use_case": use_case,
                        "urgency": urgency
                    },
                    "current_state": {
                        "current_spend": float(current_spend) if current_spend else 0,
                        "team_size": int(team_size) if team_size else 0,
                        "daily_requests": int(daily_requests) if daily_requests else 0,
                        "hours_per_week": float(hours_per_week) if hours_per_week else 0,
                        "avg_hourly_rate": float(avg_hourly_rate) if avg_hourly_rate else 0,
                        "error_rate": float(error_rate) if error_rate else 0
                    },
                    "technical": {
                        "current_providers": current_providers or [],
                        "integration_complexity": integration_complexity,
                        "data_sensitivity": data_sensitivity,
                        "compliance_requirements": compliance_requirements or []
                    },
                    "goals": {
                        "budget_range": budget_range,
                        "target_savings": float(target_savings) if target_savings else 0,
                        "roi_timeline": roi_timeline,
                        "success_metrics": success_metrics or []
                    },
                    "context": {
                        "specific_challenges": specific_challenges.strip() if specific_challenges else "",
                        "additional_requirements": additional_requirements.strip() if additional_requirements else ""
                    },
                    "timestamp": datetime.now().isoformat()
                }

                return form_data

        return None

    def convert_form_to_query(self, form_data: Dict[str, Any]) -> str:
        """Convert form data to a comprehensive query string"""
        basic = form_data.get("basic_info", {})
        current = form_data.get("current_state", {})
        technical = form_data.get("technical", {})
        goals = form_data.get("goals", {})
        context = form_data.get("context", {})

        query_parts = [
            f"Company Profile: {basic.get('company_size', 'Unknown')} {basic.get('industry', 'Unknown')} company",
            f"Primary Use Case: {basic.get('use_case', 'General AI implementation')}",
            f"Implementation Timeline: {basic.get('urgency', 'Not specified')}"
        ]

        # Add financial information
        if current.get("current_spend", 0) > 0:
            query_parts.append(f"Current monthly AI spend: ${current['current_spend']:,}")

        # Add usage information
        if current.get("daily_requests", 0) > 0:
            query_parts.append(f"Daily AI requests: {current['daily_requests']:,}")

        # Add team and cost information
        if current.get("hours_per_week", 0) > 0:
            query_parts.append(f"Manual work: {current['hours_per_week']} hours/week at ${current.get('avg_hourly_rate', 50)}/hour")

        if current.get("team_size", 0) > 0:
            query_parts.append(f"Team size: {current['team_size']} people")

        # Add technical details
        if technical.get("current_providers"):
            providers = ", ".join(technical["current_providers"])
            query_parts.append(f"Current providers: {providers}")

        # Add goals
        if goals.get("target_savings", 0) > 0:
            query_parts.append(f"Target cost reduction: {goals['target_savings']}%")

        if goals.get("budget_range"):
            query_parts.append(f"Implementation budget: {goals['budget_range']}")

        # Add context
        if context.get("specific_challenges"):
            query_parts.append(f"Challenges: {context['specific_challenges']}")

        if context.get("additional_requirements"):
            query_parts.append(f"Additional requirements: {context['additional_requirements']}")

        query_parts.append("Please provide comprehensive cost analysis, ROI projections, and implementation roadmap.")

        return ". ".join(query_parts)


def render_form_interface(orchestrator):
    """FIXED: Render the form-based interface with proper processing"""
    form_handler = AIAnalysisForm()

    # Add form toggle with better styling
    st.markdown("### 🎯 Choose Your Input Method")

    # Create columns for better layout
    col1, col2 = st.columns([3, 1])

    with col1:
        input_method = st.radio(
            "How would you like to provide information?",
            ["💬 Chat Interface", "📋 Structured Form"],
            horizontal=True,
            help="Choose between conversational chat or a structured form"
        )

    with col2:
        if st.button("ℹ️ Help", help="Get help choosing the right input method"):
            st.info("""
            **💬 Chat Interface**: Best for quick questions and conversational analysis
            
            **📋 Structured Form**: Best for comprehensive analysis with detailed requirements
            """)

    if input_method == "📋 Structured Form":
        form_data = form_handler.render_form()

        if form_data:
            # FIXED: Properly process form data without conflicts
            try:
                # Convert form data to query
                query = form_handler.convert_form_to_query(form_data)

                # Add to messages with proper formatting
                st.session_state.messages.append({
                    "role": "user",
                    "content": "Structured Analysis Request Submitted",
                    "form_data": form_data
                })

                # Process with orchestrator
                with st.spinner("🤖 Processing your comprehensive analysis request..."):
                    if "session_id" not in st.session_state:
                        st.session_state.session_id = f"form_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    response = orchestrator.analyze_request(query, session_id=st.session_state.session_id)

                    # Add assistant response
                    assistant_msg = {
                        "role": "assistant",
                        "content": "Here's your comprehensive analysis based on the form data:",
                        "analysis": response,
                        "form_based": True
                    }
                    st.session_state.messages.append(assistant_msg)

                    st.session_state.total_cost += 0.02  # Higher cost for comprehensive analysis

                    st.success("✅ Analysis completed! Scroll up to see your results.")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Form Processing Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Form Processing Error: {str(e)}"
                })
                st.rerun()

    return input_method == "💬 Chat Interface"