"""
Fixed Chat UI components with proper form handling and text formatting
"""
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
        elif message["role"] == "user" and message.get("form_data"):
            # Display form data summary
            st.markdown("**📋 Structured Analysis Request**")
            form_data = message["form_data"]
            
            # Show key form details in a clean format
            basic = form_data.get("basic_info", {})
            current = form_data.get("current_state", {})
            goals = form_data.get("goals", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Company:** {basic.get('company_size', 'Unknown')}")
                st.info(f"**Industry:** {basic.get('industry', 'Unknown')}")
            with col2:
                st.info(f"**Use Case:** {basic.get('use_case', 'Unknown')}")
                if current.get('current_spend', 0) > 0:
                    st.info(f"**Monthly Spend:** ${current['current_spend']:,}")
            with col3:
                st.info(f"**Team Size:** {current.get('team_size', 0)} people")
                if goals.get('target_savings', 0) > 0:
                    st.info(f"**Target Savings:** {goals['target_savings']}%")
        else:
            # FIXED: Properly clean content to prevent LaTeX and formatting issues
            content = message["content"]
            # Remove all problematic formatting that causes LaTeX issues
            content = content.replace("$", "\\$")    # Escape dollar signs
            content = content.replace("**", "")      # Remove bold markdown
            content = content.replace("*", "")       # Remove italic markdown
            content = content.replace("#", "")       # Remove header markdown
            content = content.replace("`", "")       # Remove code markdown
            content = content.replace("_", "\\_")    # Escape underscores
            content = content.replace("^", "\\^")    # Escape carets
            content = content.replace("{", "\\{")    # Escape braces
            content = content.replace("}", "\\}")    # Escape braces
            st.markdown(content)


def render_chat_interface(orchestrator):
    """FIXED: Render the main chat interface with proper message display order"""
    
    # ALWAYS display chat history FIRST (before any form interface)
    if st.session_state.messages:
        st.markdown("### 💬 Analysis History")
        for message in st.session_state.messages:
            display_message(message)
        st.markdown("---")
    
    # Import and render form interface AFTER showing previous messages
    from app.components.form_ui import render_form_interface
    
    # Check if user wants to use form interface
    use_chat = render_form_interface(orchestrator)
    
    if not use_chat:
        return  # Form interface is being used, don't show chat input
    
    # Chat input section
    st.markdown("### 💬 Continue Conversation")
    
    if prompt := st.chat_input("Describe your AI use case, costs, or automation needs..."):
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        
        # Display user message immediately
        with st.chat_message("user"):
            # FIXED: Clean user input to prevent formatting issues
            clean_prompt = prompt.replace("$", "\\$").replace("**", "").replace("*", "")
            st.markdown(clean_prompt)

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
                    error_msg = f"Analysis Error: {str(e)}"
                    st.error(error_msg)
                    
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

        st.rerun()