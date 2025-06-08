"""
FIXED Chat UI components with proper text formatting and message display
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
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message)
    
    # Chat input section
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