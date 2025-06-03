"""
Optimized main Streamlit application for Enterprise AI Cost Optimizer
"""
import streamlit as st
from pathlib import Path
import sys
from functools import lru_cache
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from utils.logger import logger
from agents.orchestrator import Orchestrator
from app.components.chat_ui import render_chat_interface


@lru_cache(maxsize=1)
def get_example_queries():
    """Get cached example queries"""
    return [
        "We're spending $85,000/month on AI infrastructure (OpenAI and Anthropic). Our support team of 12 handles 2,500 tickets daily (15 min each). We want to reduce costs by 40% and automate processes. ROI analysis for $200K investment?",
        "Compare costs of GPT-4 vs Claude vs Gemini for 10,000 daily requests with 200 input and 300 output tokens",
        "Calculate ROI for automating our content team that creates 50 blog posts monthly, taking 4 hours each",
        "We process 300 contracts monthly, 2 hours each for review. What automation opportunities exist?",
        "Analyze our customer service workflow for AI automation opportunities",
        "What's the most cost-effective LLM for high-volume customer support chatbot?"
    ]


def initialize_session_state():
    """Initialize Streamlit session state variables efficiently"""
    defaults = {
        'messages': [],
        'total_cost': 0.0,
        'orchestrator': None,
        'session_id': None,
        'initialization_error': None,
        'process_example_query': False,  # Flag to trigger example query processing
        'pending_query': None  # Store the pending query
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _check_service_status(service_name: str, api_key: str) -> tuple[str, bool]:
    """Check if a service is configured"""
    if api_key:
        return f"**{service_name}:** ✅ Connected", True
    return f"**{service_name}:** ❌ Not configured", False


def render_sidebar():
    """Render optimized sidebar with status and controls"""
    with st.sidebar:
        st.header("🔧 Configuration")
        st.subheader("🔌 Connection Status")

        # Check services
        services = [
            ("Lyzr Studio", getattr(config, 'lyzr_api_key', None)),
            ("Google Gemini", getattr(config, 'gemini_api_key', None))
        ]

        for service, api_key in services:
            status, is_connected = _check_service_status(service, api_key)
            if is_connected:
                st.success(status)
            else:
                st.error(status)

        # Orchestrator status
        if st.session_state.orchestrator:
            st.success("**Orchestrator:** ✅ Initialized")
        elif st.session_state.initialization_error:
            st.error("**Orchestrator:** ❌ Failed to initialize")
            with st.expander("Error Details"):
                st.code(st.session_state.initialization_error)
        else:
            st.warning("**Orchestrator:** ⏳ Not initialized")

        st.divider()

        # Usage statistics
        st.subheader("📊 Usage Statistics")
        query_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        st.metric("Total Queries", query_count)
        st.metric("Estimated Cost", f"${st.session_state.total_cost:.4f}")

        st.divider()

        # Example queries
        st.subheader("💡 Try These Examples")

        example_labels = [
            "💰 Infrastructure Cost Optimization ($85K/month)",
            "📊 LLM Cost Comparison Analysis",
            "✍️ Content Generation ROI Analysis",
            "📄 Document Processing Automation",
            "🎧 Customer Service Workflow Analysis",
            "🤖 Cost-Effective LLM Selection"
        ]

        for i, (label, query) in enumerate(zip(example_labels, get_example_queries())):
            if st.button(label, key=f"example_{i}", help=f"Click to ask: {query[:100]}..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query})
                # Set flags to trigger processing
                st.session_state.process_example_query = True
                st.session_state.pending_query = query
                st.rerun()

        st.divider()

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat") and st.session_state.messages:
                st.session_state.messages.clear()
                st.session_state.total_cost = 0.0
                st.session_state.process_example_query = False
                st.session_state.pending_query = None
                st.rerun()

        with col2:
            if st.button("🔄 Reset App"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # Debug mode toggle
        st.session_state.show_debug = st.checkbox("🔧 Debug Mode",
                                                 value=st.session_state.get("show_debug", False))


def render_welcome_screen():
    """Render welcome screen with quick start guide"""
    st.markdown("### 🚀 Welcome to Your AI Architecture Consultant")

    st.info(
        "🎯 **I can help you with:**\n\n"
        "• **💰 Cost Analysis**: Compare LLM costs and optimize spending\n"
        "• **📋 Task Automation**: Identify AI automation opportunities\n"
        "• **📊 ROI Calculation**: Calculate return on investment for AI projects\n"
        "• **🛣️ Implementation Planning**: Get step-by-step AI adoption roadmaps\n\n"
        "💬 **Just describe your situation or ask a question to get started!**"
    )

    st.markdown("### 🔥 Popular Use Cases")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🏢 For Enterprises:**
        - Analyze customer service automation ROI
        - Compare LLM providers for large-scale deployment
        - Calculate infrastructure cost optimizations
        - Plan phased AI implementation roadmaps
        """)

    with col2:
        st.markdown("""
        **💡 For Specific Projects:**
        - Document processing automation
        - Content generation workflow optimization
        - Customer support chatbot implementation
        - Data analysis and reporting automation
        """)

    st.markdown("### 📝 Sample Questions You Can Ask")
    sample_questions = [
        "We spend $50K/month on OpenAI APIs. How can we reduce costs by 30%?",
        "ROI analysis for automating our customer support with 1000 daily tickets",
        "Compare GPT-4 vs Claude for content generation with 500 articles/month",
        "What tasks should we automate first in our sales process?"
    ]

    for question in sample_questions:
        st.markdown(f"• *{question}*")


@st.cache_resource
def initialize_orchestrator():
    """Initialize orchestrator with caching and detailed error handling"""
    try:
        logger.info("Initializing orchestrator...")
        orchestrator = Orchestrator()
        logger.info("Orchestrator initialized successfully")
        return orchestrator, None
    except Exception as e:
        error_msg = f"Failed to initialize orchestrator: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, traceback.format_exc()


def render_configuration_help():
    """Render configuration error help"""
    st.error("⚠️ **Configuration Error**: AI agents could not be initialized.")

    with st.expander("🔍 Error Details & Configuration Help"):
        st.code(st.session_state.initialization_error)

        st.markdown("""
        **Common solutions:**
        
        1. **Check your `.env` file** - Make sure it contains:
        ```
        LYZR_API_KEY=your_lyzr_api_key
        LYZR_AGENT_ID=your_agent_id  
        LYZR_USER_ID=your_email
        GEMINI_API_KEY=your_gemini_key
        ```
        
        2. **Verify API keys** - Ensure all API keys are valid and active
        3. **Check network connection** - Ensure you can access external APIs
        4. **Try refreshing** - Click "Reset App" in the sidebar
        """)


def process_example_query_if_needed(orchestrator):
    """Process example query if flagged"""
    if st.session_state.get('process_example_query', False) and st.session_state.get('pending_query'):
        # Reset the flags
        st.session_state.process_example_query = False
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        
        # Process the query automatically
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your request..."):
                try:
                    response = orchestrator.process_query(query)
                    st.write(response)
                    # Add assistant response to messages
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Update cost if available
                    if hasattr(orchestrator, 'get_session_cost'):
                        st.session_state.total_cost += orchestrator.get_session_cost()
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Error processing example query: {e}", exc_info=True)


def main():
    """Optimized main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Enterprise AI Cost Optimizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # FIXED: Minimal CSS that respects Streamlit themes
    st.markdown("""
    <style>
    /* Only essential chat improvements that work with both themes */
    .stChatMessage {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    .stButton > button {
        width: 100%;
        text-align: left;
        white-space: normal !important;
        word-wrap: break-word !important;
        height: auto;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Header
    st.title("🤖 Enterprise AI Cost Optimizer")
    st.markdown("### *Your AI Architecture Consultant for Cost Optimization & Automation*")

    # Initialize orchestrator if needed
    if not st.session_state.orchestrator and not st.session_state.initialization_error:
        with st.spinner("🤖 Initializing AI agents..."):
            orchestrator, error = initialize_orchestrator()
            if orchestrator:
                st.session_state.orchestrator = orchestrator
                st.success("✅ AI agents initialized successfully!")
            else:
                st.session_state.initialization_error = error
                st.error("❌ Failed to initialize AI agents")

    # Handle configuration errors
    if st.session_state.initialization_error:
        render_configuration_help()
        render_sidebar()
        return

    render_sidebar()

    # Main content
    if not st.session_state.messages:
        render_welcome_screen()
    else:
        st.markdown("---")

    # Chat interface
    if st.session_state.orchestrator:
        # Process example query if needed (before rendering chat)
        process_example_query_if_needed(st.session_state.orchestrator)
        render_chat_interface(st.session_state.orchestrator)
    else:
        st.warning("🔄 Please fix the configuration issues above to start using the assistant.")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Built with ❤️ using Lyzr Studio")
    with col2:
        st.caption("100xEngineers Buildathon 2.0")
    with col3:
        status = "🟢 Ready for analysis" if st.session_state.orchestrator else "🔴 Configuration needed"
        st.caption(status)


if __name__ == "__main__":
    try:
        logger.info("Starting Enterprise AI Cost Optimizer")
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"💥 **Application Error**: {e}")

        with st.expander("🔍 Technical Details"):
            st.code(traceback.format_exc())

        st.markdown("**Please try refreshing the page or contact support if the issue persists.**")
