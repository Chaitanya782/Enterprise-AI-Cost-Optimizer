"""
Optimized main Streamlit application for Enterprise AI Cost Optimizer
"""
import streamlit as st
from pathlib import Path
import sys
from functools import lru_cache

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
        "We handle 500 support tickets daily, each taking 15 minutes. How can AI help?",
        "Compare costs of GPT-4 vs Gemini for 10,000 daily requests",
        "Calculate ROI for $50,000 AI chatbot implementation",
        "What tasks should we automate first in our customer service department?"
    ]


def initialize_session_state():
    """Initialize Streamlit session state variables efficiently"""
    defaults = {
        'messages': [],
        'total_cost': 0.0,
        'orchestrator': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render optimized sidebar with status and controls"""
    with st.sidebar:
        st.header("🔧 Configuration")

        # Connection status
        st.subheader("Connection Status")
        statuses = [
            ("Lyzr Studio", "✅ Connected" if config.lyzr_api_key else "❌ Not configured"),
            ("Google Gemini", "✅ Connected" if config.gemini_api_key else "❌ Not configured")
        ]

        for service, status in statuses:
            st.write(f"**{service}:** {status}")

        st.divider()

        # Usage statistics
        st.subheader("📊 Usage Statistics")
        query_count = len(st.session_state.messages) // 2
        st.metric("Total Queries", query_count)
        st.metric("Estimated Cost", f"${st.session_state.total_cost:.4f}")

        st.divider()

        # Example queries with optimized buttons
        st.subheader("💡 Example Queries")
        for i, query in enumerate(get_example_queries()):
            if st.button(f"{query[:50]}...", key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat") and st.session_state.messages:
            st.session_state.messages.clear()
            st.session_state.total_cost = 0.0
            st.rerun()


def render_welcome_screen():
    """Render welcome screen with quick start guide"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.info(
            "🎯 **How I can help you:**\n\n"
            "• **Task Analysis**: Identify AI automation opportunities\n"
            "• **Cost Calculation**: Compare LLM costs and find optimal options\n"
            "• **ROI Estimation**: Calculate return on investment for AI projects\n"
            "• **Implementation Roadmap**: Get step-by-step AI adoption plans\n\n"
            "💬 Just describe your use case or ask a question to get started!"
        )

        st.markdown("### 🚀 Quick Start")

        # Quick start cards
        cards = [
            ("🏢 For Enterprises", [
                "Analyze customer service automation",
                "Calculate AI implementation costs",
                "Compare different LLM providers",
                "Estimate ROI and payback period"
            ]),
            ("💰 Cost Optimization", [
                "Find the most cost-effective LLM",
                "Reduce token usage strategies",
                "Optimize prompt engineering",
                "Scale efficiently with growth"
            ])
        ]

        cols = st.columns(2)
        for col, (title, items) in zip(cols, cards):
            with col:
                st.markdown(f"**{title}:**")
                for item in items:
                    st.markdown(f"- {item}")


@st.cache_resource
def initialize_orchestrator():
    """Initialize orchestrator with caching"""
    try:
        return Orchestrator()
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise


def main():
    """Optimized main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title=config.app_title,
        page_icon=config.app_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Header
    st.title(config.app_title)
    st.markdown("### Your AI Architecture Consultant for Cost Optimization")

    # Configuration validation
    if not config.is_valid:
        st.error("⚠️ Configuration errors detected. Please check your .env file.")
        with st.expander("Configuration Help"):
            st.markdown("""
            **Required environment variables:**
            ```
            LYZR_API_KEY=your_lyzr_api_key
            LYZR_AGENT_ID=your_agent_id  
            LYZR_USER_ID=your_email
            GEMINI_API_KEY=your_gemini_key
            ```
            """)
        st.stop()

    # Initialize orchestrator with error handling
    if not st.session_state.orchestrator:
        with st.spinner("🤖 Initializing AI agents..."):
            try:
                st.session_state.orchestrator = initialize_orchestrator()
                logger.info("Orchestrator initialized successfully")
            except Exception as e:
                st.error(f"❌ Failed to initialize agents: {e}")
                st.stop()

    # Render sidebar
    render_sidebar()

    # Main content
    if not st.session_state.messages:
        render_welcome_screen()

    # Chat interface
    st.divider()
    render_chat_interface(st.session_state.orchestrator)

    # Footer
    st.divider()
    st.caption("Built with ❤️ using Lyzr Studio | 100xEngineers Buildathon 2.0")


if __name__ == "__main__":
    try:
        logger.info("Starting Enterprise AI Cost Optimizer")
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"💥 Application error: {e}")