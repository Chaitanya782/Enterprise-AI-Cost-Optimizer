"""
FIXED: Enhanced main Streamlit application with all UI issues resolved
"""
import streamlit as st
from pathlib import Path
import sys
from functools import lru_cache
import traceback
from datetime import datetime

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
        'show_debug': False,
        'pending_query': None,  # For auto-submitting example queries
        'form_processing': False  # To prevent conflicts
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render optimized sidebar with status and controls"""
    with st.sidebar:
        st.markdown("## 🔧 Configuration")
        
        # Connection Status
        st.markdown("### 🔌 Connection Status")
        services = [
            ("Lyzr Studio", getattr(config, 'lyzr_api_key', None)),
            ("Google Gemini", getattr(config, 'gemini_api_key', None))
        ]

        for service, api_key in services:
            if api_key:
                st.success(f"**{service}:** ✅ Connected")
            else:
                st.error(f"**{service}:** ❌ Not configured")

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
        st.markdown("### 📊 Usage Statistics")
        query_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", query_count)
        with col2:
            st.metric("Cost", f"${st.session_state.total_cost:.4f}")

        st.divider()

        # FIXED: Example queries with proper auto-submission
        st.markdown("### 💡 Example Queries")
        example_labels = [
            "💰 Infrastructure Cost Optimization",
            "📊 LLM Cost Comparison",
            "✍️ Content Generation ROI",
            "📄 Document Processing",
            "🎧 Customer Service Analysis",
            "🤖 Cost-Effective LLM Selection"
        ]

        for i, (label, query) in enumerate(zip(example_labels, get_example_queries())):
            if st.button(label, key=f"example_{i}", help=f"Click to ask: {query[:100]}...", use_container_width=True):
                # FIXED: Directly add to messages and process
                st.session_state.messages.append({"role": "user", "content": query})
                
                # Process immediately if orchestrator is available
                if st.session_state.orchestrator:
                    try:
                        if "session_id" not in st.session_state:
                            st.session_state.session_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                        response = st.session_state.orchestrator.analyze_request(query, session_id=st.session_state.session_id)

                        assistant_msg = {
                            "role": "assistant",
                            "content": "Here's my comprehensive analysis:",
                            "analysis": response
                        }
                        st.session_state.messages.append(assistant_msg)
                        st.session_state.total_cost += 0.01
                        
                    except Exception as e:
                        error_msg = f"Analysis Error: {str(e)}"
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                
                st.rerun()

        st.divider()

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True) and st.session_state.messages:
                st.session_state.messages.clear()
                st.session_state.total_cost = 0.0
                st.rerun()

        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # Debug mode toggle
        st.session_state.show_debug = st.checkbox("🔧 Debug Mode", value=st.session_state.get("show_debug", False))


def render_welcome_screen():
    """Render welcome screen with quick start guide"""
    # Hero section with better spacing
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem; color: #1f2937;">
            🚀 Welcome to Your AI Architecture Consultant
        </h1>
        <p style="font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">
            Optimize your AI costs, maximize ROI, and streamline automation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Capabilities section with cards
    st.markdown("### 🎯 What I Can Help You With")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">💰 Cost Analysis</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">
                Compare LLM costs and optimize spending across providers
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">📊 ROI Calculation</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">
                Calculate return on investment for AI projects
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">📋 Task Automation</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">
                Identify AI automation opportunities
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">🛣️ Implementation Planning</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">
                Get step-by-step AI adoption roadmaps
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Use cases section
    st.markdown("### 🔥 Popular Use Cases")
    
    col1, col2 = st.columns(2, gap="large")

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
        LYZR_AGENT_ID_ROI=your_roi_agent_id  
        LYZR_AGENT_ID_COST=your_cost_agent_id
        LYZR_AGENT_ID_TASK=your_task_agent_id
        LYZR_USER_ID=your_email
        GEMINI_API_KEY=your_gemini_key
        ```
        
        2. **Verify API keys** - Ensure all API keys are valid and active
        3. **Check network connection** - Ensure you can access external APIs
        4. **Try refreshing** - Click "Reset App" in the sidebar
        """)


def main():
    """FIXED: Enhanced main application entry point with all issues resolved"""
    # Page configuration with better defaults
    st.set_page_config(
        page_title="Enterprise AI Cost Optimizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # FIXED: Enhanced CSS for better UI without problematic features
    st.markdown("""
    <style>
    /* Main container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Typography improvements */
    .main h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    .main h3 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Chat message improvements */
    .stChatMessage {
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Button improvements */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        padding-top: 2rem !important;
    }
    
    /* Form improvements */
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px !important;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 8px !important;
    }
    
    /* Metric improvements */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Tab improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Expander improvements */
    .streamlit-expanderHeader {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Remove excessive spacing */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Improve spacing between sections */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Better column spacing */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
    
    /* Improve divider styling */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Compact header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1>🤖 Enterprise AI Cost Optimizer</h1>
            <p style="font-size: 1.1rem; color: #6b7280; margin: 0;">
                Your AI Architecture Consultant for Cost Optimization & Automation
            </p>
        </div>
        """, unsafe_allow_html=True)

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

    # Main content with better spacing
    if not st.session_state.messages:
        render_welcome_screen()
        st.markdown("<br><br>", unsafe_allow_html=True)

    # Chat interface
    if st.session_state.orchestrator:
        render_chat_interface(st.session_state.orchestrator)
    else:
        st.warning("🔄 Please fix the configuration issues above to start using the assistant.")

    # Compact footer
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Built with ❤️ using Lyzr Studio")
    with col2:
        st.caption("100xEngineers Buildathon 2.0")
    with col3:
        status = "🟢 Ready" if st.session_state.orchestrator else "🔴 Config needed"
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