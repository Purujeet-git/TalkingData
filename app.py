import streamlit as st
import os
from utils.logger import AppLogger

# Initialize logger
logger = AppLogger.get_logger("MainApp")

def initialize_app_state():
    """Initializes the global session state variables required across all pages."""
    logger.info("Initializing global session state variables.")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "df_cleaned" not in st.session_state:
        st.session_state.df_cleaned = None
    if "data_metrics" not in st.session_state:
        st.session_state.data_metrics = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "router" not in st.session_state:
        st.session_state.router = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the Analytics Studio! Upload a CSV in the sidebar to begin.", "type": "text"}
        ]
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {
            "model_name": "gemini-3.1-flash-lite",
            "temperature": 0.0,
            "max_iterations": 4
        }

def render_landing_page():
    """Renders the beautiful landing page for the application."""
    
    st.set_page_config(
        page_title="TalkingData | Enterprise AI Analytics", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for Premium Look
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .hero-title {
            color: #00d2ff;
            background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 4rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0px;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.5rem;
            color: #a0aabf;
            margin-bottom: 50px;
        }
        .feature-box {
            background-color: #1e2129;
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #00d2ff;
            height: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_app_state()

    st.markdown('<p class="hero-title">TalkingData Enterprise</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">The next-generation Hybrid AI Data Assistant</p>', unsafe_allow_html=True)

    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🧠 Hybrid Routing Engine</h3>
            <p>Our intelligent LLM router automatically classifies your natural language queries, sending text-heavy questions to a blazing-fast local FAISS vector database, and mathematical queries to an isolated Python execution sandbox.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🧹 Automated Data Cleaning</h3>
            <p>Never worry about messy data again. Upload your CSV and watch as the system automatically removes duplicate rows, cleans entirely empty columns, and profiles the health of your dataset instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>💾 True Conversational Memory</h3>
            <p>Unlike standard chatbots, TalkingData remembers the context of your previous analytical questions, allowing for seamless follow-ups like "Now divide that by 10" or "Plot a chart of that data".</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 **To get started, expand the sidebar, enter your API key, and navigate to the 'Analytics Studio' page!**")

if __name__ == "__main__":
    try:
        logger.info("Application started. Rendering landing page.")
        render_landing_page()
    except Exception as e:
        logger.critical(f"Critical application failure: {e}", exc_info=True)
        st.error("A critical system error occurred. Please check the logs.")
