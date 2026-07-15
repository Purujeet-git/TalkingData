import streamlit as st
import pandas as pd
import os
import json
from utils.logger import AppLogger
from utils.data_cleaner import DataCleaner
from utils.vector_store import VectorStoreManager
from utils.llm_router import HybridRouter

logger = AppLogger.get_logger("AnalyticsStudio")

st.set_page_config(page_title="Analytics Studio | TalkingData", page_icon="💬", layout="wide")

# CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stChatMessage { background-color: #262730; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("Analytics Studio 💬")

# Sidebar Configuration and Upload
with st.sidebar:
    st.header("Configuration & Data")
    api_key = st.text_input("Google Gemini API Key", type="password", value=st.session_state.get("api_key", ""))
    
    if api_key != st.session_state.get("api_key", ""):
        st.session_state.api_key = api_key
        st.session_state.router = None # Reset router if key changes
        
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type=['csv'])

# Core validation
if not st.session_state.get("api_key", ""):
    st.warning("👈 Please enter your Gemini API Key in the sidebar to proceed.")
    st.stop()

# Initialize Router if needed
if st.session_state.get("router") is None:
    try:
        settings = st.session_state.get("model_settings", {})
        st.session_state.router = HybridRouter(
            api_key=st.session_state.api_key,
            model_name=settings.get("model_name", "gemini-3.1-flash-lite"),
            temperature=settings.get("temperature", 0.0)
        )
        logger.info("HybridRouter initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize AI: {e}")
        st.stop()

# Handle File Upload & Pipeline Execution
if uploaded_file is not None and st.session_state.get("df_cleaned") is None:
    try:
        with st.spinner("Executing Data Processing Pipeline..."):
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            
            # 1. Cleaning
            df_raw = pd.read_csv(uploaded_file)
            cleaner = DataCleaner(df_raw)
            df_clean, metrics = cleaner.execute_cleaning_pipeline()
            
            st.session_state.df_cleaned = df_clean
            st.session_state.data_metrics = metrics
            
            # 2. Vector Indexing
            vsm = VectorStoreManager()
            st.session_state.vector_store = vsm.build_index(uploaded_file.name, df_clean)
            
            st.sidebar.success("Pipeline execution complete!")
            st.rerun() # Refresh to show dashboard
            
    except Exception as e:
        logger.error(f"Pipeline failure: {e}", exc_info=True)
        st.error(f"An error occurred while processing the file: {e}")

# Render Insights Dashboard
if st.session_state.get("data_metrics") is not None:
    metrics = st.session_state.data_metrics
    st.markdown("### 🧹 Data Insights Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Usable Rows", metrics["cleaned_rows"])
    col2.metric("Total Columns", metrics["cleaned_columns"])
    col3.metric("Duplicates Purged", metrics["duplicates_removed"])
    col4.metric("Memory Usage (MB)", metrics["memory_usage_mb"])
    st.markdown("---")

# Render Chat History
messages = st.session_state.get("messages", [])
for msg in messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"])
        else:
            st.markdown(msg["content"])

# Chat Input & Routing Logic
if prompt := st.chat_input("Ask a question about your dataset..."):
    if st.session_state.get("df_cleaned") is None:
        st.error("Please upload a CSV file first.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing request architecture..."):
                try:
                    router: HybridRouter = st.session_state.router
                    
                    # Construct Memory History
                    recent_msgs = [m for m in st.session_state.messages if m["type"] == "text"][-5:-1]
                    history_str = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent_msgs])
                    
                    # Route Query
                    route = router.classify_intent(prompt, history_str)
                    st.info(f"🚦 Routed via **{route}** pipeline")
                    
                    if route == "SEMANTIC_SEARCH":
                        # Execute RAG
                        vsm = VectorStoreManager()
                        context = vsm.query_index(st.session_state.vector_store, prompt)
                        output_text = router.execute_semantic_search(prompt, context, history_str)
                        
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "type": "text"})
                        
                    else:
                        # Execute Code Sandbox
                        settings = st.session_state.get("model_settings", {})
                        agent = router.get_pandas_agent(
                            st.session_state.df_cleaned, 
                            max_iterations=settings.get("max_iterations", 4)
                        )
                        
                        result = router.execute_pandas_agent(agent, prompt, history_str)
                        output_text = result["output"]
                        
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "type": "text"})
                        
                        # Handle Images
                        if result["has_plot"]:
                            st.image("temp_plot.png")
                            with open("temp_plot.png", "rb") as f:
                                img_bytes = f.read()
                            st.session_state.messages.append({"role": "assistant", "content": img_bytes, "type": "image"})
                            os.remove("temp_plot.png")
                            
                except Exception as e:
                    logger.error(f"Chat processing error: {e}", exc_info=True)
                    st.error(f"A critical error occurred: {e}")

# Export Options
if st.session_state.get("messages") and len(st.session_state.messages) > 1:
    with st.sidebar.expander("Export Chat History"):
        chat_json = json.dumps([m for m in st.session_state.messages if m["type"]=="text"], indent=4)
        st.download_button("Download as JSON", chat_json, "chat_history.json", "application/json")
