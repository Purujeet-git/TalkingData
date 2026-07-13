import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Enable caching to save API credits on repeated queries
set_llm_cache(InMemoryCache())

st.set_page_config(page_title="DataWhisperer", page_icon="💬", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #1e2129;
    }
    .stChatMessage {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    h1 {
        color: #00d2ff;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.title("DataWhisperer: Hybrid Analytics 📊🔍")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a CSV. I will automatically route math/charts to the Pandas Agent and text searches to a massive-scale FAISS Semantic Search!", "type": "text"}]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key, temperature=0)
agent = None

@st.cache_resource(show_spinner=False)
def create_vector_store(file_name, _df):
    """Creates and caches the FAISS vector database."""
    docs = []
    # Convert first 5000 rows (or all if <5000) to documents
    sample_df = _df.head(5000)
    for idx, row in sample_df.iterrows():
        text_content = " | ".join([f"{col}: {val}" for col, val in row.items()])
        docs.append(Document(page_content=text_content))
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        
        # --- Auto-Cleaning ---
        original_count = len(df_raw)
        df = df_raw.drop_duplicates()
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        duplicates_removed = original_count - len(df)
        missing_values = df.isna().sum().sum()
        
        st.sidebar.success("File uploaded and cleaned successfully!")
        
        # --- Insights Dashboard ---
        st.markdown("### 🧹 Data Insights Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Duplicates Removed", duplicates_removed)
        col4.metric("Missing Values Left", missing_values)
        st.markdown("---")
        
        with st.expander("Preview Cleaned Data"):
            st.dataframe(df.head())
            
        with st.spinner("Building local semantic search index (this runs once)..."):
            vector_store = create_vector_store(uploaded_file.name, df)
            
        # Initialize Pandas Agent with cleaned data
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            agent_type="tool-calling",
            max_iterations=4,
            number_of_head_rows=2
        )
    except Exception as e:
        st.error(f"Error reading file or initializing components: {e}")

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"])
        else:
            st.markdown(msg["content"])

def route_query(query, history_str):
    prompt = f"""
    Conversation History:
    {history_str}
    
    Analyze this new user query: "{query}"
    If the query asks to calculate math, count rows, plot a chart, or aggregate data over an entire dataset, reply EXACTLY with the word 'DATA_ANALYSIS'.
    If the query asks to find specific information, search for themes, read text reviews, or find specific rows matching a description, reply EXACTLY with the word 'SEMANTIC_SEARCH'.
    """
    res = llm.invoke(prompt)
    if "SEMANTIC_SEARCH" in res.content.upper():
        return "SEMANTIC_SEARCH"
    return "DATA_ANALYSIS"

if prompt := st.chat_input("Ask a question about your data..."):
    if agent is None:
        st.error("Please upload a CSV file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    # Build Conversation History (last 4 messages to save tokens)
                    recent_messages = [m for m in st.session_state.messages if m["type"] == "text"][-5:-1] # excluding the current prompt
                    history_str = ""
                    for m in recent_messages:
                        role = "User" if m["role"] == "user" else "Assistant"
                        history_str += f"{role}: {m['content']}\n"
                        
                    # 1. Route the query
                    route = route_query(prompt, history_str)
                    st.info(f"🚦 Routed to: **{route}** pipeline")
                    
                    if route == "SEMANTIC_SEARCH":
                        # Perform Semantic Search
                        docs = vector_store.similarity_search(prompt, k=5)
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        semantic_prompt = f"""
                        You are a helpful data assistant. Use ONLY the following rows retrieved from the dataset to answer the user's question. 
                        If the answer is not contained within these rows, politely say you don't have enough context.
                        
                        Conversation History:
                        {history_str}
                        
                        Retrieved Rows:
                        {context}
                        
                        Question: {prompt}
                        """
                        response = llm.invoke(semantic_prompt)
                        output_text = response.content
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "type": "text"})
                        
                    else:
                        # Perform Data Analysis (Pandas Agent)
                        full_prompt = f'''
                        You are a strict data analysis assistant. You must ONLY answer questions directly related to the dataset.
                        
                        Conversation History:
                        {history_str}
                        
                        User Query: {prompt}
                        
                        If you generate a chart, DO NOT use plt.show(). Instead, save the figure exactly as 'temp_plot.png' in the current working directory using plt.savefig('temp_plot.png'). 
                        '''
                        
                        response = agent.invoke(full_prompt)
                        raw_output = response["output"]
                        
                        # Parse output
                        if isinstance(raw_output, list):
                            text_parts = []
                            for item in raw_output:
                                if isinstance(item, dict) and "text" in item:
                                    text_parts.append(item["text"])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            output_text = "".join(text_parts) if text_parts else str(raw_output)
                        elif isinstance(raw_output, dict) and "text" in raw_output:
                            output_text = raw_output["text"]
                        else:
                            output_text = str(raw_output)
                        
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "type": "text"})
                        
                        if os.path.exists("temp_plot.png"):
                            st.image("temp_plot.png")
                            with open("temp_plot.png", "rb") as f:
                                img_bytes = f.read()
                            st.session_state.messages.append({"role": "assistant", "content": img_bytes, "type": "image"})
                            os.remove("temp_plot.png")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
