# TalkingData (formerly DataWhisperer) 📊🔍

TalkingData is an intelligent, hybrid AI application that allows you to chat with your CSV datasets in plain English. 

It leverages a state-of-the-art **Hybrid Routing Architecture**:
- **Semantic Search Pipeline:** Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) and a lightning-fast FAISS vector database to answer text-based queries using minimal API credits.
- **Data Analysis Pipeline:** Uses a LangChain Pandas Agent with a Python execution sandbox to execute mathematical queries, aggregations, and generate charts.
- **Intelligent Router:** Automatically classifies your question and routes it to the most efficient pipeline.

## Features
- **True Conversational Memory:** Remembers the context of your chat for seamless follow-up questions.
- **Auto-Cleaning & Insights Dashboard:** Automatically removes duplicate rows and empty columns upon upload, and displays a health dashboard of your data.
- **API Cost Optimization:** In-memory caching and strict token limits prevent runaway API usage.

## Installation

Follow these steps to run TalkingData on your local machine:

### 1. Clone the repository
```bash
git clone https://github.com/Purujeet-git/TalkingData.git
cd TalkingData
```

### 2. Set up a virtual environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```
*Note: The first time you run the application, it will download a small embedding model (~80MB) for local semantic search.*

### 4. Run the application
```bash
python -m streamlit run app.py
```

### 5. Configuration
- Once the app opens in your browser (usually at `http://localhost:8501`), enter your **Google Gemini API Key** in the sidebar.
- Upload your CSV file and start chatting!

## Technologies Used
- **Streamlit:** Frontend UI
- **LangChain:** Agent framework, memory, and routing
- **FAISS:** Local vector database
- **HuggingFace Sentence Transformers:** Local embeddings
- **Pandas:** Data manipulation
- **Google Gemini API:** Core LLM functionality
