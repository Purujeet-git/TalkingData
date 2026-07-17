import pandas as pd
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import streamlit as st
from utils.logger import AppLogger

logger = AppLogger.get_logger("VectorStoreManager")

class VectorStoreManager:
    """
    Manages the creation, caching, and querying of the FAISS vector database.
    
    This class isolates the heavy embedding generation process. It uses 
    HuggingFace's free, local 'all-MiniLM-L6-v2' model to convert rows of 
    a pandas DataFrame into dense vector embeddings for rapid semantic search.
    
    Attributes:
        model_name (str): The HuggingFace sentence transformer model to use.
        embeddings (HuggingFaceEmbeddings): The initialized embedding model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStoreManager.
        
        Args:
            model_name (str): The name of the HuggingFace model. Defaults to 'all-MiniLM-L6-v2'.
        """
        logger.info(f"Initializing VectorStoreManager with model: {model_name}")
        self.model_name = model_name
        # The embedding model is loaded locally. First run will download it.
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    @st.cache_resource(show_spinner=False)
    def build_index(_self, file_name: str, df: pd.DataFrame, max_rows: int = 5000) -> Optional[FAISS]:
        """
        Constructs a FAISS vector index from a pandas DataFrame.
        
        This method is decorated with @st.cache_resource to ensure that the
        computationally expensive embedding process only runs once per dataset
        upload, surviving Streamlit reruns.
        
        Args:
            file_name (str): The name of the uploaded file (used implicitly in caching keys).
            df (pd.DataFrame): The cleaned DataFrame to index.
            max_rows (int): Maximum number of rows to embed to prevent memory crashes. 
                            Defaults to 5000.
                            
        Returns:
            Optional[FAISS]: The populated FAISS vector store, or None if creation fails.
        """
        logger.info(f"Building FAISS index for {file_name}. Limiting to {max_rows} rows.")
        try:
            docs = []
            # Sample the dataframe to prevent OOM errors on massive files
            sample_df = df.head(max_rows)
            
            for idx, row in sample_df.iterrows():
                # Convert the row into a readable text document format
                # Example: "Column1: Value1 | Column2: Value2"
                text_content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                docs.append(Document(page_content=text_content))
                
            logger.info(f"Generated {len(docs)} LangChain Documents. Starting embedding process...")
            
            # This is the heavy lifting step where vectors are computed
            vector_store = FAISS.from_documents(docs, _self.embeddings)
            
            logger.info("FAISS vector store successfully created.")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}", exc_info=True)
            return None

    def query_index(self, vector_store: FAISS, query: str, k: int = 5) -> str:
        """
        Performs a semantic similarity search against the FAISS index.
        
        Args:
            vector_store (FAISS): The initialized FAISS vector database.
            query (str): The user's natural language question.
            k (int): Number of most relevant rows to retrieve. Defaults to 5.
            
        Returns:
            str: A formatted string containing the retrieved context, ready for the LLM.
        """
        logger.info(f"Performing similarity search for query: '{query}' (k={k})")
        if vector_store is None:
            logger.warning("Attempted to query a None vector_store.")
            return "No vector store initialized. Cannot perform search."
            
        try:
            docs = vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            logger.debug(f"Retrieved {len(docs)} documents for context.")
            return context
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return "Error retrieving context from vector store."
