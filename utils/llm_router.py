import os
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from utils.logger import AppLogger

logger = AppLogger.get_logger("LLMRouter")

class HybridRouter:
    """
    Intelligent Router for the TalkingData Application.
    
    This class acts as the brain of the application. It receives a user query,
    analyzes the intent, and decides whether to route the request to the
    Pandas Code Execution Agent (for math/charts) or the FAISS Semantic Search
    Pipeline (for text-based context retrieval).
    
    Attributes:
        llm (ChatGoogleGenerativeAI): The underlying Gemini model instance.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite", temperature: float = 0.0):
        """
        Initializes the Hybrid Router with the specified LLM.
        
        Args:
            api_key (str): The Google Gemini API key.
            model_name (str): The specific Gemini model version to use.
            temperature (float): The creativity metric (0.0 for deterministic).
        """
        logger.info(f"Initializing HybridRouter with model: {model_name}")
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=api_key, 
                temperature=temperature
            )
        except Exception as e:
            logger.critical(f"Failed to initialize LLM: {e}")
            raise

    def classify_intent(self, query: str, history_str: str) -> str:
        """
        Determines the correct pipeline for the user's query.
        
        Args:
            query (str): The user's input question.
            history_str (str): The formatted conversation history.
            
        Returns:
            str: Either 'DATA_ANALYSIS' or 'SEMANTIC_SEARCH'.
        """
        logger.info("Classifying query intent...")
        
        classification_prompt = f"""
        Conversation History:
        {history_str}
        
        Analyze this new user query: "{query}"
        
        Rules for classification:
        1. If the query asks to calculate math, count rows, plot a chart, aggregate data over an entire dataset, or filter numeric data, reply EXACTLY with the word 'DATA_ANALYSIS'.
        2. If the query asks to find specific text information, search for qualitative themes, read text reviews, or find specific rows matching a qualitative description, reply EXACTLY with the word 'SEMANTIC_SEARCH'.
        
        Output only the exact word, nothing else.
        """
        
        try:
            response = self.llm.invoke(classification_prompt)
            decision = response.content.strip().upper()
            
            if "SEMANTIC_SEARCH" in decision:
                logger.info("Routing decision: SEMANTIC_SEARCH")
                return "SEMANTIC_SEARCH"
            
            logger.info("Routing decision: DATA_ANALYSIS")
            return "DATA_ANALYSIS"
            
        except Exception as e:
            logger.error(f"Error during intent classification: {e}. Defaulting to DATA_ANALYSIS.")
            # Default fallback is Data Analysis as it has full data access
            return "DATA_ANALYSIS"

    def execute_semantic_search(self, query: str, context: str, history_str: str) -> str:
        """
        Executes a prompt against the retrieved FAISS context.
        
        Args:
            query (str): The user's question.
            context (str): The rows retrieved from FAISS.
            history_str (str): The conversation history.
            
        Returns:
            str: The LLM's formatted answer.
        """
        logger.info("Executing Semantic Search pipeline...")
        semantic_prompt = f"""
        You are a highly intelligent data assistant. Use ONLY the following rows retrieved from the dataset to answer the user's question. 
        If the answer is not explicitly contained within these rows, politely say you don't have enough context.
        
        Conversation History:
        {history_str}
        
        Retrieved Rows:
        {context}
        
        User Question: {query}
        """
        
        try:
            response = self.llm.invoke(semantic_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Failed to execute semantic search: {e}")
            return "I encountered an error while trying to generate the answer."

    def get_pandas_agent(self, df: pd.DataFrame, max_iterations: int = 4):
        """
        Initializes and returns the LangChain Pandas DataFrame Agent.
        
        Args:
            df (pd.DataFrame): The dataframe to bind to the agent.
            max_iterations (int): Token-saving hard limit on loops.
            
        Returns:
            AgentExecutor: The configured Pandas Agent.
        """
        logger.info("Initializing Pandas Agent for Data Analysis pipeline...")
        try:
            agent = create_pandas_dataframe_agent(
                self.llm, 
                df, 
                verbose=True, 
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                agent_type="tool-calling",
                max_iterations=max_iterations,
                number_of_head_rows=2 # Sent specifically to save prompt tokens
            )
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize Pandas Agent: {e}")
            return None

    def execute_pandas_agent(self, agent, query: str, history_str: str) -> Dict[str, Any]:
        """
        Executes the user's query through the Pandas sandbox.
        
        Args:
            agent: The initialized Pandas Agent.
            query (str): The user's query.
            history_str (str): The conversation history.
            
        Returns:
            Dict[str, Any]: A dictionary containing the text 'output' and boolean 'has_plot'.
        """
        logger.info("Executing Pandas Agent pipeline...")
        
        full_prompt = f'''
        You are a strict, expert data analysis assistant. You must ONLY answer questions directly related to the dataset provided to you.
        
        Conversation History:
        {history_str}
        
        User Query: {query}
        
        CRITICAL INSTRUCTION: If you generate a matplotlib or seaborn chart, DO NOT use plt.show(). 
        Instead, you MUST save the figure exactly as 'temp_plot.png' in the current working directory using plt.savefig('temp_plot.png'). 
        If you create a table, use pandas to output it nicely.
        '''
        
        result = {"output": "Error processing request.", "has_plot": False}
        
        try:
            response = agent.invoke(full_prompt)
            raw_output = response.get("output", "")
            
            # Parse output logic for tool-calling models (like Gemini)
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
                
            result["output"] = output_text
            
            # Check for generated plot
            if os.path.exists("temp_plot.png"):
                logger.info("Detected generated plot artifact.")
                result["has_plot"] = True
                
        except Exception as e:
            logger.error(f"Error during pandas agent execution: {e}", exc_info=True)
            result["output"] = f"An error occurred while executing data analysis: {e}"
            
        return result
