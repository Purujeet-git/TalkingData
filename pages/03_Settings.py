import streamlit as st
from utils.logger import AppLogger

logger = AppLogger.get_logger("Settings")

st.set_page_config(page_title="Settings | TalkingData", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #00d2ff; }
    .stSlider > div > div > div > div { background-color: #00d2ff; }
    </style>
""", unsafe_allow_html=True)

st.title("System Settings ⚙️")
st.markdown("Fine-tune the behavior of the TalkingData Hybrid AI engine.")

if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "model_name": "gemini-3.1-flash-lite",
        "temperature": 0.0,
        "max_iterations": 4
    }

settings = st.session_state.model_settings

st.markdown("### Language Model Configuration")

model_choice = st.selectbox(
    "Select Gemini Model Version",
    options=["gemini-3.1-flash-lite", "gemini-3.1-pro"],
    index=0 if settings["model_name"] == "gemini-3.1-flash-lite" else 1,
    help="Flash is faster and cheaper. Pro is better for complex mathematical reasoning."
)

temperature = st.slider(
    "Model Creativity (Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=settings["temperature"],
    step=0.1,
    help="0.0 is deterministic and highly factual (Recommended for data analysis). 1.0 is highly creative."
)

max_iterations = st.slider(
    "Pandas Agent Max Iterations",
    min_value=2,
    max_value=10,
    value=settings["max_iterations"],
    step=1,
    help="The maximum number of times the agent can retry failed code before giving up. Higher values consume more API tokens."
)

if st.button("Save Settings"):
    st.session_state.model_settings["model_name"] = model_choice
    st.session_state.model_settings["temperature"] = temperature
    st.session_state.model_settings["max_iterations"] = max_iterations
    
    # Invalidate the current router so it gets re-initialized with new settings on the Analytics page
    st.session_state.router = None 
    
    logger.info(f"Settings updated: {st.session_state.model_settings}")
    st.success("Settings saved successfully! They will apply to your next query.")
    
st.markdown("---")
st.markdown("### System Logs")
st.markdown("For administrative debugging and university project auditing.")
if st.button("Download System Logs"):
    import os
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = f"logs/talkingdata_{current_date}.log"
    
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            log_data = f.read()
        st.download_button("Download Logs", log_data, "system_logs.txt", "text/plain")
    else:
        st.error("Log file for today not found.")
