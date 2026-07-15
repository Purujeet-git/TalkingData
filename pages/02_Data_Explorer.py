import streamlit as st
import pandas as pd
from utils.logger import AppLogger
from utils.data_cleaner import DataCleaner

logger = AppLogger.get_logger("DataExplorer")

st.set_page_config(page_title="Data Explorer | TalkingData", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #00d2ff; }
    </style>
""", unsafe_allow_html=True)

st.title("Data Explorer 🔍")

# Verify Data Exists
if st.session_state.get("df_cleaned") is None:
    st.warning("⚠️ No dataset found. Please upload a CSV on the Analytics Studio page first.")
    st.stop()

df: pd.DataFrame = st.session_state.df_cleaned
logger.info("Rendering Data Explorer page.")

st.markdown("### Raw Data Viewer")
st.markdown("Use this interface to manually inspect, sort, and filter your sanitized dataset.")

# Filtering Controls
col1, col2 = st.columns(2)
with col1:
    search_term = st.text_input("Global Search (Text columns only)")
with col2:
    sort_column = st.selectbox("Sort By Column", options=["None"] + list(df.columns))

# Apply Operations
display_df = df.copy()

if search_term:
    # Filter only string columns
    str_cols = display_df.select_dtypes(include=['object', 'string']).columns
    if len(str_cols) > 0:
        mask = display_df[str_cols].apply(lambda x: x.astype(str).str.contains(search_term, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]

if sort_column != "None":
    display_df = display_df.sort_values(by=sort_column, ascending=True)

st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# Statistical Summary
st.markdown("### Statistical Summary (Numerical Data)")
try:
    cleaner = DataCleaner(df)
    summary_df = cleaner.generate_summary_statistics()
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No numerical columns available for statistical summary.")
except Exception as e:
    logger.error(f"Failed to generate summary on Data Explorer: {e}")
    st.error("Failed to generate summary statistics.")

# Export Cleaned Data
st.markdown("### Export")
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Cleaned Dataset as CSV",
    data=csv_data,
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)
