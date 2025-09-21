import streamlit as st
import pandas as pd
import os
from datetime import datetime
import base64
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Career Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom HTML for header
st.markdown("""
<div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: #4a90e2;">🧭 Career Compass</h1>
    <p>Upload your resume and get AI-powered insights to optimize your job application success.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["History"])
    
    st.header("About")
    st.write("Career Compass helps you optimize your resume for better job application success using AI analysis.")

# Main content based on selected page
if page == "History":
    st.header("Analysis History")
    
    if 'history' in st.session_state and st.session_state.history:
        # Convert history to DataFrame for better display
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
    else:
        st.info("No analysis history yet. Upload and analyze a resume to see history here.")

# Footer
st.markdown("""
<div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-top: 30px; text-align: center;">
    <p>© 2023 Career Compass | AI-Powered Resume Analysis</p>
</div>
""", unsafe_allow_html=True)