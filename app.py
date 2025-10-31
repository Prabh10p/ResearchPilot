import streamlit as st

st.header("Research Copilot")

# User question input
question = st.text_input("Enter your question here:")

# Research URL input
research_url = st.text_input("Enter a research URL:")

# PDF upload
uploaded_pdf = st.file_uploader("Upload your PDF here", type=["pdf"])

