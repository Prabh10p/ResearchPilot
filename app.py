# app.py
import streamlit as st
from langchain_unstructured import UnstructuredLoader
from src.main import build_and_run_graph

st.set_page_config(page_title="Research Copilot", layout="centered")
st.header("🧠 Research Copilot")

# User input
question = st.selectbox("What would you like to generate?", 
                        ["Generate Report", "Generate Insights", "Generate Summary", "Generate Quiz"])
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if st.button("Run"):
    if uploaded_pdf is not None:
        with st.spinner("Extracting text and running workflow..."):
            loader = UnstructuredLoader(uploaded_pdf)
            docs = loader.load()
            pdf_text = " ".join([d.page_content for d in docs])

            initial_state = {"user_input": question, "user_data": pdf_text}
            response = build_and_run_graph(initial_state)

        st.success("✅ Process complete!")
        st.subheader("Output")
        st.write(response)
    else:
        st.warning("Please upload a PDF first.")
