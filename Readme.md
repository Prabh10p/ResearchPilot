# 🧠 ResearchPilot

### Your AI-Powered Research Copilot

![ResearchPilot Logo](Artifacts/ChatGPT Image Oct 26, 2025 at 10_00_51 PM.png)

**ResearchPilot** is an AI-driven assistant built with **LangGraph** and **Hugging Face**, designed to help students, data scientists, and researchers understand academic papers faster.  
It reads PDFs, analyzes research URLs, and even searches for related studies — summarizing and extracting key insights with citation-aware responses.

---

## 🚀 Features

✅ **Multi-Source Input**
- Upload PDFs
- Paste research URLs
- Search for related papers via APIs (e.g., arXiv, Semantic Scholar)

✅ **Intelligent Orchestration**
- Built with **LangGraph** to manage multi-agent workflows
- Summarization, insight extraction, and question answering nodes

✅ **Real-World AI Models**
- Powered by **Hugging Face** models (e.g., Mistral, Llama 3, Sentence Transformers)
- Combines retrieval and reasoning for accurate, context-rich answers

✅ **Smart Output**
- Generates structured research briefs
- Exports reports as PDFs or displays results via a clean Streamlit UI

---

## 🧩 Workflow Overview

1. **Input** — Upload PDF / enter URL / search topic  
2. **Extraction** — Clean and chunk text for embedding  
3. **Embedding Storage** — Store in **FAISS** or **Chroma** for retrieval  
4. **LangGraph Nodes**
   - Summarization Agent
   - Insight Extraction Agent
   - Question-Answering Agent
   - Report Generator  




## Helful Resources
1. **Document loaders** - https://python.langchain.com/docs/integrations/document_loaders/ 




