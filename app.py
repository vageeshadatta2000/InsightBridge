import streamlit as st
from dotenv import load_dotenv
import os
from utils.document_loader import load_and_split
from utils.vectorstore import build_vectorstore
from utils.qa_pipeline import get_qa_chain
from utils.visualizer import extract_entities
import tempfile
import fitz
import pdfplumber

load_dotenv()

st.set_page_config(page_title="InsightBridge", layout="wide")
st.title("📄 InsightBridge")
st.caption("Explore and understand documents with LLM-powered insights.")

uploaded_file = st.file_uploader("Upload Document (TXT, XML, PDF)", type=["txt", "xml", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp:
        tmp.write(uploaded_file.getvalue())
        file_path = tmp.name

    st.info("Processing document...")
    docs = load_and_split(file_path)
    vs = build_vectorstore(docs)
    qa_chain = get_qa_chain(vs)
    st.success("Document ready!")

    query = st.text_input("🔍 Ask a question about the document")
    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)
            st.write("**Answer:**")
            st.markdown(result["result"])
            with st.expander("🔎 Sources"):
                for source in result['source_documents']:
                    st.markdown(f"- {source.page_content[:300]}...")

    with st.expander("📊 View Entities"):
        if st.button("Extract Entities"):
            entities = extract_entities(docs)
            st.json(entities)
