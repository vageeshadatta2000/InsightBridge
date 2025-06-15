# 🚀 InsightBridge

**InsightBridge** is an AI-powered document analysis system built with Streamlit and LangChain. Upload complex documents (PDF, XML, TXT), ask natural language questions, extract entities, and get structured summaries with source-aware responses.


## 🧠 Features

- 📄 Upload support for TXT, XML, and PDF (via `pdfplumber` and `PyMuPDF`)
- 🔍 Ask natural questions, get AI-generated answers with citations
- 🧩 Powered by LangChain, OpenAI (or HuggingFace), and FAISS
- 🧠 Entity extraction using spaCy and D3.js-compatible output
- 🛠️ Modular architecture, easy to expand (RAG, summarization, etc.)
- ⚡ Fast local search with vector embedding indexing


## 🎯 Use Cases

- Researchers analyzing academic papers
- Finance teams reviewing long reports
- Lawyers digesting contracts
- Students summarizing complex readings

## 🎥 Demo

> 📽️ Watch the demo video below (hosted on GitHub):

<video src="assets/Insight-Demo.mp4" controls width="700">
  Demo Video
</video>



## 🧰 Tech Stack

| Layer         | Tools/Frameworks                                 |
|---------------|--------------------------------------------------|
| Backend       | Python, LangChain, OpenAI/HuggingFace, FAISS     |
| Frontend      | Streamlit                                        |
| Embeddings    | OpenAI Embeddings (default), Sentence-Transformers |
| PDF Parsing   | PyMuPDF (`fitz`), pdfplumber                     |
| Entity NLP    | spaCy (en_core_web_sm)                           |
| Infrastructure| Docker-ready (coming soon)                       |


