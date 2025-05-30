from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_documents(docs, embeddings)
