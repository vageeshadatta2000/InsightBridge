from langchain.document_loaders import TextLoader, UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import fitz
import pdfplumber

def load_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
    return text

def load_and_split(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".xml":
        loader = UnstructuredXMLLoader(file_path)
        docs = loader.load()
    elif suffix == ".pdf":
        text = load_pdf(file_path)
        docs = [text]
    else:
        loader = TextLoader(file_path, encoding="utf8")
        docs = loader.load()

    if isinstance(docs, list):
        from langchain.schema import Document
        docs = [Document(page_content=d) for d in docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)
