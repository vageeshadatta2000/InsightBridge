import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- MODIFICATION: Import Google's classes instead of OpenAI's ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# --- END MODIFICATION ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

# --- UI Configuration ---
st.set_page_config(page_title="InsightBridge (Gemini Edition)", page_icon="💡")
st.title("💡 InsightBridge: LLM-Powered Document Analysis")
st.write(
    "Upload a long-form document (PDF) and ask questions. "
    "InsightBridge will use a RAG pipeline to provide context-aware answers."
)

# --- MODIFICATION: API Key Management for Google Gemini ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key) # This configuration is specific to Google
except KeyError:
    st.error("Google API key not found. Please add it to your Streamlit secrets.")
    st.info("For more info, see https://docs.streamlit.io/deploy/concepts/secrets-management")
    st.stop()
# --- END MODIFICATION ---

# --- Core Logic Functions ---

def process_document(uploaded_file):
    with st.spinner("Processing document... This may take a moment."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""], length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            st.error("Could not extract text. Please try another PDF.")
            os.remove(tmp_file_path)
            return None

        # --- MODIFICATION: Use Google's embedding model ---
        # The model "models/embedding-001" is Google's standard text embedding model.
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # --- END MODIFICATION ---

        try:
            vector_store = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            st.error(f"Error creating vector store: {e}. Check API key & network.")
            os.remove(tmp_file_path)
            return None
        
        os.remove(tmp_file_path)
        return vector_store.as_retriever()

def setup_rag_chain(retriever):
    prompt_template = """
    Answer the user's question based only on the following context:
    {context}
    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # --- MODIFICATION: Use Google's Chat Model (Gemini Pro) ---
    # `convert_system_message_to_human` is important for compatibility with some LangChain prompts.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.7, 
        convert_system_message_to_human=True
    )
    # --- END MODIFICATION ---
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# --- Main Application Flow (No changes needed here) ---

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    if st.button("Process Document"):
        if uploaded_file:
            retriever = process_document(uploaded_file)
            if retriever:
                st.session_state.rag_chain = setup_rag_chain(retriever)
                st.success("Document processed successfully! You can now ask questions.")
        else:
            st.warning("Please upload a PDF file first.")

st.header("Ask Your Document")

if st.session_state.rag_chain:
    user_question = st.text_input("Enter your question here:")
    if user_question:
        with st.spinner("Generating answer with Gemini..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": user_question})
                st.write(response["answer"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload and process a document to start the Q&A.")
