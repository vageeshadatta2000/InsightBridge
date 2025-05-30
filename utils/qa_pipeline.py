from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def get_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
