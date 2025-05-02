from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # <- now it's visible to LangChain

def load_qa_chain(index_path="vectorstore"):
    vectorstore = FAISS.load_local(index_path, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    llm = ChatOpenAI(model_name="gpt-4")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return chain
