
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def set_qa_system(file_path):
    loader=PyPDFLoader(file_path)
    docs=loader.load_and_split()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)

    embeddings=OpenAIEmbeddings()
    vector_store=FAISS.from_documents(chunks,embeddings)

    retriever=vector_store.as_retriever()
    llm=ChatOpenAI(temperature=0,model_name='gpt-4o')

    qa_chain=RetrievalQA.from_chain_type(llm,retriever=retriever)

    return qa_chain


if __name__ == '__main__':
    qa_chain=set_qa_system('LLm_dukeuni.pdf')

    while True:
        question=input('\nAsk a question:   ')
        if question.lower() == 'exit':
            break
        
        answer=qa_chain.invoke(question)
        answer = qa_chain.invoke({"query": question})
        print(answer["result"])

        
