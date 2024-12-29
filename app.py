import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import uuid

load_dotenv()

llm=ChatGroq(model="gemma2-9b-it")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
embeddings=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based on the provided context.
Think step by step before providing the answer.
<context>
{context}
</context>
Question:{input}
"""
)

def create_vectors():
        total_documents=[]
        for file in files:
            random_filename = f"{uuid.uuid4()}.pdf"
            file_path = os.path.join('files', random_filename)
            with open(file_path, mode='wb') as w:
                w.write(file.getvalue())
            loader=PyPDFLoader(file_path)
            docs=loader.load()
            documents=text_splitter.split_documents(docs)
            total_documents.extend(documents)
            os.remove(file_path)
        
        st.session_state.vectors=FAISS.from_documents(total_documents,embeddings)
        st.session_state.retriever=st.session_state.vectors.as_retriever()
        st.session_state.document_chain=create_stuff_documents_chain(llm,prompt)
        st.session_state.chain=create_retrieval_chain(st.session_state.retriever,st.session_state.document_chain)

st.title("RAG App")



files=st.file_uploader(label="Upload PDFs", type=['pdf'], accept_multiple_files=True,)

if st.button('Create VectorDB'):
    if files != []:
        create_vectors()
        st.write('Done')
    else:
         st.write("Upload PDFs")
    


if "vectors" in st.session_state:
    input = st.text_input('input')
    if st.button('Submit') and input:
        response=st.session_state.chain.invoke({'input':input})['answer']
        st.write(response)


    
