import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.model import Credentials
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader
import pdfplumber
from io import StringIO



# Page title
st.set_page_config(page_title='🦜🔗 Document Summarization App')
st.title('🦜🔗 Ask questions about the document')

# File input

uploaded_file = st.file_uploader("Add text file !")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    #st.write(string_data)

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunked_docs = text_splitter.split_documents(string_data)
st.write(len(chunked_docs)

    
