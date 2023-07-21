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
from io import StringIO
from langchain import OpenAI
from langchain.document_loaders import TextLoader


# Page title
st.set_page_config(page_title='🦜🔗 Document Summarization App')
st.title('🦜🔗 Summarize the document')

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.text_input("Select max new tokens", type="default")
min_new_tokens = st.sidebar.text_input("Select min new tokens", type="default")

# File input

uploaded_file = st.file_uploader("Add text file !")
loader = TextLoader(uploaded_file)
loaded_docs=loader.load()
st.write(loaded_docs)

splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=10)
chunked_docs = splitter.split_documents(loaded_docs)
st.write(len(chunked_docs))
         
    
