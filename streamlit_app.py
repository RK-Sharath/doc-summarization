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
from langchain import OpenAI



# Page title
st.set_page_config(page_title='🦜🔗 Document Summarization App')
st.title('🦜🔗 Summarize the document')

genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.text_input("Select max new tokens", type="default")
min_new_tokens = st.sidebar.text_input("Select min new tokens", type="default")

# File input

uploaded_file = st.file_uploader("Add text file !")

def extract_data():
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        return string_data
    
    #st.write(string_data)
    # Create multiple documents
    #docs = [Document(page_content=t) for t in texts]
    #st.write(texts)

def generate_res(data):
     
    # Instantiate the LLM model

    llm = OpenAI(temperature=0, openai_api_key=genai_api_key)
    
    # Text summarization
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.create_documents(data)
    #docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(texts)
    
data = extract_data()
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            response = generate_res(data)
            result.append(response)
            del genai_api_key

if len(result):
     st.info(response)
    
