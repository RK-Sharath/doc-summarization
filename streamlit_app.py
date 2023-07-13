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
import pdfplumber



def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None

def generate_res(data):
    #Define llm
    llm = LangChainInterface(
        model=ModelType.FLAN_T5_11B,
        credentials=Credentials(api_key=genai_api_key),
        params=GenerateParams(
            decoding_method="greedy",
            max_new_tokens=600,
            min_new_tokens=150,
            repetition_penalty=2,
        ).dict())
    # Split text
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunked_docs = splitter.create_documents(data)
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(chunked_docs)


# Page title
st.set_page_config(page_title='🦜🔗 Document Summarization App')
st.title('🦜🔗 Document Summarization App')


# File input
doc_input = st.file_uploader('Choose your .pdf file', type="pdf")
if doc_input is not None:
    data = extract_data(doc_input)
    
#doc_input = st.file_uploader('Choose your .pdf file', type="pdf")

# Form to accept user's input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    genai_api_key = st.text_input('GenAI API Key', type = 'password', disabled=not doc_input)
    submitted = st.form_submit_button('Submit')
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            response = generate_res(doc_input)
            result.append(response)
            del genai_api_key

if len(result):
    st.info(response)
    
