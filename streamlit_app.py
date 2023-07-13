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
import pdfminer
from pdfminer.high_level import extract_pages



# Page title
st.set_page_config(page_title='🦜🔗 Document Summarization App')
st.title('🦜🔗 Document Summarization App')

# File input

uploaded_file = st.file_uploader("Choose a file", "pdf")
if uploaded_file is not None:
    for page_layout in extract_pages(uploaded_file):
        for element in page_layout:
            st.write(element)

def generate_res(text):
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
    chunked_docs = splitter.create_documents(text)
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(chunked_docs)



# Form to accept user's input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    genai_api_key = st.text_input('GenAI API Key', type = 'password', disabled=not uploaded_file)
    submitted = st.form_submit_button('Submit')
    if submitted and genai_api_key.startswith('pak-'):
        with st.spinner('Working on it...'):
            response = generate_res(uploaded_file)
            result.append(response)
            del genai_api_key

if len(result):
    st.info(response)
    
