import streamlit as st
import os
import tempfile
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
import PyPDF2
from io import StringIO
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate


st.title("Document Summarization App powered by IBM Watsonx")
st.caption("This app was developed by Sharath Kumar RK, IBM Ecosystem Engineering Watsonx team")

model = st.radio("Select the Watsonx LLM model",('google/flan-t5-xl','google/flan-t5-xxl','google/flan-ul2'))
genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.number_input("Select max new tokens", value=600)
min_new_tokens = st.sidebar.number_input("Select min new tokens", value=150)
chunk_size = st.sidebar.number_input("Select chunk size", value=1200)
chunk_overlap = st.sidebar.number_input("Select chunk overlap", value=100)
chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
with st.sidebar:
    decoding_method = st.radio(
        "Select decoding method",
        ('greedy', 'sample')
    )
temperature = st.sidebar.number_input("Temperature (Choose a decimal number between 0 & 2)", value=0.4)
repetition_penalty = st.sidebar.number_input("Repetition penalty (Choose either 1 or 2)", value=2)
num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)



        
uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)


def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf file.', icon="⚠️")
    return all_text


text = load_docs(uploaded_files)
#st.write(text)

def setup_documents(chunk_size, chunk_overlap):
    docs_raw = text
    #docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #docs = text_splitter.create_documents(docs_raw_text)
    docs_split = text_splitter.split_text(docs_raw)
    docs = text_splitter.create_documents(docs_split)
    return docs

def custom_summary(docs,llm, custom_prompt, chain_type, num_summaries):
    
    custom_prompt = custom_prompt + """:\n\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])
    
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=chain_type, 
                                    map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)
        
    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    
    return summaries

def main():
    
    if 'genai_api_key' not in st.session_state:
        genai_api_key = st.text_input(
            'Please enter your GenAI API key', value="", placeholder="Enter the GenAI API key which begins with pak-")
        if genai_api_key:
            st.session_state.genai_api_key = genai_api_key
            os.environ["GENAI_API_KEY"] = genai_api_key
        else:
            return
    else:
        os.environ["GENAI_API_KEY"] = st.session_state.genai_api_key
        
    user_prompt = st.text_input("Enter the user prompt")
    creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
    # Define parameters
    params = GenerateParams(decoding_method=decoding_method, temperature=temperature, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, repetition_penalty=repetition_penalty)
    # Instantiate LLM model
    llm=LangChainInterface(model=model, params=params, credentials=creds)
    if uploaded_files != "":
        docs = setup_documents(chunk_size, chunk_overlap)
    # Display the number of text chunks
    num_chunks = len(docs)
    st.write(f"Number of text chunks: {num_chunks}")
    st.write("Doc was loaded successfully")
    if st.button("Summarize"):
        with st.spinner('Working on it...'):
            result = custom_summary(docs,llm, user_prompt, chain_type, num_summaries)
            st.write("Summaries:")
            for summary in result:
                st.write(summary)

if __name__ == "__main__":
    main()
