from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.model import Credentials
import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.chains import RetrievalQA
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma


# Page title
st.set_page_config(page_title='🔗 Retriever Augmented Generation Demo')
st.caption("This demo is prepared by Sharath Kumar RK, Senior Data Scientist, Watsonx team")
st.title('🦜🔗 Ask questions about your document')

#genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_new_tokens = st.sidebar.text_input("Select max new tokens", type="default")
min_new_tokens = st.sidebar.text_input("Select min new tokens", type="default")

@st.cache_data
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

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits
         

@st.cache_resource
def create_retriever(_embeddings, splits):
         vectorstore = Chroma.from_texts(splits, _embeddings)
         retriever = vectorstore.as_retriever
         return retriever


def main():

# Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"
    embeddings = HuggingFaceInstructEmbeddings()

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

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
                 # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

    retriever = create_retriever(embeddings, splits, retriever_type)
    qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)
    st.write("Ready to answer questions.")

        # Question and answering
    user_question = st.text_input("Enter your question:")
    if user_question:
        answer = qa.run(user_question)
        st.write("Answer:", answer)


if __name__ == "__main__":
    main()
    
