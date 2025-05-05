import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle


@st.cache_data
def load_and_process_pdfs(pdf_folder_path, force_reprocess=False):
    # Cache file for storing chunks
    chunks_cache = os.path.join(pdf_folder_path, "document_chunks.pkl")
    
    # Return cached chunks if they exist and we're not forcing reprocessing
    if not force_reprocess and os.path.exists(chunks_cache):
        print("Loading cached document chunks...")
        with open(chunks_cache, 'rb') as f:
            return pickle.load(f)
    
    # Process documents if cache doesn't exist or we're forcing reprocessing
    print("Processing documents and creating chunks...")
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Save chunks to cache file
    with open(chunks_cache, 'wb') as f:
        pickle.dump(splits, f)
    
    return splits