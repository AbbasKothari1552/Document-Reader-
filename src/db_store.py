from langchain_community.vectorstores import FAISS
import streamlit as st

import os

@st.cache_data
def get_vector_store(_texts, _embeddings, force_rebuild=False):
    if not force_rebuild and os.path.exists("faiss_index/index.faiss"):
        print("Loading existing FAISS index...")
        return FAISS.load_local("faiss_index", _embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(documents=_texts, embedding=_embeddings)
        vector_store.save_local("faiss_index")
        return vector_store