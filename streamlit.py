import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
st.title("YOLO Expert")

from main import process

user_input = st.text_input("Enter your question about YOLO:", "")

if st.button("Submit"):
    try:
        response = process(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")