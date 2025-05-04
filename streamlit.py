import streamlit as st

from main import process

st.title("YOLO Expert")

user_input = st.text_input("Enter your question about YOLO:", "")

if st.button("Submit"):
    try:
        response = process(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")