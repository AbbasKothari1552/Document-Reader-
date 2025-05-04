from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


import os
from dotenv import load_dotenv
load_dotenv()

from src.doc_loader import load_and_process_pdfs 
from src.db_store import get_vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# path to documents
PDF_FOLDER_PATH = "pdfs"




def process(question):
    # split documents in chunks
    split_docs = load_and_process_pdfs(PDF_FOLDER_PATH)

    # load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # load or create vector database
    vectorstore = get_vector_store(split_docs, embeddings) # force_rebuild=True to build database. Default (False)


    yolo_expert_template = """
    You are a YOLO (You Only Look Once) model expert assistant. Your task is to provide accurate, detailed, and practical answers about:
    - YOLO architecture (v1 to v10)
    - Training, inference, and optimization
    - Model comparisons (speed, accuracy, use cases)
    - Integration with tools like Ultralytics, OpenCV, or TensorRT

    Use the following context to answer the question. If unsure, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question:
    {question}

    Answer in markdown with clear sections (e.g., **Architecture**, **Training Tips**, **Code Example** if needed).
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=yolo_expert_template
    )

    # API Keys
    os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY")

    # GEMINI LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # query = "What is YOLO?"
    # docs = vectorstore.similarity_search(query)
    # print(docs[0].page_content)
    
    return rag_chain.invoke(question)









