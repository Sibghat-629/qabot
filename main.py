import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers.pipelines import pipeline
import tempfile

st.set_page_config(page_title="🧠 PDF Question Answering")

st.title("📄 PDF Q&A Chatbot (Offline)")
st.write("Upload a PDF and ask any question about it using HuggingFace models.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded and loaded successfully.")

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = text_splitter.split_documents(pages)

    # Create embeddings
    with st.spinner("🔍 Generating vector embeddings..."):
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding)

    # Create HuggingFace LLM
    with st.spinner("🧠 Loading language model..."):
        qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            max_length=556,
            truncation=True
        )
        llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Ask a question
    st.subheader("❓ Ask a question about your PDF")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("🔍 Thinking..."):
            answer = qa_chain.invoke(query)
        st.success("✅ Answer:")
        st.write(answer)
