import streamlit as st
from rag_pipeline import load_documents, split_documents, create_vectorstore, build_qa_chain
from langchain.vectorstores import Chroma

st.title("📊 Annual Report RAG Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload an Annual Report (PDF)", type=["pdf"])

if uploaded_file:
    with open("data/annual_report.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing document...")

    docs = load_documents("data/annual_report.pdf")
    chunks = split_documents(docs)
    vectordb = create_vectorstore(chunks)
    qa_chain = build_qa_chain(vectordb)

    st.success("Chatbot ready! Ask questions about the report below.")

    # Chat interface
    query = st.text_input("Ask a question:")
    if query:
        answer = qa_chain.run(query)
        st.write("💬 **Answer:**", answer)
