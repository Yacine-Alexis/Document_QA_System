# ← Streamlit/Flask frontend


import streamlit as st
from ingest import ingest_document
from qa_chain import load_qa_chain
import os

st.set_page_config(page_title="📄 Document Q&A System")

st.title("📄 Upload PDF and Ask Questions")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf:
    with open(f"documents/{pdf.name}", "wb") as f:
        f.write(pdf.read())
    st.success("PDF uploaded!")
    if st.button("Embed Document"):
        ingest_document(f"documents/{pdf.name}")
        st.success("Document embedded into vector store.")

# Ask questions
query = st.text_input("Ask a question about the PDF")
if query:
    chain = load_qa_chain()
    result = chain.run(query)
    st.write("🔍 Answer:")
    st.write(result)
