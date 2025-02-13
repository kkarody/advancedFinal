import streamlit as st
import ollama  # type: ignore
import chromadb  # type: ignore
from docx import Document  # type: ignore 
from PyPDF2 import PdfReader  # type: ignore 

client = chromadb.Client()

if "uploaded_docs" in [col.name for col in client.list_collections()]:
    collection = client.get_collection(name="uploaded_docs")
else:
    collection = client.create_collection(name="uploaded_docs")

st.title("QA by docs app")
st.markdown("Please, upload documents and ask questions related to content from them!")

uploaded_files = st.file_uploader(
    "Upload one or multiple docs:",
    accept_multiple_files=True,
    type=["txt", "pdf", "docx"]
)

model_choice = st.radio("Select AI Model:", ("llama2", "mistral"))

user_query = st.text_input("Enter your query:")

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

if st.button("get response"):
    if uploaded_files:
        for file in uploaded_files:
            if file.type == "text/plain":
                text = file.read().decode("utf-8")
            elif file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            else:
                st.error("This file type is not supported.")
                continue

            embedding = ollama.embeddings(model="mxbai-embed-large", prompt=text)["embedding"]
            collection.add(ids=[file.name], embeddings=[embedding], documents=[text])

        query_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=user_query)["embedding"]
        
        results = collection.query(query_embeddings=[query_embedding], n_results=1)

        if results["documents"]:
            relevant_doc = results["documents"][0][0]
            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": f"Context: {relevant_doc}. Question: {user_query}"}]
            )
            st.success(response["message"])
        else:
            st.error("document not found")
    else:
        st.error("upload at least one document")
