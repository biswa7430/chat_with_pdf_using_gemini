import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PIL import Image

load_dotenv()



api_key=os.getenv("GOOGLE_API_KEY")

model= genai.GenerativeModel("gemini-1.5-flash")

from io import BytesIO


def get_pdf_texts(pdf_docs):
    text = ""
    if pdf_docs:
            pdf_reader = PdfReader(pdf_docs)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible by using the given context.
    Don't provide wrong answers. If you don't know the answer, just say "I don't know".
    Context:\n{context}?\n
    Question:\n{question}\n
    
    Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if the FAISS index exists before loading
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        print(response)
        st.write("Reply: ", response["output_text"])
    else:
        st.write("FAISS index not found. Please upload and process the PDF first.")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a question that might be present in the PDF")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_texts(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()