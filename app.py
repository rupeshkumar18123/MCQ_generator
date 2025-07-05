from langchain_openai import AzureChatOpenAI
import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
st.set_page_config(page_title="MCQ Generator", page_icon="💹", layout="centered")
st.title("💹 AI‑Powered MCQ generator from pdf")

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model_name="gpt-4o",
    temperature=0.3,
)

# Upload PDF
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

# 1. Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# 2. Chunk the text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# 3. Store chunks in FAISS
def create_faiss_index(chunks, embeddings):
    return FAISS.from_texts(chunks, embeddings)

# 4. Define the prompt (✅ now includes {question})
prompt_template = """
You are an expert in generating multiple choice questions (MCQs) and short answer questions from a given text.
Given the following content, generate:
- 15 MCQs with 4 options each and indicate the correct answer.
- 5 short answer questions with concise answers.

Content:
{context}

Question:
{question}
"""

# 5. Main logic
if pdf_file is not None:
    # Save uploaded PDF to a temp file for PyPDF2
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file.flush()
        text = extract_text_from_pdf(tmp_file.name)

    # Chunk and embed
    chunks = chunk_text(text)
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("embedding_AZURE_OPENAI_API_BASE"),
        azure_deployment=os.getenv("embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("embedding_AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("embedding_AZURE_OPENAI_API_VERSION"),
    )
    faiss_index = create_faiss_index(chunks, embeddings)

    # RetrievalQA chain
    retriever = faiss_index.as_retriever(search_kwargs={"k": 6})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": ChatPromptTemplate.from_template(prompt_template)
        }
    )

    # Generate MCQs and Short Answers
    result = qa_chain({"query": "Generate 15 MCQs with answers and 5 short answer questions from the content."})
    st.write(result["result"])
