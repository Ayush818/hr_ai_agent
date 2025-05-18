import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # For interacting with local Ollama LLM

# LangChain components
from langchain_community.vectorstores import Chroma  # Updated import for Chroma
from langchain_core.prompts import PromptTemplate

# Embedding model (same as in ingest.py)
# from langchain.embeddings import HuggingFaceEmbeddings # Old way
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # New way (if you updated per warning)
from pydantic import BaseModel  # For defining request/response models

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file, if any

# Paths and Model Names (should match ingest.py where applicable)
CHROMA_DB_PATH = "chroma_db_hr"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_KWARGS = {
    "device": "cpu"
}  # Or 'mps' if you want to try M1 GPU, 'cuda' for Nvidia
ENCODE_KWARGS = {"normalize_embeddings": False}

OLLAMA_MODEL_NAME = "mistral:7b-instruct-q4_K_M"  # The Ollama model you pulled

# --- Global Variables (initialized on startup) ---
embeddings_model = None
vector_store = None
qa_chain = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HR AI Agent API",
    description="API for interacting with the HR AI Agent for policy Q&A.",
    version="0.1.0",
)


# --- Pydantic Models for Request and Response ---
class QuestionRequest(BaseModel):
    query: str
    session_id: str = "default_session"  # Optional: for future session management


class AnswerResponse(BaseModel):
    answer: str
    source_documents: list = []  # Optional: to show which docs were used


# --- Helper Functions / Initialization Logic ---
def initialize_components():
    """Initializes embeddings, vector store, and QA chain on app startup."""
    global embeddings_model, vector_store, qa_chain

    print("Initializing API components...")

    # 1. Initialize Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    )

    # 2. Load Vector Store
    if not os.path.exists(CHROMA_DB_PATH):
        print(
            f"ERROR: Chroma DB path not found: {CHROMA_DB_PATH}. Please run ingest.py first."
        )
        # You might want to raise an exception here or handle it more gracefully
        return

    print(f"Loading vector store from: {CHROMA_DB_PATH}")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH, embedding_function=embeddings_model
    )
    print(f"Vector store loaded. Collection count: {vector_store._collection.count()}")

    # 3. Initialize LLM
    print(f"Initializing Ollama LLM with model: {OLLAMA_MODEL_NAME}")
    llm = Ollama(model=OLLAMA_MODEL_NAME)
    # You can add parameters like temperature, top_k etc.
    # llm = Ollama(model=OLLAMA_MODEL_NAME, temperature=0.7, top_k=50)

    # 4. Create RetrievalQA Chain
    # This chain combines retrieval from vector_store and generation with llm
    prompt_template = """You are an AI assistant for answering questions about company policies.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer from the context or the context isn't relevant, say that you cannot answer based on the provided information.
    Do not make up an answer outside of the provided context.
    Be concise and helpful.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # k=3 means retrieve the top 3 most similar documents/chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" puts all retrieved docs directly into the prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,  # So we can see which documents were used
    )
    print("RetrievalQA chain created successfully.")
    print("API components initialized.")


# --- FastAPI Event Handler for Startup ---
@app.on_event("startup")
async def startup_event():
    """This function will be called when FastAPI starts up."""
    initialize_components()
    if not qa_chain:
        print("FATAL: QA Chain could not be initialized. API might not work correctly.")
        # In a production system, you might want the app to exit or enter a degraded state.


# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "Welcome to the HR AI Agent API. Use the /ask endpoint to ask questions."
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Receives a question, retrieves relevant documents,
    and generates an answer using the LLM.
    """
    if not qa_chain:
        return AnswerResponse(
            answer="Error: QA system not initialized. Please check server logs.",
            source_documents=[],
        )

    print(f"Received question: {request.query}")
    try:
        # Invoke the QA chain
        # The input to qa_chain is a dictionary, typically with a "query" key
        result = qa_chain.invoke({"query": request.query})

        answer = result.get("result", "Sorry, I could not find an answer.")
        sources = result.get("source_documents", [])

        # Langchain Document objects are not directly JSON serializable for Pydantic.
        # We need to extract the relevant parts (e.g., page_content and metadata).
        source_docs_serializable = []
        for doc in sources:
            source_docs_serializable.append(
                {"page_content": doc.page_content, "metadata": doc.metadata}
            )

        print(f"Generated answer: {answer}")
        if source_docs_serializable:
            print(
                f"Sources used: {[s['metadata'].get('source', 'N/A') for s in source_docs_serializable]}"
            )

        return AnswerResponse(answer=answer, source_documents=source_docs_serializable)

    except Exception as e:
        print(f"Error during QA processing: {e}")
        # Consider more specific error handling here
        return AnswerResponse(
            answer=f"An error occurred: {str(e)}", source_documents=[]
        )


# To run this FastAPI application (from your terminal, in the hr_ai_agent directory with venv active):
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
