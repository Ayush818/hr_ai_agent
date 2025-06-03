import os

from dotenv import load_dotenv

# Vector Store
from langchain_chroma import Chroma  # Updated import

# LangChain components for document loading and processing
from langchain_community.document_loaders import (  # Assuming you might use these
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

# Embedding model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # Updated import if you use this package directly

# from langchain.text_splitter import RecursiveCharacterTextSplitter # Older import


# --- Configuration ---
load_dotenv()

POLICY_DOCS_PATH = "policies"
CHROMA_DB_PATH = "chroma_db_hr"  # For ChromaDB

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": False}


# --- Helper Functions ---
def load_documents(source_dir: str):
    """Loads documents from various file types in the source directory."""
    # Loader for .txt files
    text_loader_kwargs = {"autodetect_encoding": True}
    # Using TextLoader directly for .txt and .md for simplicity here.
    # DirectoryLoader can be used if you prefer its globbing and multi-file handling.

    all_docs = []
    print(f"Scanning for documents in: {source_dir}")

    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".txt") or file.endswith(".md"):
                    print(f"Loading text/md file: {file_path}")
                    loader = TextLoader(file_path, **text_loader_kwargs)
                    all_docs.extend(loader.load())
                elif file.endswith(".pdf"):
                    print(f"Loading PDF file: {file_path}")
                    loader = PyPDFLoader(file_path)
                    all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not all_docs:
        print(f"No documents found or loaded from {source_dir}.")
    else:
        print(f"Successfully loaded {len(all_docs)} document(s) from {source_dir}")
    return all_docs


def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Splits loaded documents into smaller chunks."""
    if not documents:
        print("No documents to split.")
        return []

    # Ensure you have langchain_text_splitters installed if using the new import
    # pip install langchain-text-splitters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def create_embeddings_model(model_name, model_kwargs, encode_kwargs):
    """Initializes the sentence transformer embedding model."""
    print(f"Initializing embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings


def create_and_persist_vector_store(chunks, embeddings_model, persist_directory):
    """Creates a Chroma vector store from document chunks and persists it."""
    if not chunks:
        print("No chunks to process. Vector store not created.")
        return None

    print(f"Creating/updating vector store at: {persist_directory}")
    # Chroma.from_documents will create or add to an existing persisted store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
    )
    # No explicit .persist() needed if persist_directory is given to from_documents
    print(
        f"Vector store operation completed. Collection count now: {vector_store._collection.count()}"
    )
    return vector_store


# --- Main Ingestion Logic ---
def main():
    print("Starting HR policy ingestion process...")

    # 1. Load documents
    documents = load_documents(POLICY_DOCS_PATH)
    if not documents:
        print("No documents loaded. Exiting.")
        return

    # 2. Split documents into chunks
    chunks = split_text_into_chunks(documents)
    if not chunks:  # Could happen if documents were empty or splitting failed
        print("No chunks created from documents. Exiting.")
        return

    # 3. Initialize embedding model
    embeddings_model = create_embeddings_model(
        EMBEDDING_MODEL_NAME, MODEL_KWARGS, ENCODE_KWARGS
    )

    # 4. Create and persist vector store
    vector_store = create_and_persist_vector_store(
        chunks, embeddings_model, CHROMA_DB_PATH
    )

    if vector_store:
        print("Ingestion process completed successfully!")
    else:
        print(
            "Ingestion process did not complete successfully or vector store not created."
        )


if __name__ == "__main__":
    # Ensure necessary packages for text splitting are installed if you changed the import
    # e.g. pip install langchain-text-splitters
    main()
