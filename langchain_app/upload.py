import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_PATH = "vectorstore/langchain_faiss"


def process_file(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )
    else:
        raise ValueError("Unsupported file type.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=75
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    vectorstore.save_local(INDEX_PATH)

    return len(documents), len(chunks)