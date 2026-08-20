import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


DOCS_DIR = "docs"
VECTORSTORE_PATH = "vectorstore/langchain_faiss"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_documents():

    documents = []

    for filename in os.listdir(DOCS_DIR):

        path = os.path.join(
            DOCS_DIR,
            filename
        )

        if not os.path.isfile(path):
            continue

        lower = filename.lower()

        if lower.endswith(".pdf"):

            loader = PyPDFLoader(path)

        elif lower.endswith(".md"):

            loader = UnstructuredMarkdownLoader(path)

        elif lower.endswith(".txt"):

            loader = TextLoader(
                path,
                encoding="utf-8"
            )

        else:
            continue

        documents.extend(
            loader.load()
        )

    return documents


def main():

    documents = load_documents()

    print(
        f"Loaded {len(documents)} document pages."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        "vectorstore",
        exist_ok=True
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    print(
        f"FAISS index saved to {VECTORSTORE_PATH}"
    )


if __name__ == "__main__":
    main()