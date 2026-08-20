from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

def load_documents(folder="docs"):
    documents = []

    for file in Path(folder).glob("*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))

        elif file.suffix in [".txt", ".md"]:
            loader = TextLoader(
                str(file),
                encoding="utf-8"
            )

        else:
            continue

        documents.extend(loader.load())

    return documents