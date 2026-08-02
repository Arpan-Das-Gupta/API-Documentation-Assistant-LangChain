from pathlib import Path
from pypdf import PdfReader

def load_markdown(path):

    with open(path, "r", encoding="utf-8") as file:
        return file.read()
    
def load_text(path):

    with open(path, "r", encoding="utf-8") as file:
        return file.read()
    
def load_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text

def load_document(path):

    extension = Path(path).suffix.lower()

    if extension == ".md":
        return load_markdown(path)

    elif extension == ".txt":
        return load_text(path)

    elif extension == ".pdf":
        return load_pdf(path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )