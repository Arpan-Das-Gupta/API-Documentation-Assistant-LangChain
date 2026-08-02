from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.document_loader import load_document
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

all_text = ""

docs_folder = Path("docs")

for file in docs_folder.iterdir():

    if file.suffix.lower() in [".md", ".txt", ".pdf"]:

        print(f"Loading {file.name}")

        all_text += load_document(file)
        all_text += "\n\n"

chunks = [chunk.strip() for chunk in all_text.split("\n\n") if chunk.strip()]

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vectorstore/faiss_index.bin")

with open("vectorstore/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("FAISS index saved successfully!")
print("Chunks saved successfully!")

# print("\nChunks:\n")

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}")
#     print(chunk)
#     print("-" * 40)