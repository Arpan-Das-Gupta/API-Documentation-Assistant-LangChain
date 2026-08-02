from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_index():
    return faiss.read_index("vectorstore/faiss_index.bin")


def load_chunks():
    with open("vectorstore/chunks.pkl", "rb") as file:
        return pickle.load(file)


def search(question, index, chunks):
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, k=3)

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)


# Load saved resources
index = load_index()
chunks = load_chunks()

question = "Monday"

result = search(question, index, chunks)

print("Question:")
print(question)

print("\nBest Match:")
print(result)