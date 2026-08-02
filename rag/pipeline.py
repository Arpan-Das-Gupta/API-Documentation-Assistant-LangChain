import os
import pickle
import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from functools import lru_cache

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = os.getenv("GEMINI_MODEL")

@lru_cache(maxsize=1)
def load_resources():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        "vectorstore/faiss_index.bin"
    )

    with open("vectorstore/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return embedding_model, index, chunks

def retrieve(question):
    embedding_model, index, chunks = load_resources()

    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, k=1)

    return chunks[indices[0][0]]

def build_prompt(question, context):
    prompt = f"""
You are an API Documentation Assistant.

Answer ONLY using the documentation below.

Documentation:
{context}

Question:
{question}

If the answer is not in the documentation, say:
"I couldn't find that information in the documentation."
"""

    return prompt

def generate_answer(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text

    except Exception:
        return (
            "⚠️ Gemini is currently experiencing high demand.\n\n"
            "Please wait a few seconds and try again."
        )

def ask_question(question):
    context = retrieve(question)

    prompt = build_prompt(question, context)

    answer = generate_answer(prompt)

    return context, answer