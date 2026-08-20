import os
import re

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from transformers import pipeline


VECTORSTORE_PATH = "vectorstore/langchain_faiss"

FALLBACK = "I couldn't find that information in the documentation."

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def get_generator():
    return pipeline(
        "text-generation",
        model=LLM_MODEL
    )


@st.cache_resource
def load_rag():
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = list(vectorstore.docstore._dict.values())

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5

    generator = get_generator()

    return vectorstore, bm25, generator


def rebuild_vectorstore(new_docs):
    """
    Rebuild the FAISS index using the existing sample documentation
    plus the newly uploaded documents.
    """

    embeddings = get_embeddings()

    # --------------------------------------------------
    # Load existing documentation if available
    # --------------------------------------------------

    existing_docs = []

    if os.path.exists(VECTORSTORE_PATH):
        try:
            existing_store = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            existing_docs = list(
                existing_store.docstore._dict.values()
            )

        except Exception:
            existing_docs = []

    # --------------------------------------------------
    # Combine existing + new documents
    # --------------------------------------------------

    all_docs = existing_docs + new_docs

    # --------------------------------------------------
    # Deduplicate
    # --------------------------------------------------

    unique = {}

    for doc in all_docs:

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content.strip()
        )

        unique[key] = doc

    all_docs = list(unique.values())

    # --------------------------------------------------
    # Create FAISS
    # --------------------------------------------------

    vectorstore = FAISS.from_documents(
        all_docs,
        embeddings
    )

    os.makedirs(
        os.path.dirname(VECTORSTORE_PATH),
        exist_ok=True
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    # Clear cached RAG objects
    load_rag.clear()

    return vectorstore


def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def ask_question(question):

    vectorstore, bm25, generator = load_rag()

    question = question.strip()

    if not question:
        return FALLBACK, []

    # --------------------------------------------------
    # Semantic search
    # --------------------------------------------------

    semantic_results = vectorstore.similarity_search_with_score(
        question,
        k=8
    )

    # Keep reasonably relevant semantic results
    semantic_docs = []

    for doc, score in semantic_results:

        if score <= 1.8:
            semantic_docs.append(
                (doc, score)
            )

    # --------------------------------------------------
    # Keyword search
    # --------------------------------------------------

    keyword_docs = bm25.invoke(question)

    # --------------------------------------------------
    # Combine results
    # --------------------------------------------------

    candidates = {}

    for doc, score in semantic_docs:

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content
        )

        candidates[key] = {
            "doc": doc,
            "score": score
        }

    for rank, doc in enumerate(keyword_docs):

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content
        )

        if key not in candidates:

            candidates[key] = {
                "doc": doc,
                # BM25 results come after semantic results
                "score": 1.7 + (rank * 0.01)
            }

    # --------------------------------------------------
    # No relevant documents
    # --------------------------------------------------

    if not candidates:
        return FALLBACK, []

    # --------------------------------------------------
    # Rank
    # --------------------------------------------------

    ranked = sorted(
        candidates.values(),
        key=lambda x: x["score"]
    )

    documents = [
        item["doc"]
        for item in ranked[:5]
    ]

    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    context = "\n\n".join(
        clean_text(doc.page_content)
        for doc in documents
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""<|im_start|>system
You are a documentation question-answering assistant.

You MUST answer using ONLY the documentation provided.

Rules:
- Use only facts explicitly present in the documentation.
- Do not use outside knowledge.
- Do not guess.
- Do not infer.
- Do not invent information.
- If the documentation does not contain the answer, say exactly:

I couldn't find that information in the documentation.

Keep the answer short and direct.
<|im_end|>

<|im_start|>user
DOCUMENTATION:

{context}

QUESTION:

{question}
<|im_end|>

<|im_start|>assistant
"""

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    result = generator(
        prompt,
        max_new_tokens=60,
        return_full_text=False,
        do_sample=False
    )

    answer = result[0]["generated_text"].strip()

    # --------------------------------------------------
    # Clean output
    # --------------------------------------------------

    answer = answer.split(
        "<|im_end|>"
    )[0].strip()

    if not answer:
        answer = FALLBACK

    lower_answer = answer.lower()

    if (
        "couldn't find" in lower_answer
        or "could not find" in lower_answer
        or "not found in the documentation" in lower_answer
        or "not mentioned in the documentation" in lower_answer
        or "not provided in the documentation" in lower_answer
    ):
        answer = FALLBACK

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    sources = []

    seen = set()

    for doc in documents:

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page")
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        })

    return answer, sources