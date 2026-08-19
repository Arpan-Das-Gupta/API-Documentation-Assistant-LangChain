import os
import hashlib

import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_app.rag import ask_question, rebuild_vectorstore


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="API Documentation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        .title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            margin-bottom: 2rem;
        }

        .answer-box {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background: rgba(128,128,128,0.08);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            line-height: 1.6;
        }

        .source {
            color: #777;
            font-size: 0.85rem;
            margin: 0.2rem 0;
        }

        .question-label {
            color: #777;
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📁 Documentation")

    uploaded_file = st.file_uploader(
        "Upload Documentation",
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT or Markdown documentation."
    )

    st.caption("200 MB per file • PDF, TXT, MD")

    if uploaded_file:

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        file_bytes = uploaded_file.getvalue()

        # Create a stable hash for the uploaded file.
        # This lets us detect whether the exact same file
        # has already been indexed.
        file_hash = hashlib.md5(file_bytes).hexdigest()

        os.makedirs(
            "data/uploads",
            exist_ok=True
        )

        file_path = os.path.join(
            "data/uploads",
            uploaded_file.name
        )

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        st.session_state.uploaded_file = uploaded_file.name

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        # ----------------------------------------------------
        # Only index NEW files
        # ----------------------------------------------------

        if st.session_state.uploaded_file_hash != file_hash:

            with st.spinner("Indexing documentation..."):

                # ------------------------------------------------
                # Load document
                # ------------------------------------------------

                if file_path.lower().endswith(".pdf"):

                    loader = PyPDFLoader(
                        file_path
                    )

                elif file_path.lower().endswith(".md"):

                    # Load Markdown as plain text.
                    #
                    # This avoids requiring the optional
                    # "unstructured" package.
                    loader = TextLoader(
                        file_path,
                        encoding="utf-8"
                    )

                else:

                    loader = TextLoader(
                        file_path,
                        encoding="utf-8"
                    )

                docs = loader.load()

                # ------------------------------------------------
                # Split document
                # ------------------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100
                )

                chunks = splitter.split_documents(
                    docs
                )

                # ------------------------------------------------
                # Give every chunk a unique ID
                # ------------------------------------------------

                for index, chunk in enumerate(chunks):

                    chunk.metadata["chunk_id"] = (
                        f"{file_hash}_{index}"
                    )

                # ------------------------------------------------
                # Rebuild FAISS
                # ------------------------------------------------

                rebuild_vectorstore(
                    chunks
                )

                # ------------------------------------------------
                # Save session information
                # ------------------------------------------------

                st.session_state.uploaded_file_hash = (
                    file_hash
                )

                st.session_state.uploaded_file = (
                    uploaded_file.name
                )

                st.session_state.indexed = True

            st.success(
                f"Documentation indexed: {len(chunks)} chunks"
            )

        else:

            st.info(
                "This document is already indexed."
            )

    elif st.session_state.uploaded_file:

        st.info(
            f"Current document:\n"
            f"{st.session_state.uploaded_file}"
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="title">'
    '🤖 API Documentation Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your uploaded documentation.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for item in st.session_state.history:

    # --------------------------------------------------------
    # User
    # --------------------------------------------------------

    st.markdown(
        '<div class="question-label">You</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"**{item['question']}**"
    )

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    st.markdown(
        '<div class="question-label">Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="answer-box">'
        f'{item["answer"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    if item["sources"]:

        st.markdown("**📚 Sources**")

        seen = set()

        for source in item["sources"]:

            key = (
                source.get("source"),
                source.get("page")
            )

            if key in seen:
                continue

            seen.add(key)

            filename = os.path.basename(
                source.get(
                    "source",
                    "Unknown"
                )
            )

            page = source.get("page")

            if page is not None:

                st.markdown(
                    f'<div class="source">'
                    f'📄 {filename} — Page {page + 1}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f'<div class="source">'
                    f'📄 {filename}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.divider()


# ============================================================
# BOTTOM QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documentation..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    with st.spinner("Searching documentation..."):

        answer, sources = ask_question(
            question
        )

    st.session_state.history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    )

    st.rerun()