import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_app.rag import ask_question, rebuild_vectorstore


st.set_page_config(
    page_title="API Documentation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- CSS ----------
st.markdown("""
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
""", unsafe_allow_html=True)


# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# ---------- Sidebar ----------
with st.sidebar:

    st.title("📁 Documentation")

    uploaded_file = st.file_uploader(
        "Upload Documentation",
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT or Markdown documentation."
    )

    st.caption("200 MB per file • PDF, TXT, MD")

    if uploaded_file:

        os.makedirs("data/uploads", exist_ok=True)

        file_path = os.path.join(
            "data/uploads",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file = uploaded_file.name

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        # Load document
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file_path.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)

        else:
            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        docs = loader.load()

        # Split document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        # Build FAISS vector store
        rebuild_vectorstore(chunks)

        st.success(
            f"Documentation indexed: {len(chunks)} chunks"
        )

    elif st.session_state.uploaded_file:

        st.info(
            f"Current document:\n"
            f"{st.session_state.uploaded_file}"
        )


# ---------- Main ----------
st.markdown(
    '<div class="title">🤖 API Documentation Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your uploaded documentation.'
    '</div>',
    unsafe_allow_html=True
)


# ---------- Chat History ----------
for item in st.session_state.history:

    st.markdown(
        '<div class="question-label">You</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"**{item['question']}**"
    )

    st.markdown(
        '<div class="question-label">Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="answer-box">{item["answer"]}</div>',
        unsafe_allow_html=True
    )

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
                source.get("source", "Unknown")
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


# ---------- Bottom Question Input ----------
question = st.chat_input(
    "Ask a question about your documentation..."
)


if question:

    with st.spinner("Searching documentation..."):

        answer, sources = ask_question(question)

    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })

    st.rerun()