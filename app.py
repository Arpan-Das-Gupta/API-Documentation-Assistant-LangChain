import streamlit as st
from rag.pipeline import ask_question

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="API Documentation Assistant",
    page_icon="🤖",
    layout="centered"
)

with st.sidebar:

    st.title("🤖 API Assistant")

    st.markdown("---")

    st.markdown("### 🚀 Tech Stack")

    st.markdown("""
- 🧠 Gemini 3.5 Flash
- 🔍 FAISS Vector Search
- 📄 Sentence Transformers
- ⚡ Streamlit
""")

    st.markdown("---")

    st.markdown("### 📚 About")

    st.info(
        "Ask questions about your API documentation."
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

st.title("🤖 AI Documentation Assistant")

st.caption(
    "Ask questions about your documentation using Retrieval-Augmented Generation (RAG)."
)

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.write("Ask questions about your API documentation.")

question = st.chat_input(
    "Ask a question:"
    # placeholder="Example: How do I authenticate?"
)

if question:

    question = question.strip()

    if not question:
        st.warning("Please enter a valid question.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.spinner("Searching documentation..."):
        context, answer = ask_question(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.expander("📄 Source Documentation"):
        st.write(context)


st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit, FAISS, Sentence Transformers and Gemini."
)