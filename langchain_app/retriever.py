from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore/langchain_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

question = input("Question: ")

results = vectorstore.similarity_search_with_score(
    question,
    k=3
)

for document, score in results:
    print("\nScore:", score)
    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page"))
    print("\n", document.page_content[:1000])