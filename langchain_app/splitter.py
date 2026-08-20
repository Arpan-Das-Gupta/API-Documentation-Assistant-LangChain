from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("docs/sample_api.md")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):

    print(f"Chunk {i+1}")
    print("-" * 40)
    print(chunk.page_content)
    print()