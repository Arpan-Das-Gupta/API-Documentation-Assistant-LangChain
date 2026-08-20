from langchain_community.document_loaders import TextLoader

loader = TextLoader("docs/sample_api.md")

documents = loader.load()

print(documents)