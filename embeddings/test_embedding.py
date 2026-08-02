from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "How do I login?"
sentence2 = "How do I authenticate?"
sentence3 = "How do I cook pasta?"

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

similarity_12 = cosine_similarity(
    [embedding1],
    [embedding2]
)

similarity_13 = cosine_similarity(
    [embedding1],
    [embedding3]
)

print("Login vs Authenticate: ")
print(similarity_12)

print()

print("Login vs Cook Pasta: ")
print(similarity_13)