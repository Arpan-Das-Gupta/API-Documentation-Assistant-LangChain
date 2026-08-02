import faiss
import numpy as np

# Three vectors with 2 dimensions each
vectors = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
], dtype="float32")

# Embedding dimension
dimension = vectors.shape[1]

# Create a FAISS index
index = faiss.IndexFlatL2(dimension)

# Add vectors to the index
index.add(vectors)

print("Dimension:", dimension)
print("Number of vectors:", index.ntotal)

query = np.array([
    [2.1, 3.1]
], dtype="float32")

distances, indices = index.search(query, k=1)

print("Distances:", distances)
print("Indices:", indices)