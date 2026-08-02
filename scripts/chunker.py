def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start : end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


sample_text = """
Authentication:
Use your API key to authenticate requests.

Rate Limits:
Each user can make 100 requests per minute.

Errors:
A 401 error means authentication failed.
"""

chunks = chunk_text(sample_text, chunk_size=60, overlap=15)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}")
    print("-" * 30)
    print(chunk)