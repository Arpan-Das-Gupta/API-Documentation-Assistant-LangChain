# 🤖 API Documentation Assistant

A local Retrieval-Augmented Generation (RAG) application that lets users upload PDF, TXT, or Markdown documentation and ask questions about its contents.

The application uses semantic search with FAISS, Hugging Face embeddings, and a local language model to generate answers grounded in the uploaded documentation.

---

## ✨ Features

- 📄 Upload PDF, TXT, and Markdown documentation
- ✂️ Automatically split documents into smaller chunks
- 🔎 Semantic document retrieval using FAISS
- 🧠 Sentence-transformer embeddings
- 🤖 Local LLM-based answer generation
- 📚 Displays source documents and page numbers
- 🛡️ Answers are restricted to the uploaded documentation
- 🚫 Returns a fallback response when relevant information cannot be found
- 🖥️ Simple Streamlit interface
- 🔒 Runs locally without requiring an external LLM API

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Streamlit      │
                    │       app.py        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Document Upload   │
                    │   PDF / TXT / MD    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document Processing │
                    │  Loading + Chunking │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Embeddings      │
                    │ all-MiniLM-L6-v2    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        FAISS        │
                    │    Vector Store     │
                    └──────────┬──────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Similarity       │
                    │      Search         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Local LLM       │
                    │ Qwen2.5-0.5B-Instruct│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Sources    │
                    └─────────────────────┘


🛠️ Tech Stack

| Technology            | Purpose                    |
| --------------------- | -------------------------- |
| Python                | Application language       |
| Streamlit             | Web interface              |
| LangChain             | RAG application components |
| FAISS                 | Vector similarity search   |
| Hugging Face          | Embeddings and local LLM   |
| Sentence Transformers | Document embeddings        |
| Qwen2.5-0.5B-Instruct | Local answer generation    |
| PyPDF                 | PDF document loading       |


📁 Project Structure
.
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── langchain_app/
│   ├── ingest.py
│   └── rag.py
│
├── utils/
│   └── document_loader.py
│
├── docs/
│   ├── faq.txt
│   └── sample_api.md
│
├── assets/
│
├── data/
│   └── uploads/
│
├── vectorstore/
│   └── langchain_faiss/
│
└── tests/

Some development and experimental files may exist in the repository while the project is being refined.



🚀 Getting Started
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/API-Documentation-Assistant-LangChain.git
cd API-Documentation-Assistant-LangChain
2. Create a virtual environment

Linux/macOS:

python3 -m venv venv
source venv/bin/activate

Windows:

python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Start the application
streamlit run app.py

The application will open in your browser.

📄 Using the Application
Step 1 — Upload documentation

Upload a:

PDF
TXT
Markdown file
Step 2 — Index the document

The application processes the uploaded document and creates vector embeddings for its content.

Step 3 — Ask a question

Enter a question related to the uploaded documentation.

For example:

How did the house try to save itself?
Step 4 — View the answer

The application returns:

The generated answer
Relevant source documents
Page numbers when available
🔎 Retrieval

The application converts document chunks into vector embeddings using:

sentence-transformers/all-MiniLM-L6-v2

The embeddings are stored in a FAISS vector index.

When a user asks a question, the application performs similarity search to retrieve the most relevant document chunks.

A similarity threshold is also used to reduce answers to questions that are unrelated to the uploaded documentation.

🤖 Local Language Model

Answer generation uses:

Qwen/Qwen2.5-0.5B-Instruct

The model runs locally through the Hugging Face Transformers pipeline.

The prompt instructs the model to:

Use only the retrieved documentation
Avoid guessing
Avoid adding external facts
Return a fallback message when the answer is not present

Fallback response:

I couldn't find that information in the documentation.
📚 Source Citations

Retrieved documents are returned along with their metadata.

For PDFs, the interface displays the corresponding page number when available.

Example:

📄 documentation.pdf — Page 4

This makes it easier to verify the generated answer against the original documentation.

⚙️ Configuration

The main RAG configuration is currently defined in the application code.

Important components include:

Embedding Model:
sentence-transformers/all-MiniLM-L6-v2


Generation Model:
Qwen/Qwen2.5-0.5B-Instruct


Vector Store:
FAISS


Similarity Threshold:
1.2
🔐 Environment Variables

The current application is designed to run locally and does not require an external LLM API key.

An .env.example file is included for future configuration.

Never commit real secrets to GitHub.

🧪 Testing

Run the application:

streamlit run app.py

Then test with:

1. Upload a documentation file
2. Ask a question that is answered by the document
3. Ask an unrelated question
4. Verify the fallback response
5. Verify source documents and page numbers
⚠️ Limitations
Small local language models may occasionally produce imperfect answers.
Retrieval quality depends on document chunking and embedding quality.
Very large documents may require additional optimization.
The application currently focuses on local document-based question answering.
Generated answers should be verified against the displayed sources for important use cases.
🔮 Future Improvements

Possible future improvements include:

 Conversation history
 Multiple document management
 Better hybrid retrieval
 Reranking retrieved chunks
 Streaming responses
 Improved source highlighting
 Document deletion/management
 Retrieval evaluation metrics
 Automated tests
 Docker support
 Production deployment configuration
📜 License

This project is licensed under the MIT License.

See LICENSE for details.

👨‍💻 Author

Your Name

GitHub: https://github.com/YOUR_USERNAME



Just replace:


```text
YOUR_USERNAME
Your Name
