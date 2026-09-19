# 🔎 OmniSearch AI

**OmniSearch AI** is an agentic AI research assistant that combines **uploaded PDF documents, web search, conversational memory, and LLM-powered tool calling** into one intelligent system.

Instead of relying only on a fixed RAG pipeline, OmniSearch uses **LangGraph** to decide when it needs to search uploaded documents, search the web, retrieve document information, or answer directly from conversation context.

---

## ✨ Features

* 🤖 **Agentic AI workflow** using LangGraph
* 📄 **Multi-PDF document upload**
* 🔎 **Semantic document search** with ChromaDB
* 🌐 **Internet search** using DuckDuckGo
* 🧠 **Conversational memory** with SQLite checkpoints
* 📚 **Session-based document isolation**
* 📌 **Document-specific retrieval**
* 🆕 **Latest uploaded document detection**
* 🛠️ **LLM tool calling**
* ⚡ **Real-time token streaming**
* 🚀 **FastAPI backend**
* 🌐 **Web-based frontend**
* 🔐 Environment-variable based API key configuration
* 🧵 Unique conversation sessions using LangGraph thread IDs

---

## 🧠 How It Works

OmniSearch uses an agentic workflow rather than simply sending every question to a vector database.

```text
                    ┌─────────────────┐
                    │     User        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   FastAPI API   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LangGraph     │
                    │     Agent       │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 │           │           │
                 ▼           ▼           ▼
          ┌───────────┐ ┌──────────┐ ┌──────────────┐
          │ Document  │ │  Web     │ │ Conversation │
          │ Search    │ │ Search   │ │   Memory     │
          └─────┬─────┘ └────┬─────┘ └──────┬───────┘
                │             │              │
                ▼             ▼              ▼
          ┌───────────┐ ┌──────────┐ ┌──────────────┐
          │ ChromaDB  │ │DuckDuckGo│ │   SQLite     │
          └───────────┘ └──────────┘ └──────────────┘
                 \            |             /
                  \           |            /
                   └──────────┴───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Groq LLM     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Streaming Reply │
                    └─────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

* **Python**
* **FastAPI**
* **Uvicorn**

### AI / LLM

* **LangChain**
* **LangGraph**
* **Groq**
* **GPT-OSS-20B**

### RAG

* **ChromaDB**
* **Hugging Face Embeddings**
* **all-MiniLM-L6-v2**
* **PyPDF**
* **Recursive Character Text Splitter**

### Web Search

* **DuckDuckGo Search**

### Memory

* **LangGraph SQLite Checkpointer**
* **aiosqlite**

### Frontend

* HTML/CSS/JavaScript
* FastAPI-served frontend

---

## 🔧 Agent Tools

The agent currently has four tools.

### 1. Internet Search

Searches the web for current or external information.

```text
internet_search(query)
```

Useful for:

* Current events
* Recent information
* External research
* Information not contained in uploaded documents

---

### 2. Document Search

Performs semantic search over uploaded PDFs.

```text
document_search(
    query,
    document_id=None
)
```

The tool supports:

* Searching all documents in the current session
* Searching a specific document
* Session-based filtering
* Returning document names and page numbers

---

### 3. List Uploaded Documents

Returns the documents uploaded during the current session.

```text
list_uploaded_documents()
```

This allows the agent to identify available PDFs before searching them.

---

### 4. Get Latest Document

Identifies the most recently uploaded document.

```text
get_latest_document()
```

This is useful for queries such as:

```text
"What is the latest document about?"
"What does the last PDF say?"
"Summarize the newest document."
```

The agent can first identify the latest document and then perform retrieval specifically against it.

---

## 📄 RAG Pipeline

Uploaded PDFs follow this pipeline:

```text
PDF Upload
    │
    ▼
PyPDFLoader
    │
    ▼
Text Extraction
    │
    ▼
Recursive Character Splitter
    │
    ▼
Document Chunks
    │
    ▼
Hugging Face Embeddings
    │
    ▼
ChromaDB
```

Each chunk stores metadata including:

```text
session_id
document_id
source_doc
uploaded_at
page
```

This metadata allows OmniSearch to isolate documents between different users/sessions and perform document-specific searches.

---

## 🧠 Conversational Memory

OmniSearch uses **LangGraph's SQLite checkpointer** to maintain conversation state.

Each session receives a unique:

```text
thread_id
```

This allows the agent to understand follow-up questions such as:

```text
User:
What is this paper about?

Assistant:
...

User:
What methodology did they use?

Assistant:
...
```

The second question can be interpreted using the previous conversation context.

---

## ⚡ Streaming

Responses are streamed to the frontend using FastAPI's `StreamingResponse`.

The application uses LangGraph's:

```python
graph.astream(
    ...,
    stream_mode="messages"
)
```

This allows the user to see the response as it is generated instead of waiting for the complete response.

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

Add it to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
chroma_db/
uploads/
agent_checkpoints.db
```

---

## 📁 Project Structure

A typical project structure looks like:

```text
OmniSearch-AI/
│
├── app.py
├── index.html
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
├── .gitignore
│
├── uploads/
│
├── chroma_db/
│
└── agent_checkpoints.db
```

Runtime-generated directories and databases should **not** be committed to GitHub.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/omnisearch-ai.git

cd omnisearch-ai
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Start the application

```bash
uvicorn app:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

---

## 🐳 Docker

The application can also be containerized using Docker.

Build the image:

```bash
docker build -t omnisearch-ai .
```

Run:

```bash
docker run -p 8000:8000 --env-file .env omnisearch-ai
```

Or with Docker Compose:

```bash
docker compose up --build
```

---

## 🔌 API Endpoints

### `GET /`

Serves the web application.

### `POST /upload`

Upload one or multiple PDF documents.

### `GET /documents/{session_id}`

Returns documents associated with a session.

### `POST /reset_docs`

Deletes documents belonging to a session.

### `POST /chat`

Sends a message to the AI agent and streams the response.

Example request:

```json
{
  "session_id": "user-session-123",
  "query": "What is this document about?"
}
```

### `GET /health`

Returns application health information.

---

## 🎯 Example Use Cases

### Research Assistant

Upload multiple research papers and ask:

```text
Compare the methodologies used in these papers.
```

### Document Analysis

Upload a technical document and ask:

```text
What are the main findings?
```

### Latest Document

Upload several PDFs and ask:

```text
What is the latest document about?
```

The agent can identify the newest uploaded document before searching it.

### Web + Documents

Ask:

```text
Based on my uploaded paper, how does this compare
with the latest information available online?
```

The agent can combine information from the uploaded documents and web search.

---

## 🔒 Session Isolation

Documents are associated with a session ID.

For example:

```text
Session A
├── paper1.pdf
└── report.pdf

Session B
├── research.pdf
└── notes.pdf
```

A document search from Session A only searches documents belonging to Session A.

This prevents documents from different sessions from being mixed during retrieval.

---

## ⚙️ Configuration

### LLM

The application currently uses:

```python
ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)
```

### Embeddings

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Retrieval

```python
vectorstore.similarity_search(
    query,
    k=5,
    filter=search_filter,
)
```

The application retrieves the top 5 relevant chunks.

---

## 🧩 Why LangGraph?

Instead of implementing a fixed sequence such as:

```text
Question → Retrieval → LLM → Answer
```

OmniSearch uses an agentic loop:

```text
Question
   ↓
Agent
   ↓
Does the agent need a tool?
   │
   ├── No ──→ Answer
   │
   └── Yes
          ↓
        Tool
          ↓
        Result
          ↓
        Agent
          ↓
        Answer
```

This allows the model to decide which capability is appropriate for the user's request.

---

## 🚧 Future Improvements

Planned improvements include:

* [ ] Source citations in the UI
* [ ] Better document management
* [ ] Document deletion by individual file
* [ ] Hybrid search
* [ ] Reranking
* [ ] OCR support for scanned PDFs
* [ ] More web search providers
* [ ] Authentication
* [ ] User accounts
* [ ] Background document processing
* [ ] Docker deployment
* [ ] Production database
* [ ] Observability and tracing
* [ ] More agent tools

---

## 👨‍💻 Author

**Tarek Adel**

AI / ML / Generative AI Engineer

GitHub: [TarekAdel1](https://github.com/TarekAdel1)

---

## 📜 License

This project is available for educational and portfolio purposes.
