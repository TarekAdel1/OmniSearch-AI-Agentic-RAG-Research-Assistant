# 🔎 OmniSearch AI

**OmniSearch AI** is an agentic research assistant that lets you upload PDF documents, search the web, and ask questions using natural language.

Instead of using only a traditional RAG pipeline, OmniSearch uses **LangGraph** to decide whether it should search your uploaded documents, search the web, use conversation memory, or answer directly.

## ✨ What Can It Do?

* 📄 Upload one or multiple PDF files
* 🔍 Search inside your documents using semantic search
* 🌐 Search the web for external or recent information
* 🤖 Use an AI agent to choose the right tool
* 🧠 Remember previous messages in the conversation
* 📚 Keep documents separated between sessions
* ⚡ Stream AI responses in real time
* 📌 Retrieve information from specific documents
* 🆕 Find and search your latest uploaded document

## 🧠 How It Works

```text
                    Your Question
                         │
                         ▼
                  ┌─────────────┐
                  │ AI Agent    │
                  │ LangGraph   │
                  └──────┬──────┘
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        PDF Search   Web Search   Memory
             │           │           │
             └───────────┼───────────┘
                         ▼
                     AI Answer
```

The agent decides which capability is needed for each question.

For example:

> **"What is this PDF about?"**

The agent can search your uploaded document.

> **"What is the latest information about this topic?"**

The agent can use web search.

> **"What methodology did they use?"**

The agent can use the previous conversation and search the relevant document.

---

# 🛠️ Tech Stack

### AI

* Python
* LangChain
* LangGraph
* Groq
* GPT-OSS-20B

### RAG

* ChromaDB
* Hugging Face Embeddings
* `all-MiniLM-L6-v2`
* PyPDF
* Recursive Character Text Splitter

### Backend

* FastAPI
* Uvicorn

### Frontend

* HTML
* CSS
* JavaScript

### Other

* Docker
* DuckDuckGo Search
* SQLite / LangGraph Checkpoints

---

# 📁 Project Structure

```text
OmniSearch-AI/
│
├── ai_engine.py          # AI agent, RAG, tools and LLM logic
├── backend.py            # FastAPI backend and API endpoints
├── index.html            # Web interface
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── README.md
```

---

# 🚀 Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/TarekAdel1/OmniSearch-AI-Agentic-RAG-Research-Assistant.git
cd OmniSearch-AI-Agentic-RAG-Research-Assistant
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add your API key

Create a `.env` file in the project folder:

```env
GROQ_API_KEY=your_groq_api_key
```

**Never upload your `.env` file to GitHub.**

## 5. Start the application

```bash
uvicorn backend:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 🐳 Run with Docker

Build the image:

```bash
docker build -t omnisearch-ai .
```

Run it:

```bash
docker run -p 8000:8000 --env-file .env omnisearch-ai
```

Or use Docker Compose:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 📖 How To Use

### 1. Upload PDFs

Upload one or more PDF documents through the web interface.

OmniSearch processes the documents and creates a searchable vector index.

### 2. Ask questions

You can ask questions about your documents using normal language.

Example:

```text
What is this document about?
```

```text
What are the main findings?
```

```text
Compare the methodologies used in these documents.
```

### 3. Ask follow-up questions

You can continue the conversation naturally:

```text
User:
What is this paper about?

AI:
...

User:
What methodology did they use?

AI:
...
```

The conversation memory allows OmniSearch to understand the context.

### 4. Search the web

You can also ask questions that require information from the internet:

```text
What are the latest developments in this field?
```

The agent can decide to use web search when appropriate.

### 5. Combine documents and web search

You can ask questions that require both your documents and external information:

```text
Compare the approach in my paper with the latest research available online.
```

---

# 🔧 Main API Endpoints

| Endpoint                  | Method | Description               |
| ------------------------- | ------ | ------------------------- |
| `/`                       | GET    | Web interface             |
| `/upload`                 | POST   | Upload PDF documents      |
| `/documents/{session_id}` | GET    | Get uploaded documents    |
| `/reset_docs`             | POST   | Remove session documents  |
| `/chat`                   | POST   | Send a question to the AI |
| `/health`                 | GET    | Check application status  |

---

# 🔐 Environment Variables

The application requires:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit API keys or `.env` files to GitHub.

Runtime files such as uploaded documents, vector databases, and checkpoints should also remain outside the repository.

---

# 🎯 Example Use Cases

### 📚 Research

Upload research papers and ask the AI to summarize or compare them.

### 📄 Document Analysis

Upload a technical or business document and ask questions about its contents.

### 🌐 Research + Web

Combine information from your uploaded documents with current information from the web.

### 🧠 Conversational Research

Ask follow-up questions without repeating the context every time.

---

# 👨‍💻 Author

**Tarek Adel**

AI / ML / Generative AI Engineer

GitHub:
https://github.com/TarekAdel1

---

## 📜 License

This project is available for educational and portfolio purposes.
