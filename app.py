import os
import uuid
import shutil
import asyncio
import time

from pathlib import Path
from typing import Annotated

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
)

from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
)

from pydantic import BaseModel

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
    InjectedState,
)

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
CHECKPOINT_DB = BASE_DIR / "agent_checkpoints.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CHROMA
# ============================================================

vectorstore = Chroma(
    collection_name="omnisearch_documents",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)


# ============================================================
# WEB SEARCH
# ============================================================

web_search = DuckDuckGoSearchRun()


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(MessagesState):
    session_id: str


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_text(content) -> str:
    """
    Safely extract text from LangChain streamed content.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text")

                if isinstance(text, str):
                    parts.append(text)

            elif hasattr(block, "text"):

                text = block.text

                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts)

    return ""


def get_session_documents(session_id: str):
    """
    Get document metadata belonging to one session.
    """

    results = vectorstore.get(
        where={
            "session_id": session_id
        }
    )

    metadatas = results.get("metadatas", [])

    documents = {}

    for metadata in metadatas:

        if not metadata:
            continue

        document_id = metadata.get("document_id")
        source_doc = metadata.get("source_doc")
        uploaded_at = metadata.get("uploaded_at")

        if not document_id or not source_doc:
            continue

        if document_id not in documents:

            documents[document_id] = {
                "document_id": document_id,
                "source_doc": source_doc,
                "uploaded_at": uploaded_at,
            }

    return list(documents.values())


# ============================================================
# TOOLS
# ============================================================

@tool
def internet_search(query: str) -> str:
    """
    Search the internet for current, external, or web-based
    information.

    Use this when the user asks about current events,
    recent information, public information from the web,
    or anything that requires an internet search.
    """

    start = time.perf_counter()

    print(f"[WEB] Searching: {query}")

    try:

        result = web_search.run(query)

        elapsed = time.perf_counter() - start

        print(f"[WEB] Finished in {elapsed:.2f}s")

        if not result:

            return "No useful web search results were found."

        return result

    except Exception as e:

        elapsed = time.perf_counter() - start

        print(
            f"[WEB] Failed after {elapsed:.2f}s: {e}"
        )

        return f"Web search failed: {str(e)}"


@tool
def list_uploaded_documents(
    state: Annotated[dict, InjectedState],
) -> str:
    """
    List all PDFs uploaded by the current session.

    Use this when the user asks what documents are uploaded,
    refers to a document by name, or you need to identify
    the latest/last uploaded document.
    """

    start = time.perf_counter()

    session_id = state["session_id"]

    print(f"[DOCS] Listing documents for {session_id}")

    try:

        documents = get_session_documents(session_id)

        elapsed = time.perf_counter() - start

        print(
            f"[DOCS] Listed {len(documents)} documents "
            f"in {elapsed:.2f}s"
        )

        if not documents:
            return "No documents have been uploaded."

        documents = sorted(
            documents,
            key=lambda x: x.get("uploaded_at") or "",
        )

        lines = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            lines.append(
                f"{index}. "
                f"{document['source_doc']} "
                f"(document_id: {document['document_id']})"
            )

        return "\n".join(lines)

    except Exception as e:

        print(f"[DOCS] Error: {e}")

        return f"Could not list documents: {str(e)}"


@tool
def get_latest_document(
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Identify the most recently uploaded PDF in the current
    session.

    Use this when the user says:
    - latest document
    - last document
    - final document
    - newest PDF
    - last uploaded PDF
    - this document

    Returns the document filename and document ID.
    """

    start = time.perf_counter()

    session_id = state["session_id"]

    print(
        f"[LATEST DOC] Finding latest document "
        f"for {session_id}"
    )

    try:

        documents = get_session_documents(session_id)

        if not documents:

            return "No documents have been uploaded."

        latest = max(
            documents,
            key=lambda x: x.get("uploaded_at") or "",
        )

        elapsed = time.perf_counter() - start

        print(
            f"[LATEST DOC] {latest['source_doc']} "
            f"found in {elapsed:.2f}s"
        )

        return (
            f"Latest uploaded document: "
            f"{latest['source_doc']}\n"
            f"Document ID: {latest['document_id']}"
        )

    except Exception as e:

        print(
            f"[LATEST DOC] Error: {e}"
        )

        return (
            f"Could not determine the latest document: "
            f"{str(e)}"
        )


@tool
async def document_search(
    query: str,
    state: Annotated[dict, InjectedState],
    document_id: str | None = None,
) -> str:
    """
    Search the user's uploaded PDFs.

    If document_id is provided, search ONLY that document.

    If document_id is not provided, search across all PDFs
    belonging to the current session.

    IMPORTANT:
    If the user refers to the latest, last, newest, final,
    or "this" document, first use get_latest_document and
    then search using the returned document ID.
    """

    start = time.perf_counter()

    session_id = state["session_id"]

    print(
        f"[RAG] Search started | "
        f"query={query!r} | "
        f"document_id={document_id}"
    )

    try:

        # ----------------------------------------------------
        # Build Chroma filter
        # ----------------------------------------------------

        if document_id:

            search_filter = {
                "$and": [
                    {
                        "session_id": session_id
                    },
                    {
                        "document_id": document_id
                    },
                ]
            }

        else:

            search_filter = {
                "session_id": session_id
            }

        # ----------------------------------------------------
        # Run Chroma in a worker thread
        # ----------------------------------------------------

        results = await asyncio.to_thread(
            vectorstore.similarity_search,
            query,
            k=5,
            filter=search_filter,
        )

        elapsed = time.perf_counter() - start

        print(
            f"[RAG] Chroma finished in "
            f"{elapsed:.2f}s | "
            f"results={len(results)}"
        )

        if not results:

            return (
                "No relevant information was found "
                "in the uploaded documents."
            )

        # ----------------------------------------------------
        # Format results
        # ----------------------------------------------------

        formatted = []

        max_chars_per_chunk = 3500

        for i, doc in enumerate(
            results,
            start=1,
        ):

            source = doc.metadata.get(
                "source_doc",
                "Unknown document",
            )

            page = doc.metadata.get(
                "page",
                "Unknown",
            )

            # PyPDFLoader pages are zero-indexed
            if isinstance(page, int):
                page = page + 1

            content = doc.page_content.strip()

            if len(content) > max_chars_per_chunk:

                content = (
                    content[:max_chars_per_chunk]
                    + "\n[Chunk truncated]"
                )

            formatted.append(
                f"Result {i}\n"
                f"Document: {source}\n"
                f"Page: {page}\n\n"
                f"{content}"
            )

        result_text = "\n\n---\n\n".join(
            formatted
        )

        total_elapsed = time.perf_counter() - start

        print(
            f"[RAG] Total search time: "
            f"{total_elapsed:.2f}s"
        )

        return result_text

    except Exception as e:

        elapsed = time.perf_counter() - start

        print(
            f"[RAG] Failed after "
            f"{elapsed:.2f}s: {e}"
        )

        return (
            f"Document search failed: "
            f"{str(e)}"
        )


# ============================================================
# TOOL LIST
# ============================================================

tools = [
    internet_search,
    document_search,
    list_uploaded_documents,
    get_latest_document,
]


# ============================================================
# LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are OmniSearch AI, an intelligent research assistant.

You have access to:

1. Conversation history.
2. The user's uploaded PDF documents.
3. Internet search.

============================================================
CONVERSATION
============================================================

- Always use conversation history to understand follow-up
  questions and references such as:
  "it", "this", "that", "him", "her", "the previous one",
  "what did you say?", etc.

- Treat the conversation as continuous.

- Do not ask the user to repeat information that already
  exists in the conversation.

============================================================
DOCUMENTS
============================================================

You have access to the user's uploaded PDF documents.

Use document_search when the answer may be inside a PDF.

If the user asks:

- "What is this document about?"
- "What is the latest document about?"
- "What is the last PDF about?"
- "Tell me about the final document."
- "What does the newest PDF say?"

You MUST first use get_latest_document.

Then use document_search with the returned document ID.

If the user explicitly names a PDF, use list_uploaded_documents
if necessary to identify its document ID, then search ONLY that
document.

If the user asks a general question about all uploaded PDFs,
you may search across all documents.

Never assume that the latest document is the same as an older
document.

When using document information, mention the document name and
page when possible.

============================================================
INTERNET
============================================================

Use internet_search when the question requires:

- current information
- recent information
- external information
- web research
- information that is not available in the uploaded PDFs

Never claim that you searched the web unless you actually used
internet_search.

============================================================
TOOL USAGE
============================================================

Do not use tools unnecessarily.

For a simple question that you already know, answer directly.

For questions requiring documents or current web information,
use the appropriate tool.

============================================================
ACCURACY
============================================================

Never invent information.

If the available information is insufficient, say so.

Clearly distinguish information obtained from:
- conversation
- uploaded documents
- internet search

============================================================
RESPONSE STYLE
============================================================

- Natural
- Direct
- Helpful
- Concise when possible
- Detailed when necessary
- Do not unnecessarily repeat the question
"""


# ============================================================
# AGENT NODE
# ============================================================

async def agent_node(
    state: AgentState,
):

    start = time.perf_counter()

    messages = state["messages"]

    print(
        f"[AGENT] Starting generation | "
        f"messages={len(messages)}"
    )

    system_message = SystemMessage(
        content=SYSTEM_PROMPT
    )

    response = await llm_with_tools.ainvoke(
        [
            system_message,
            *messages,
        ]
    )

    elapsed = time.perf_counter() - start

    tool_calls = getattr(
        response,
        "tool_calls",
        [],
    )

    print(
        f"[AGENT] Finished in {elapsed:.2f}s | "
        f"tool_calls={len(tool_calls)}"
    )

    if tool_calls:

        for call in tool_calls:

            print(
                f"[AGENT] Tool requested: "
                f"{call.get('name')}"
            )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

builder.add_edge(
    START,
    "agent",
)

builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge(
    "tools",
    "agent",
)


# ============================================================
# GLOBAL RUNTIME STATE
# ============================================================

checkpointer = None
graph = None

session_locks: dict[str, asyncio.Lock] = {}


def get_session_lock(
    session_id: str,
) -> asyncio.Lock:

    if session_id not in session_locks:

        session_locks[session_id] = asyncio.Lock()

    return session_locks[session_id]


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global checkpointer
    global graph

    async with AsyncSqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as saver:

        checkpointer = saver

        graph = builder.compile(
            checkpointer=checkpointer
        )

        print()
        print("=" * 60)
        print("OmniSearch AI started")
        print("LangGraph checkpointer: SQLite")
        print("Real token streaming: enabled")
        print("RAG: ChromaDB")
        print("Web search: DuckDuckGo")
        print("=" * 60)
        print()

        yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="OmniSearch AI",
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    query: str


class ResetRequest(BaseModel):
    session_id: str


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home():

    index_file = BASE_DIR / "index.html"

    if not index_file.exists():

        return HTMLResponse(
            """
            <h1>OmniSearch AI</h1>
            <p>index.html was not found.</p>
            """,
            status_code=404,
        )

    return HTMLResponse(
        index_file.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

@app.post("/upload")
async def upload_files(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):

    session_id = session_id.strip()

    if not session_id:

        return {
            "success": False,
            "error": "Session ID cannot be empty.",
        }

    session_dir = UPLOAD_DIR / session_id

    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    uploaded = []

    for uploaded_file in files:

        if not uploaded_file.filename:
            continue

        filename = Path(
            uploaded_file.filename
        ).name

        if not filename.lower().endswith(".pdf"):
            continue

        file_id = str(uuid.uuid4())

        safe_filename = (
            f"{file_id}_{filename}"
        )

        file_path = (
            session_dir /
            safe_filename
        )

        contents = await uploaded_file.read()

        file_path.write_bytes(contents)

        start = time.perf_counter()

        print()
        print(
            f"[UPLOAD] Processing: {filename}"
        )

        try:

            loader = PyPDFLoader(
                str(file_path)
            )

            documents = await asyncio.to_thread(
                loader.load
            )

            chunks = splitter.split_documents(
                documents
            )

            uploaded_at = str(
                time.time()
            )

            for chunk in chunks:

                chunk.metadata.update(
                    {
                        "session_id": session_id,
                        "document_id": file_id,
                        "source_doc": filename,
                        "uploaded_at": uploaded_at,
                    }
                )

            ids = [
                f"{file_id}_{i}"
                for i in range(len(chunks))
            ]

            if chunks:

                await asyncio.to_thread(
                    vectorstore.add_documents,
                    documents=chunks,
                    ids=ids,
                )

            elapsed = time.perf_counter() - start

            print(
                f"[UPLOAD] Finished: {filename} | "
                f"pages={len(documents)} | "
                f"chunks={len(chunks)} | "
                f"time={elapsed:.2f}s"
            )

            uploaded.append(
                {
                    "filename": filename,
                    "document_id": file_id,
                    "chunks": len(chunks),
                }
            )

        except Exception as e:

            print(
                f"[UPLOAD] Failed: {filename} | {e}"
            )

            if file_path.exists():
                file_path.unlink()

            return {
                "success": False,
                "error": str(e),
                "uploaded": uploaded,
            }

    return {
        "success": True,
        "session_id": session_id,
        "uploaded": uploaded,
    }


# ============================================================
# GET DOCUMENTS
# ============================================================

@app.get(
    "/documents/{session_id}"
)
async def get_documents(
    session_id: str,
):

    try:

        documents = await asyncio.to_thread(
            get_session_documents,
            session_id,
        )

        documents = sorted(
            documents,
            key=lambda x: x.get("uploaded_at") or "",
        )

        return {
            "success": True,
            "documents": [
                document["source_doc"]
                for document in documents
            ],
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# RESET DOCUMENTS
# ============================================================

@app.post(
    "/reset_docs"
)
async def reset_documents(
    request: ResetRequest,
):

    session_id = request.session_id

    try:

        results = await asyncio.to_thread(
            vectorstore.get,
            where={
                "session_id": session_id
            },
        )

        ids = results.get(
            "ids",
            [],
        )

        if ids:

            await asyncio.to_thread(
                vectorstore.delete,
                ids=ids,
            )

        session_dir = (
            UPLOAD_DIR /
            session_id
        )

        if session_dir.exists():

            shutil.rmtree(
                session_dir
            )

        print(
            f"[RESET] Documents removed "
            f"for session {session_id}"
        )

        return {
            "success": True,
            "message": "Documents reset successfully.",
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(
    request: ChatRequest,
):

    session_id = request.session_id.strip()
    message = request.query.strip()

    if not session_id:

        return {
            "error": "Session ID cannot be empty."
        }

    if not message:

        return {
            "error": "Query cannot be empty."
        }

    if graph is None:

        return {
            "error": "Graph is not initialized."
        }

    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    async def generate():

        lock = get_session_lock(
            session_id
        )

        async with lock:

            request_start = time.perf_counter()

            print()
            print("=" * 60)
            print(
                f"[CHAT] {message}"
            )
            print(
                f"[CHAT] session={session_id}"
            )
            print("=" * 60)

            try:

                async for chunk in graph.astream(
                    {
                        "messages": [
                            HumanMessage(
                                content=message
                            )
                        ],
                        "session_id": session_id,
                    },
                    config=config,
                    stream_mode="messages",
                ):

                    message_chunk, metadata = chunk

                    # Only stream text produced by the agent.
                    if metadata.get(
                        "langgraph_node"
                    ) != "agent":

                        continue

                    content = extract_text(
                        message_chunk.content
                    )

                    if not content:
                        continue

                    yield content

                total_time = (
                    time.perf_counter()
                    - request_start
                )

                print(
                    f"[CHAT] Completed in "
                    f"{total_time:.2f}s"
                )

                print("=" * 60)
                print()

            except asyncio.CancelledError:

                print(
                    f"[CHAT] Client disconnected: "
                    f"{session_id}"
                )

                raise

            except Exception as e:

                total_time = (
                    time.perf_counter()
                    - request_start
                )

                print()
                print("=" * 60)
                print("[CHAT] ERROR")
                print(
                    f"Type: {type(e).__name__}"
                )
                print(
                    f"Error: {e}"
                )
                print(
                    f"Time: {total_time:.2f}s"
                )
                print("=" * 60)
                print()

                yield (
                    "\n\n"
                    "⚠️ Error during processing: "
                    f"{str(e)}"
                )

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "graph_initialized": graph is not None,
        "checkpoint_db": str(CHECKPOINT_DB),
    }