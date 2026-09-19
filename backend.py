import os
import uuid
import shutil
import asyncio
import time

from pathlib import Path
from contextlib import asynccontextmanager

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
)

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_engine import (
    vectorstore,
    get_session_documents,
    builder,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
CHECKPOINT_DB = BASE_DIR / "agent_checkpoints.db"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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