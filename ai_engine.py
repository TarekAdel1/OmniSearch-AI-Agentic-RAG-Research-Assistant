import os
import asyncio
import time

from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
)

from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.tools import DuckDuckGoSearchRun

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

CHROMA_DIR = BASE_DIR / "chroma_db"

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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