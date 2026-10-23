
from __future__ import annotations

import os
import sqlite3
import tempfile
import hashlib
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# --------------------------------------------------
# 1. Environment
# --------------------------------------------------

load_dotenv(find_dotenv())

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "hybrid-search-index",
)
PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1",
)

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing from your .env file."
    )


# --------------------------------------------------
# 2. LLM + embeddings
# --------------------------------------------------

llm = ChatGroq(model="openai/gpt-oss-120b")


embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

bm25_encoder = BM25Encoder().default()


# --------------------------------------------------
# 3. Pinecone hybrid index
# --------------------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [
    item.name for item in pc.list_indexes()
]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_REGION,
        ),
    )

index = pc.Index(PINECONE_INDEX_NAME)


# --------------------------------------------------
# 4. PDF ingestion
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """Extract and index any uploaded PDF using hybrid search."""

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    filename = filename or "uploaded.pdf"
    namespace = str(thread_id)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        docs = PyPDFLoader(temp_path).load()
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        texts = [doc.page_content for doc in chunks]
        metadatas = [
            {
                "source": filename,
                "page": int(doc.metadata.get("page", 0)),
                "thread_id": namespace,
                # PineconeHybridSearchRetriever pops text_key ("context" by default)
                # from metadata on retrieval to reconstruct Document.page_content.
                "context": doc.page_content,
            }
            for doc in chunks
        ]

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        ids = [
            hashlib.sha256(
                (
                    f"{namespace}:{file_hash}:"
                    f"{i}:{hashlib.sha256(doc.page_content.encode()).hexdigest()}"
                ).encode()
            ).hexdigest()
            for i, doc in enumerate(chunks)
        ]

        # Build dense + sparse vectors and upsert with explicit namespace.
        # PineconeHybridSearchRetriever.add_texts() ignores the namespace
        # parameter — so we upsert directly to the raw Pinecone index.
        dense_vectors = embed_model.embed_documents(texts)
        sparse_vectors = bm25_encoder.encode_documents(texts)

        pinecone_vectors = []
        for vec_id, dense, sparse, meta in zip(
            ids, dense_vectors, sparse_vectors, metadatas
        ):
            pinecone_vectors.append({
                "id": vec_id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": meta,
            })

        # Upsert in batches of 100
        batch_size = 100
        for start in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[start : start + batch_size]
            index.upsert(vectors=batch, namespace=namespace)

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_has_document(thread_id: str) -> bool:
    """Check whether a conversation namespace contains vectors."""

    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    namespace = str(thread_id)
    namespace_stats = namespaces.get(namespace)

    if namespace_stats is None:
        return False

    # Pinecone SDK returns NamespaceSummary objects or plain dicts
    count = (
        namespace_stats.get("vector_count", 0)
        if isinstance(namespace_stats, dict)
        else getattr(namespace_stats, "vector_count", 0)
    )
    return count > 0


# --------------------------------------------------
# 5. Tools
# --------------------------------------------------

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """Perform addition, subtraction, multiplication or division."""

    if operation == "add":
        result = first_num + second_num

    elif operation == "sub":
        result = first_num - second_num

    elif operation == "mul":
        result = first_num * second_num

    elif operation == "div":
        if second_num == 0:
            return {
                "error": "Division by zero is not allowed."
            }

        result = first_num / second_num

    else:
        return {
            "error": f"Unsupported operation: {operation}"
        }

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def rag_tool(
    query: str,
    config: RunnableConfig,
) -> dict:
    """Search and retrieve relevant text chunks from the uploaded PDF document for this conversation.
    Call this tool whenever the user asks questions about concepts, definitions, queries, topics, or facts from their document."""

    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")

    if not thread_id:
        return {
            "error": "Cannot identify the current conversation."
        }

    namespace = str(thread_id)

    if not thread_has_document(namespace):
        return {
            "error": (
                "No PDF is indexed in this conversation. "
                "Please upload a PDF first."
            ),
            "query": query,
        }

    dense_vec = embed_model.embed_query(query)
    sparse_vec = bm25_encoder.encode_queries(query)
    dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, alpha=0.5)
    sparse_vec["values"] = [float(val) for val in sparse_vec["values"]]

    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=4,
        include_metadata=True,
        namespace=namespace,
    )

    matches = (
        results.get("matches", [])
        if isinstance(results, dict)
        else getattr(results, "matches", [])
    )

    if not matches:
        return {
            "query": query,
            "context": [],
            "message": "No relevant passages were found in the document.",
        }

    contexts = []
    for match in matches:
        meta = (
            match.get("metadata")
            if isinstance(match, dict)
            else getattr(match, "metadata", {})
        ) or {}
        content = meta.get("context") or meta.get("text") or ""
        contexts.append(
            {
                "content": content,
                "source": meta.get("source", "PDF"),
                "page": meta.get("page", "unknown"),
            }
        )

    valid_contexts = [c for c in contexts if c["content"]]
    if not valid_contexts:
        return {
            "query": query,
            "context": [],
            "message": (
                "Vectors were found for this conversation, but text content was missing from metadata. "
                "Please re-upload the PDF to re-index it properly."
            ),
        }

    return {
        "query": query,
        "context": valid_contexts,
    }


# Stock-price tool removed
tools = [
    search_tool,
    calculator,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# 6. State
# --------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------
# 7. Chat node
# --------------------------------------------------

def chat_node(state: ChatState, config=None):
    """Call the LLM, which may answer or request a tool."""

    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id")
    has_pdf = thread_has_document(thread_id) if thread_id else False

    if has_pdf:
        doc_instructions = (
            "A PDF document IS CURRENTLY UPLOADED AND INDEXED in the vector database for this conversation.\n"
            "CRITICAL: For ANY question about concepts, terms, definitions, explanations, or facts that could be answered by the document, "
            "YOU MUST ALWAYS call the 'rag_tool' tool FIRST to retrieve the relevant passages.\n"
            "Do NOT answer from general knowledge without searching the document first.\n"
            "Do NOT claim 'no PDF is uploaded' because a document is uploaded and ready in rag_tool."
        )
    else:
        doc_instructions = (
            "No PDF document is currently uploaded in this conversation. "
            "If the user asks questions about an uploaded document, explain that they should upload a PDF first."
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful and knowledgeable assistant.\n\n"
            f"{doc_instructions}\n\n"
            "Guidelines:\n"
            "1. When answering from retrieved passages, cite the source filename and page number.\n"
            "2. If rag_tool returns passages, synthesize a comprehensive answer strictly based on those passages.\n"
            "3. If rag_tool returns no relevant passages, clearly inform the user.\n"
            "4. Use web search for current real-world events and calculator for mathematical calculations."
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)


# --------------------------------------------------
# 8. SQLite checkpointer
# --------------------------------------------------

DB_PATH = Path(__file__).resolve().parent / "chatbot.db"

conn = sqlite3.connect(
    database=str(DB_PATH),
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=conn)


# --------------------------------------------------
# 9. LangGraph
# --------------------------------------------------

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer
)


# --------------------------------------------------
# 10. Conversation helpers
# --------------------------------------------------

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = (
            checkpoint.config["configurable"]["thread_id"]
        )
        all_threads.add(thread_id)

    return list(all_threads)


def thread_document_metadata(thread_id: str) -> dict:
    """Return vector-count information for a conversation."""

    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    namespace_stats = namespaces.get(str(thread_id))

    if not namespace_stats:
        return {}

    count = (
        namespace_stats.get("vector_count", 0)
        if isinstance(namespace_stats, dict)
        else getattr(namespace_stats, "vector_count", 0)
    )

    return {
        "thread_id": str(thread_id),
        "chunks": count,
    }
