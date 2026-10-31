import os
import threading
import time
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

# Models (loaded once at import time, shared by all sessions)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Per-session state (in memory only; nothing is persisted to disk)
stores: dict[str, FAISS] = {}     # session_id -> FAISS index (holds the embeddings)
last_seen: dict[str, float] = {}  # session_id -> last activity timestamp
_lock = threading.Lock()


def touch(session_id: str):
    """Mark a session as active (used by the idle-session sweeper)."""
    last_seen[session_id] = time.time()

# Loading
def load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()
    if ext in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    raise ValueError(f"Unsupported file type: {ext}")


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def ingest(session_id: str, path: str, filename: str) -> int:
    """
    Load a file, chunk it, embed it, and add it to the session's index.
    Returns the number of chunks added. The temp file is deleted as soon as
    its text has been loaded, so uploads never linger on disk.
    """
    try:
        docs = load_file(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No readable text found (scanned PDFs need OCR before they can be indexed)"
        )

    for c in chunks:
        c.metadata["source"] = filename

    # Embed outside the lock (slow), then merge into the session's index
    new_store = FAISS.from_documents(chunks, embeddings)

    with _lock:
        if session_id in stores:
            stores[session_id].merge_from(new_store)
        else:
            stores[session_id] = new_store

    touch(session_id)
    return len(chunks)

# Cleanup
def clear_session(session_id: str):
    """Drop the session's index (and its embeddings) and stop tracking it."""
    with _lock:
        stores.pop(session_id, None)
    last_seen.pop(session_id, None)