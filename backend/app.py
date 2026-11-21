import asyncio
import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from graph import (
    InvalidToolApprovalError,
    PendingToolApprovalError,
    clear_thread,
    get_pending_tool_approval,
    resolve_tool_approval,
)
from config import CORS_ORIGINS, SESSION_IDLE_TTL
from graph import graph
from ingest import clear_session, ingest, last_seen, stores, touch
from pathlib import Path
from fastapi.staticfiles import StaticFiles
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

last_seen: dict[str, float] = {}   # session_id -> last activity timestamp

def touch(session_id: str):
    last_seen[session_id] = time.time()

# Helpers
async def _run_graph(session_id: str, mode: str, message: str) -> dict:
    """Invoke the LangGraph workflow; the graph routes on `mode`."""
    touch(session_id)
    graph_config = {"configurable": {"thread_id": f"{session_id}:{mode}"}}
    if mode == "general":
        current_state = await graph.aget_state(graph_config)
        if current_state.values.get("approval_id") and current_state.values.get("tool_calls"):
            raise PendingToolApprovalError("Approve or deny the pending tool call before sending another message.")

    approval_id = uuid.uuid4().hex if mode == "general" else None
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(message)],
            "mode": mode,
            "session_id": session_id,
            "answer": None,
            "approval_id": approval_id,
            "tool_calls": None,
        },
        config=graph_config,
    )
    return result





def _end_session(session_id: str):
    for cleanup, arg in (
        (clear_session, session_id),
        (clear_thread, f"{session_id}:general"),
        (clear_thread, f"{session_id}:document"),
        (clear_thread, f"{session_id}:general_agent"),
    ):
        try:
            cleanup(arg)
        except Exception:
            pass  

# Idle-session sweeper (safety net when the close beacon never arrives)
async def _sweeper():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for sid in [s for s, t in list(last_seen.items()) if now - t > SESSION_IDLE_TTL]:
            _end_session(sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_sweeper())
    yield
    task.cancel()


app = FastAPI(title="Dual-Mode RAG Chatbot", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str | None = None
    mode: str
    approval_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolApprovalRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    approval_id: str = Field(..., min_length=1)
    approved: bool


class PingRequest(BaseModel):
    session_id: str = Field(..., min_length=1)

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/new")
def new_session():
    """Optional: let the server issue a session id."""
    sid = str(uuid.uuid4())
    touch(sid)
    return {"session_id": sid}


@app.post("/build_index")
def build_index(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload one or more documents and build/extend this session's index."""
    indexed, rejected = [], []

    for f in files:
        filename = os.path.basename(f.filename or "")
        ext = os.path.splitext(filename)[1].lower()

        if not filename or ext not in ALLOWED_EXTENSIONS:
            rejected.append({"file": f.filename, "reason": "unsupported file type"})
            continue

        # Saved to a temp file; ingest() registers it for deletion on cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.file.read())
            tmp_path = tmp.name

        try:
            n_chunks = ingest(session_id, tmp_path, filename)
            indexed.append({"file": filename, "chunks": n_chunks})
        except Exception as exc:  # corrupt PDF, empty file, etc.
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            rejected.append({"file": filename, "reason": str(exc)})

    if not indexed:
        raise HTTPException(
            status_code=422,
            detail={"error": "No documents could be indexed", "rejected": rejected},
        )

    touch(session_id)
    return {"session_id": session_id, "indexed": indexed, "rejected": rejected}


@app.post("/chat", response_model=ChatResponse)
async def general_chat(req: ChatRequest):
    """General chat: plain LLM conversation, no documents involved."""
    try:
        result = await _run_graph(req.session_id, "general", req.message)
    except PendingToolApprovalError as exc:
        pending = await get_pending_tool_approval(req.session_id)
        if pending is None:
            raise HTTPException(status_code=409, detail=str(exc))
        return ChatResponse(
            mode="general",
            approval_id=pending["approval_id"],
            tool_calls=pending["tool_calls"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")
    return ChatResponse(
        answer=result.get("answer"),
        mode="general",
        approval_id=result.get("approval_id"),
        tool_calls=result.get("tool_calls"),
    )


@app.get("/chat/pending-approval", response_model=ChatResponse)
async def pending_tool_approval(session_id: str = Query(..., min_length=1)):
    """Restore a pending tool approval when the chat UI reconnects."""
    pending = await get_pending_tool_approval(session_id)
    if pending is None:
        return ChatResponse(mode="general")
    return ChatResponse(
        mode="general",
        approval_id=pending["approval_id"],
        tool_calls=pending["tool_calls"],
    )


@app.post("/docs_chat", response_model=ChatResponse)
async def docs_chat(req: ChatRequest):
    """Document chat: rewrite -> retrieve -> rerank -> topic check -> answer/reject."""
    if req.session_id not in stores:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed for this session. Upload files via /build_index first.",
        )
    try:
        result = await _run_graph(req.session_id, "document", req.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document chat failed: {exc}")
    return ChatResponse(answer=result["answer"], mode="document")


@app.post("/chat/tool-approval", response_model=ChatResponse)
async def tool_approval(req: ToolApprovalRequest):
    """Approve or decline a pending general-chat tool call."""
    try:
        answer = await resolve_tool_approval(
            req.session_id,
            req.approval_id,
            req.approved,
        )
    except InvalidToolApprovalError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tool approval failed: {exc}")
    return ChatResponse(answer=answer, mode="general")


@app.post("/session/ping")
def ping(req: PingRequest):
    """Keep-alive so active users are not expired by the TTL sweeper."""
    touch(req.session_id)
    return {"ok": True}


@app.post("/session/end")
async def end_session(request: Request):
    """
    Called on tab close. navigator.sendBeacon posts the session id as a
    text/plain body (or JSON if wrapped in a Blob), so accept both.
    """
    raw = (await request.body()).decode().strip()
    session_id = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            session_id = str(parsed.get("session_id", "")).strip()
    except json.JSONDecodeError:
        pass

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    _end_session(session_id)
    return {"cleared": True}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )