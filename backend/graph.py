from typing import Annotated, Literal, TypedDict
from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from config import (
    GROQ_API_KEY,
    GROQ_MAIN_MODEL,
    REJECT_THRESHOLD,
    RERANK_TOP_K,
    RETRIEVE_K,
)
from ingest import stores
from rerank import rerank

# Models
llm_main = ChatGroq(model=GROQ_MAIN_MODEL, api_key=GROQ_API_KEY, temperature=0, max_retries=3)

HISTORY_WINDOW = 12  # number of recent messages sent to the LLM

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer the user's questions clearly."
)

REJECTION_MESSAGE = (
    "I can only answer questions about your uploaded documents, and I couldn't "
    "find anything relevant to that question. Try rephrasing it, or switch to "
    "General Chat."
)
# State
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    mode: Literal["general", "document"]
    session_id: str
    rewritten_query: str
    context_docs: list
    top_score: float
    answer: str
# Routing
def route_by_mode(state: State) -> str:
    return "general_chat" if state["mode"] == "general" else "rewrite_query"


def check_topic(state: State) -> str:
    """Topic rejection: if even the best reranked chunk scores too low, refuse."""
    return "generate" if state.get("top_score", float("-inf")) > REJECT_THRESHOLD else "reject"

# Nodes: general chat
def general_chat(state: State):
    history = state["messages"][-HISTORY_WINDOW:]
    resp = llm_main.invoke([SystemMessage(GENERAL_SYSTEM_PROMPT), *history])
    return {"answer": resp.content, "messages": [AIMessage(resp.content)]}

# Nodes: document chat
def _format_history(messages: list) -> str:
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def rewrite_query(state: State):
    """Turn a follow-up question into a standalone search query."""
    question = state["messages"][-1].content
    history = state["messages"][:-1][-HISTORY_WINDOW:]

    # First question in the thread: nothing to resolve, skip the LLM call
    if not history:
        return {"rewritten_query": question}

    prompt = (
        "Rewrite the user's last question as a standalone search query for "
        "searching a document collection. Resolve pronouns and references "
        "using the chat history. Keep the key terms. Return ONLY the query.\n\n"
        f"Chat history:\n{_format_history(history)}\n\n"
        f"Last question: {question}\n\nStandalone query:"
    )
    rewritten = llm_main.invoke(prompt).content.strip().strip('"')
    return {"rewritten_query": rewritten or question}


def retrieve_and_rerank(state: State):
    """Retrieve wide with the vector store, then rerank with the cross-encoder."""
    store = stores.get(state["session_id"])
    if store is None:
        return {"context_docs": [], "top_score": float("-inf")}

    query = state["rewritten_query"]
    candidates = store.similarity_search(query, k=RETRIEVE_K)
    if not candidates:
        return {"context_docs": [], "top_score": float("-inf")}

    ranked = rerank(query, candidates, top_k=RERANK_TOP_K)  # [(doc, score), ...]
    return {
        "context_docs": [doc for doc, _ in ranked],
        "top_score": float(ranked[0][1]),
    }


def generate(state: State):
    docs = state["context_docs"]
    context = "\n\n".join(f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)
    question = state["messages"][-1].content
    history = state["messages"][:-1][-HISTORY_WINDOW:]

    prompt = (
        "Answer the question using ONLY the context below. If the context does "
        "not contain the answer, say you couldn't find it in the documents. "
        "Be concise and do not make things up.\n\n"
        f"Context:\n{context}\n\n"
        + (f"Chat history:\n{_format_history(history)}\n\n" if history else "")
        + f"Question: {question}"
    )
    answer = llm_main.invoke(prompt).content

    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    if sources:
        answer += "\n\nSources: " + ", ".join(sources)

    return {"answer": answer, "messages": [AIMessage(answer)]}


def reject(state: State):
    return {"answer": REJECTION_MESSAGE, "messages": [AIMessage(REJECTION_MESSAGE)]}

# Graph
builder = StateGraph(State)

builder.add_node("general_chat", general_chat)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("retrieve", retrieve_and_rerank)
builder.add_node("generate", generate)
builder.add_node("reject", reject)

builder.add_conditional_edges(START, route_by_mode, ["general_chat", "rewrite_query"])
builder.add_edge("rewrite_query", "retrieve")
builder.add_conditional_edges("retrieve", check_topic, ["generate", "reject"])
builder.add_edge("general_chat", END)
builder.add_edge("generate", END)
builder.add_edge("reject", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Cleanup
def clear_thread(thread_id: str):
    """Delete a thread's chat memory (called when a session ends)."""
    try:
        checkpointer.delete_thread(thread_id)
    except AttributeError:
        # Older langgraph versions: remove the entries directly
        checkpointer.storage.pop(thread_id, None)
        for key in [k for k in checkpointer.writes if k[0] == thread_id]:
            checkpointer.writes.pop(key, None)