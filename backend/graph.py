import sys
from typing import Annotated, Literal, TypedDict
import os
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver, logger
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from langgraph.graph import END, START, StateGraph
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.graph.message import add_messages
from config import (
    GROQ_API_KEY,
    GROQ_MAIN_MODEL,
    RERANK_TOP_K,
    RETRIEVE_K,
    DATABASE_URL,
)
from ingest import stores
from citation import CitationExtractor
from rerank import rerank
from prompt import (
    GENERAL_SYSTEM_PROMPT,
    REJECTION_MESSAGE,
)
# Models
llm_main = ChatGroq(model=GROQ_MAIN_MODEL, api_key=GROQ_API_KEY, temperature=0.4, max_retries=3)
# llm_foundry=ChatOpenAI(
#     model="gpt-5",
#     base_url=os.getenv("PROJECT_ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     temperature=0.4,
#     max_retries=3,
# )
HISTORY_WINDOW = 12  # number of recent messages sent to the LLM


async def get_mcp_tool() -> list:
    """Start the news MCP server over stdio and return its tools."""
    client = MultiServerMCPClient(
        {
            "news": {
                "transport": "stdio",
                "command": os.environ.get("PYTHON_EXECUTABLE", sys.executable),
                "args": [os.path.join(os.path.dirname(__file__), "news_server.py")],
            }
        }
    )
    tools = await client.get_tools()
    return tools

# State
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    mode: Literal["general", "document"]
    session_id: str
    rewritten_query: str
    context_docs: list
    answer: str | None
    approval_id: str | None
    tool_calls: list[dict] | None
# Routing
def route_by_mode(state: State) -> str:
    return "general_chat" if state["mode"] == "general" else "rewrite_query"


def check_topic(state: State) -> str:
    """Generate from retrieved context; reject only when retrieval found nothing."""
    return "generate" if state.get("context_docs") else "reject"

# Nodes: general chat
async def general_chat(state: State):
    history = state["messages"][-HISTORY_WINDOW:]
    tools = await get_mcp_tool()
    agent = create_agent(
        tools=tools,
        model=llm_main,
        system_prompt=GENERAL_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )

    agent_config = {"configurable": {"thread_id": _agent_thread_id(state["session_id"])}}
    existing_state = agent.get_state(agent_config)
    if "tools" in existing_state.next:
        raise PendingToolApprovalError("A tool call is already waiting for approval.")
    if existing_state.values:
        clear_thread(_agent_thread_id(state["session_id"]))

    result = await agent.ainvoke({"messages": history}, config=agent_config)
    paused_state = agent.get_state(agent_config)
    if "tools" in paused_state.next:
        pending_message = paused_state.values["messages"][-1]
        return {
            "answer": None,
            "approval_id": state["approval_id"],
            "tool_calls": [
                {"id": call["id"], "name": call["name"], "args": call["args"]}
                for call in pending_message.tool_calls
            ],
        }

    answer = result["messages"][-1].content
    print(answer)
    clear_thread(_agent_thread_id(state["session_id"]))
    return {
        "answer": answer,
        "approval_id": None,
        "tool_calls": None,
        "messages": [AIMessage(answer)],
    }


class PendingToolApprovalError(Exception):
    """Raised when a new chat turn arrives while a tool call awaits approval."""


class InvalidToolApprovalError(Exception):
    """Raised when no matching tool approval is pending."""


async def get_pending_tool_approval(session_id: str) -> dict | None:
    """Return a pending general-chat approval, if one exists."""
    snapshot = await graph.aget_state(
        {"configurable": {"thread_id": f"{session_id}:general"}}
    )
    values = snapshot.values or {}
    approval_id = values.get("approval_id")
    tool_calls = values.get("tool_calls")
    if not approval_id or not tool_calls:
        return None
    return {"approval_id": approval_id, "tool_calls": tool_calls}


def _agent_thread_id(session_id: str) -> str:
    return f"{session_id}:general_agent"


async def resolve_tool_approval(
    session_id: str, approval_id: str, approved: bool
) -> str:
    tools = await get_mcp_tool()
    agent = create_agent(
        tools=tools,
        model=llm_main,
        system_prompt=GENERAL_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    agent_config = {"configurable": {"thread_id": _agent_thread_id(session_id)}}
    paused_state = agent.get_state(agent_config)
    if "tools" not in paused_state.next:
        raise InvalidToolApprovalError("There is no tool call waiting for approval.")

    conversation_config = {"configurable": {"thread_id": f"{session_id}:general"}}
    conversation_state = await graph.aget_state(conversation_config)
    if conversation_state.values.get("approval_id") != approval_id:
        raise InvalidToolApprovalError("This tool approval request is no longer active.")

    if approved:
        result = await agent.ainvoke(None, config=agent_config)
        answer = result["messages"][-1].content
    else:
        answer = "I didn't call the tool. You can ask me to answer without it."

    clear_thread(_agent_thread_id(session_id))
    await graph.aupdate_state(
        conversation_config,
        {
            "answer": answer,
            "approval_id": None,
            "tool_calls": None,
            "messages": [AIMessage(answer)],
        },
        as_node="general_chat",
    )
    return answer

# Nodes: document chat
def _format_history(messages: list) -> str:
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)




def get_database_url() -> str | None:
    database_url = DATABASE_URL

    if not database_url:
        return None

    parsed_url = urlsplit(database_url)
    query_params = parse_qs(parsed_url.query, keep_blank_values=True)

    if "sslmode" not in query_params:
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        sslmode = "disable" if parsed_url.hostname in loopback_hosts else "require"
        query_params["sslmode"] = [sslmode]
        database_url = urlunsplit(
            parsed_url._replace(query=urlencode(query_params, doseq=True))
        )

    return database_url



def get_checkpointer() -> InMemorySaver | PostgresSaver:
    """Create a checkpointer for the graph, using the database if available."""
    database_url = get_database_url()
    checkpointer= None
    if database_url:
      
        try:
            pool = ConnectionPool(
                conninfo=database_url,
                min_size=1,
                max_size=2,
                max_idle=240,                      # close idle conns before Neon does
                check=ConnectionPool.check_connection,  # test conn before handing it out
                timeout=15,                        # wait for Neon to wake up
                kwargs={
                    "autocommit": True,            # required by PostgresSaver
                    "row_factory": dict_row,       # required by PostgresSaver
                    "prepare_threshold": 0,        # needed behind Neon's pooler (PgBouncer)
                    "connect_timeout": 15,
                },
                open=False,
            )
            pool.open(wait=True, timeout=20)
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
        except Exception as exc:
            logger.warning(
                "PostgreSQL is unavailable (%s); using in-memory checkpoints. "
                "Conversation state will not survive an application restart.",
                type(exc).__name__,
            )
            checkpointer = None

    if checkpointer is None:
        checkpointer = InMemorySaver()


    return checkpointer    





def rewrite_query(state: State):
    """Turn a follow-up question into a standalone search query."""
    question = state["messages"][-1].content
    history = state["messages"][:-1][-HISTORY_WINDOW:]

    # First question in the thread: nothing to resolve, skip the LLM call
    if not history:
        return {"rewritten_query": question}

    prompt = (
        "Rewrite the user's last question as a  search query for document retrieval in details,"
        "using the chat history. Keep the key terms. Return ONLY the query.\n\n"
        f"Chat history:\n{_format_history(history)}\n\n"
        f"Last question: {question}\n\nStandalone query:"
    )
    rewritten = llm_main.invoke(prompt).content.strip().strip('"')
    print (f"Rewritten query: {rewritten}")
    return {"rewritten_query": rewritten or question}


def retrieve_and_rerank(state: State):
    """Retrieve candidates with hybrid search, then rerank with the cross-encoder."""

    store = stores.get(state["session_id"])
    if store is None:
        return {"context_docs": []}

    query = state["rewritten_query"]
    candidates = store.invoke(query)[:RETRIEVE_K]
    print(f"Retrieved {len(candidates)} candidates for query: {query}")
    if not candidates:
        return {"context_docs": []}

    ranked = rerank(query, candidates, top_k=RERANK_TOP_K)  # [(doc, score), ...]
    if not ranked:
        return {"context_docs": []}

    print(f"Top reranker score: {float(ranked[0][1]):.4f}")
    return {"context_docs": [doc for doc, _ in ranked]}

def generate(state: State):
    docs = state["context_docs"]

    context_blocks = []
    for d in docs:
        metadata = d.metadata or {}
        citation = CitationExtractor.format_citation({
            "paper_id": metadata.get("paper_id", ""),
            "section": metadata.get("section_path", ""),
            "page": metadata.get("page_num", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "node_type": metadata.get("node_type", ""),
        })
        context_blocks.append(
            f"[{metadata.get('source', 'unknown')} | {citation}]\n{d.page_content}"
        )

    context = "\n\n".join(context_blocks)
    question = state["messages"][-1].content
    history = state["messages"][:-1][-HISTORY_WINDOW:]

    prompt = (
        "Answer the question using ONLY the context below. If the context does "
        "not contain the answer, say you couldn't find it in the documents. "
        "Be concise, do not make things up, and do not add a Sources, Pages, or "
        "Sections footer; citation details are added separately.\n\n"
        f"Context:\n{context}\n\n"
        + (f"Chat history:\n{_format_history(history)}\n\n" if history else "")
        + f"Question: {question}"
    )
    answer = llm_main.invoke(prompt).content

    metadata = [d.metadata or {} for d in docs]
    sources = sorted({str(m.get("source", "unknown")) for m in metadata if m.get("source")})
    page_nums = sorted({str(m["page_num"]) for m in metadata if m.get("page_num")})
    section_paths = sorted({str(m["section_path"]) for m in metadata if m.get("section_path")})

    citation_lines = []
    if sources:
        citation_lines.append("Sources: " + ", ".join(sources))
    if page_nums:
        citation_lines.append("Pages: " + ", ".join(page_nums))
    if section_paths:
        citation_lines.append("Sections: " + ", ".join(section_paths))
    if citation_lines:
        answer += "\n\n[Citations]\n" + "\n".join(citation_lines)

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

builder.add_conditional_edges(
    START,
    route_by_mode,
    ["general_chat", "rewrite_query"],
)
builder.add_edge("rewrite_query", "retrieve")
builder.add_conditional_edges("retrieve", check_topic, ["generate", "reject"])
builder.add_edge("general_chat", END)
builder.add_edge("generate", END)
builder.add_edge("reject", END)

checkpointer = get_checkpointer()
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
