"""
LangGraph multi-agent StateGraph.

Architecture:
  START
    │
  [router_node]  ← classifies intent
    │
    ├─ hr_policy   → [hr_rag_node]      → END
    ├─ leave_*     → [hr_leave_node]    → END
    ├─ it_ticket   → [it_support_node]  → END
    ├─ it_asset    → [it_support_node]  → END
    └─ general     → [general_node]     → END

Memory: SqliteSaver (persists across server restarts, per employee_id thread)
"""
import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


from app.agents.router_agent import classify_intent
from app.agents.rag_agent import answer_policy_question
from app.agents.llm_factory import get_llm_with_fallback

logger = logging.getLogger(__name__)

# ── State definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # auto-appended message history
    user_id: int
    employee_id: str
    role: str
    department: str
    intent: str
    response: str
    sources: list
    confidence: float


# ── Node: Router ──────────────────────────────────────────────────────────────
def router_node(state: AgentState) -> dict:
    """Classify intent from the latest user message."""
    last_message = state["messages"][-1].content
    history = state.get("messages", [])
    previous_turns = []
    for msg in history[:-1]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        previous_turns.append(f"{role}: {msg.content}")
    context = "\n".join(previous_turns[-6:])  # keep prompt short and recent

    message_for_classification = (
        f"Previous conversation:\n{context}\n\nCurrent user message:\n{last_message}"
        if context else last_message
    )
    result = classify_intent(message_for_classification)
    logger.info(
        f"[Router] user={state.get('employee_id')} | "
        f"intent={result.get('intent')} | confidence={result.get('confidence')}"
    )
    return {
        "intent": result.get("intent", "general"),
        "confidence": result.get("confidence", 0.5)
    }


# ── Node: HR RAG (Policy Q&A) ─────────────────────────────────────────────────
def hr_rag_node(state: AgentState) -> dict:
    """Answer HR policy questions using the LangChain RAG chain."""
    question = state["messages"][-1].content
    messages = state.get("messages", [])

    # Resolve vague follow-ups like "elaborate more on that" using prior turn context.
    follow_up_markers = [
        "elaborate", "more", "that", "this", "explain", "details",
        "expand", "tell me more", "clarify"
    ]
    is_follow_up = any(marker in question.lower() for marker in follow_up_markers)
    if is_follow_up and len(messages) > 1:
        last_user_topic = ""
        last_assistant_answer = ""
        for msg in reversed(messages[:-1]):
            if not last_assistant_answer and isinstance(msg, AIMessage):
                last_assistant_answer = msg.content
            if isinstance(msg, HumanMessage):
                last_user_topic = msg.content
                break

        if last_user_topic:
            question = (
                "Use the previous topic to answer this follow-up.\n"
                f"Previous user topic: {last_user_topic}\n"
                f"Previous assistant answer: {last_assistant_answer}\n"
                f"Follow-up question: {state['messages'][-1].content}"
            )

    result = answer_policy_question(question, state.get("role", "employee"))
    ai_msg = AIMessage(content=result["answer"])
    return {
        "messages": [ai_msg],
        "response": result["answer"],
        "sources": result.get("sources", [])
    }


# ── Node: HR Leave ────────────────────────────────────────────────────────────
def hr_leave_node(state: AgentState) -> dict:
    """Guide users on leave-related actions using LLM."""
    message = state["messages"][-1].content
    intent = state.get("intent", "leave_balance")

    guidance = {
        "leave_balance":  ("check your leave balance",
                           "GET /hr/leave/balance"),
        "leave_apply":    ("apply for leave",
                           "POST /hr/leave/apply  (fields: leave_type, start_date, end_date, reason)"),
        "leave_status":   ("view your leave request history",
                           "GET /hr/leave/my-requests"),
        "leave_cancel":   ("cancel a leave request",
                           "Get your request_id from GET /hr/leave/my-requests, "
                           "then contact HR to cancel — cancellation endpoint coming soon"),
        "leave_approve":  ("approve or reject a leave request",
                           "POST /hr/leave/approve  (fields: request_id, action, notes)"),
    }

    action, endpoint = guidance.get(intent, ("manage leave", "GET /hr/leave/my-requests"))

    llm = get_llm_with_fallback(temperature=0.3, max_tokens=300)
    response = llm.invoke([
        SystemMessage(content=f"""You are a friendly HR Leave Management assistant.
The employee wants to {action}.
Guide them to use the endpoint: {endpoint}
Keep your response under 4 sentences. Be warm and helpful."""),
        HumanMessage(content=message)
    ])

    ai_msg = AIMessage(content=response.content)
    return {
        "messages": [ai_msg],
        "response": response.content,
        "sources": []
    }


# ── Node: IT Support ──────────────────────────────────────────────────────────
def it_support_node(state: AgentState) -> dict:
    """Guide users on IT support actions using LLM."""
    message = state["messages"][-1].content
    intent = state.get("intent", "it_ticket")

    guidance = {
        "it_ticket":        ("raise an IT support ticket",
                             "POST /it/tickets/create  (fields: issue_type, description)\n"
                             "Common issue types: VPN, Laptop, Email/Outlook, Printer, Network, Software"),
        "it_asset":         ("request an IT asset",
                             "POST /it/assets/request  (fields: asset_type, justification)\n"
                             "Available assets: Laptop, Monitor, Keyboard, Mouse, VPN Token, Software License"),
        "it_ticket_update": ("update a ticket status",
                             "PUT /it/tickets/update  (IT Team only — fields: ticket_id, status, resolution_notes)"),
    }

    action, endpoint = guidance.get(intent, ("get IT support", "POST /it/tickets/create"))

    llm = get_llm_with_fallback(temperature=0.3, max_tokens=300)
    response = llm.invoke([
        SystemMessage(content=f"""You are a helpful IT Support assistant.
The employee wants to {action}.
Guide them to use the endpoint: {endpoint}
Keep your response under 4 sentences. Be warm and concise."""),
        HumanMessage(content=message)
    ])

    ai_msg = AIMessage(content=response.content)
    return {
        "messages": [ai_msg],
        "response": response.content,
        "sources": []
    }


# ── Node: General ─────────────────────────────────────────────────────────────
def general_node(state: AgentState) -> dict:
    """Handle greetings and unclear queries."""
    message = state["messages"][-1].content
    role = state.get("role", "employee")
    name = state.get("employee_id", "there")

    llm = get_llm_with_fallback(temperature=0.5, max_tokens=200)
    response = llm.invoke([
        SystemMessage(content=f"""You are a friendly Enterprise HR & IT Copilot.
The user's role is: {role}.
You can help with:
  • HR policy questions — just ask anything about company policies
  • Leave management — apply, check balance, view or approve requests
  • IT support — raise tickets or request equipment

Greet them warmly and tell them what you can help with in 2-3 sentences."""),
        HumanMessage(content=message)
    ])

    ai_msg = AIMessage(content=response.content)
    return {
        "messages": [ai_msg],
        "response": response.content,
        "sources": []
    }


# ── Conditional routing ───────────────────────────────────────────────────────
def _route(state: AgentState) -> str:
    intent = state.get("intent", "general")
    if intent == "hr_policy":
        return "hr_rag"
    if intent in {"leave_balance", "leave_apply", "leave_status",
                  "leave_cancel", "leave_approve"}:
        return "hr_leave"
    if intent in {"it_ticket", "it_asset", "it_ticket_update"}:
        return "it_support"
    return "general"


# ── Graph builder ─────────────────────────────────────────────────────────────
_compiled_graph = None

def _build_graph():
    import os
    import sqlite3
    os.makedirs("db", exist_ok=True)

    # Create or connect to the database
    # LangGraph checkpoint writes can execute on worker threads,
    # so SQLite must allow cross-thread access for this shared connection.
    db_conn = sqlite3.connect("db/graph_memory.db", check_same_thread=False)
    memory = SqliteSaver(db_conn)

    g = StateGraph(AgentState)

    g.add_node("router",     router_node)
    g.add_node("hr_rag",     hr_rag_node)
    g.add_node("hr_leave",   hr_leave_node)
    g.add_node("it_support", it_support_node)
    g.add_node("general",    general_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", _route,
        {
            "hr_rag": "hr_rag",
            "hr_leave": "hr_leave",
            "it_support": "it_support",
            "general": "general"
        }
    )

    g.add_edge("hr_rag",     END)
    g.add_edge("hr_leave",   END)
    g.add_edge("it_support", END)
    g.add_edge("general",    END)

    return g.compile(checkpointer=memory)


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
        logger.info("LangGraph StateGraph compiled with SqliteSaver memory")
    return _compiled_graph


def invoke_graph(message: str, user_id: int, employee_id: str,
                 role: str, department: str = "") -> dict:
    """
    Invoke the graph for a user message.
    thread_id = employee_id → each employee gets their own conversation memory.
    """
    try:
        logger.info(f"invoke_graph: Starting for {employee_id}")
        graph = get_graph()
        logger.info(f"invoke_graph: Graph compiled successfully")

        config = {"configurable": {"thread_id": employee_id}}
        logger.info(f"invoke_graph: Config created with thread_id={employee_id}")

        logger.info(f"invoke_graph: About to call graph.invoke with message: {message[:50]}...")
        result = graph.invoke(
            {
                "messages":    [HumanMessage(content=message)],
                "user_id":     user_id,
                "employee_id": employee_id,
                "role":        role,
                "department":  department,
                "intent":      "",
                "response":    "",
                "sources":     [],
                "confidence":  0.0
            },
            config=config
        )
        logger.info(f"invoke_graph: Completed successfully")

        return {
            "response":   result.get("response", ""),
            "intent":     result.get("intent", "general"),
            "sources":    result.get("sources", []),
            "confidence": result.get("confidence", 0.0)
        }
    except Exception as e:
        logger.error(f"invoke_graph error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise