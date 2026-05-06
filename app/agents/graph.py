"""
LangGraph multi-agent StateGraph.
Key improvements:
  - History trimming: only last 10 messages sent to LLM (full history in SqliteSaver)
  - Conversational form-filling: leave application, IT ticket, asset request
  - Role-aware responses
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

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: int        # integer DB id — used directly in tool calls inside nodes
    employee_id: str
    role: str
    department: str
    intent: str
    response: str
    sources: list
    confidence: float


# ── History trimming ──────────────────────────────────────────────────────────
def _trim(messages: list, n: int = 10) -> list:
    """
    Only send the last n messages to the LLM.
    The COMPLETE history lives in SqliteSaver — this only limits LLM input cost.
    """
    return messages[-n:] if len(messages) > n else messages


def _history_str(messages: list, n: int = 8) -> str:
    """Readable conversation history for form-filling prompts."""
    result = []
    for msg in _trim(messages, n):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        text = msg.content[:300] + "…" if len(msg.content) > 300 else msg.content
        result.append(f"{role}: {text}")
    return "\n".join(result)


# ── Node: Router ──────────────────────────────────────────────────────────────
def router_node(state: AgentState) -> dict:
    last = state["messages"][-1].content
    trimmed = _trim(state["messages"], 6)
    ctx_lines = [
        f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {m.content[:150]}"
        for m in trimmed[:-1]
    ]
    classify_input = (
        f"Recent context:\n{chr(10).join(ctx_lines)}\n\nCurrent: {last}"
        if ctx_lines else last
    )
    result = classify_intent(classify_input)
    logger.info(f"[Router] {state.get('employee_id')} → {result.get('intent')} ({result.get('confidence')})")
    return {"intent": result.get("intent", "general"), "confidence": result.get("confidence", 0.5)}


# ── Node: HR RAG ──────────────────────────────────────────────────────────────
def hr_rag_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    trimmed = _trim(state["messages"])

    # Resolve follow-ups using recent context only
    follow_ups = {"elaborate", "more", "that", "this", "explain", "details", "expand", "clarify"}
    if any(w in question.lower() for w in follow_ups) and len(trimmed) > 1:
        last_topic = last_answer = ""
        for msg in reversed(trimmed[:-1]):
            if not last_answer and isinstance(msg, AIMessage):
                last_answer = msg.content[:400]
            if isinstance(msg, HumanMessage):
                last_topic = msg.content
                break
        if last_topic:
            question = f"Follow-up.\nPrevious topic: {last_topic}\nPrevious answer: {last_answer}\nFollow-up: {state['messages'][-1].content}"

    result = answer_policy_question(question, state.get("role", "employee"))
    return {
        "messages": [AIMessage(content=result["answer"])],
        "response": result["answer"],
        "sources": result.get("sources", [])
    }


# ── Conversational form prompts ───────────────────────────────────────────────
_LEAVE_FORM = """You are an HR assistant collecting leave application details.

Conversation so far:
{history}

Collect: leave_type (casual|sick|privilege|maternity|paternity), start_date (YYYY-MM-DD),
end_date (YYYY-MM-DD), reason (optional).

Rules:
- Scan the conversation for fields already provided
- Ask for ONE missing field at a time, naturally
- Convert natural language dates (e.g. "next Monday", "June 15") to YYYY-MM-DD
- If they say "3 days", compute end_date from start_date
- When leave_type + start_date + end_date are all confirmed, output EXACTLY:
  SUBMIT|leave_type|start_date|end_date|reason
  (use empty string for reason if not given. No other text.)

Current message: {msg}"""

_TICKET_FORM = """You are an IT Support assistant raising a support ticket.

Conversation so far:
{history}

Collect: issue_type (VPN|Laptop|Email/Outlook|Printer|Network|Software|Monitor or similar),
description (clear explanation of the problem, at least one sentence).

Rules:
- Scan conversation for what's been provided
- If both fields are clear, output EXACTLY:
  SUBMIT|issue_type|description
  (no other text)
- Otherwise ask for the missing info naturally, one field at a time

Current message: {msg}"""

_ASSET_FORM = """You are an IT assistant processing an asset request.

Conversation so far:
{history}

Collect: asset_type (Laptop|Monitor|Keyboard|Mouse|VPN Token|Software License),
justification (business reason, at least one sentence).

Rules:
- Scan conversation for what's been provided
- If both fields are clear, output EXACTLY:
  SUBMIT|asset_type|justification
  (no other text)
- Otherwise ask naturally

Current message: {msg}"""


# ── Conversational executors ──────────────────────────────────────────────────
def _run_leave_form(state: AgentState) -> dict:
    llm = get_llm_with_fallback(temperature=0.2, max_tokens=200)
    resp = llm.invoke([SystemMessage(content=_LEAVE_FORM.format(
        history=_history_str(state["messages"]),
        msg=state["messages"][-1].content
    ))]).content.strip()

    if resp.startswith("SUBMIT|"):
        parts = resp.split("|")
        if len(parts) >= 4:
            leave_type, start_date, end_date = parts[1].strip(), parts[2].strip(), parts[3].strip()
            reason = parts[4].strip() if len(parts) > 4 else ""

            from app.database import SessionLocal
            from app.tools.leave_tools import apply_leave as _apply
            from fastapi import HTTPException
            db = SessionLocal()
            try:
                r = _apply(state["user_id"], leave_type, start_date, end_date, reason, db)
                msg = (
                    f"✅ **Leave request submitted!**\n\n"
                    f"| Field | Value |\n|---|---|\n"
                    f"| Request ID | `{r['request_id']}` |\n"
                    f"| Type | {leave_type.title()} |\n"
                    f"| Dates | {start_date} → {end_date} |\n"
                    f"| Working days | {r['num_days']} |\n"
                    f"| Status | {r['status'].title()} |\n\n"
                    f"Your manager will be notified if approval is required."
                )
            except HTTPException as e:
                msg = f"❌ **Could not submit:** {e.detail}\n\nPlease check the details and try again."
            except Exception as e:
                msg = f"❌ Error: {str(e)}"
            finally:
                db.close()
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}

    return {"messages": [AIMessage(content=resp)], "response": resp, "sources": []}


def _run_ticket_form(state: AgentState) -> dict:
    llm = get_llm_with_fallback(temperature=0.2, max_tokens=200)
    resp = llm.invoke([SystemMessage(content=_TICKET_FORM.format(
        history=_history_str(state["messages"]),
        msg=state["messages"][-1].content
    ))]).content.strip()

    if resp.startswith("SUBMIT|"):
        parts = resp.split("|", 2)
        if len(parts) == 3:
            issue_type, description = parts[1].strip(), parts[2].strip()
            from app.database import SessionLocal
            from app.tools.ticket_tools import create_ticket as _create
            db = SessionLocal()
            try:
                r = _create(state["user_id"], issue_type, description, db)
                if r.get("ticket_created"):
                    msg = (
                        f"✅ **IT Ticket created!**\n\n"
                        f"| Field | Value |\n|---|---|\n"
                        f"| Ticket ID | `{r['ticket_id']}` |\n"
                        f"| Issue | {issue_type} |\n"
                        f"| Priority | {r['priority'].title()} |\n"
                        f"| Status | Open |\n\n"
                        f"The IT team will be in touch shortly."
                    )
                else:
                    msg = f"ℹ️ {r.get('message', 'Unable to create ticket.')}"
            except Exception as e:
                msg = f"❌ Error: {str(e)}"
            finally:
                db.close()
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}

    return {"messages": [AIMessage(content=resp)], "response": resp, "sources": []}


def _run_asset_form(state: AgentState) -> dict:
    llm = get_llm_with_fallback(temperature=0.2, max_tokens=200)
    resp = llm.invoke([SystemMessage(content=_ASSET_FORM.format(
        history=_history_str(state["messages"]),
        msg=state["messages"][-1].content
    ))]).content.strip()

    if resp.startswith("SUBMIT|"):
        parts = resp.split("|", 2)
        if len(parts) == 3:
            asset_type, justification = parts[1].strip(), parts[2].strip()
            from app.database import SessionLocal
            from app.models import AssetRequest, AssetStatusEnum
            import uuid
            db = SessionLocal()
            try:
                req_id = f"AST{uuid.uuid4().hex[:8].upper()}"
                asset = AssetRequest(
                    request_id=req_id, requester_id=state["user_id"],
                    asset_type=asset_type, justification=justification,
                    status=AssetStatusEnum.pending_manager
                )
                db.add(asset)
                db.commit()
                msg = (
                    f"✅ **Asset request submitted!**\n\n"
                    f"| Field | Value |\n|---|---|\n"
                    f"| Request ID | `{req_id}` |\n"
                    f"| Asset | {asset_type} |\n"
                    f"| Status | Pending Manager Approval |\n\n"
                    f"Your manager will review this request."
                )
            except Exception as e:
                msg = f"❌ Error: {str(e)}"
            finally:
                db.close()
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}

    return {"messages": [AIMessage(content=resp)], "response": resp, "sources": []}


# ── Node: HR Leave ────────────────────────────────────────────────────────────
def hr_leave_node(state: AgentState) -> dict:
    intent = state.get("intent", "leave_balance")
    if intent == "leave_apply":
        return _run_leave_form(state)

    # For other leave intents, use LLM with trimmed context
    guidance = {
        "leave_balance":  "Check balance: GET /hr/leave/balance",
        "leave_status":   "View requests: GET /hr/leave/my-requests",
        "leave_cancel":   "Cancel leave: POST /hr/leave/cancel?request_id=XXXX",
        "leave_approve":  "Approve/reject: POST /hr/leave/approve (request_id, action, notes)",
    }
    hint = guidance.get(intent, "GET /hr/leave/my-requests")
    llm = get_llm_with_fallback(temperature=0.3, max_tokens=250)
    resp = llm.invoke([
        SystemMessage(content=f"HR Leave assistant. Hint: {hint}. "
                              f"Context: {_history_str(state['messages'], 4)}. "
                              f"Be warm, brief (2-3 sentences)."),
        HumanMessage(content=state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=resp.content)], "response": resp.content, "sources": []}


# ── Node: IT Support ──────────────────────────────────────────────────────────
def it_support_node(state: AgentState) -> dict:
    intent = state.get("intent", "it_ticket")
    if intent == "it_ticket":
        return _run_ticket_form(state)
    if intent == "it_asset":
        return _run_asset_form(state)

    llm = get_llm_with_fallback(temperature=0.3, max_tokens=250)
    resp = llm.invoke([
        SystemMessage(content=f"IT Support assistant. "
                              f"Context: {_history_str(state['messages'], 4)}. "
                              f"For ticket update: PUT /it/tickets/update. Be brief."),
        HumanMessage(content=state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=resp.content)], "response": resp.content, "sources": []}


# ── Node: General ─────────────────────────────────────────────────────────────
def general_node(state: AgentState) -> dict:
    role = state.get("role", "employee")
    llm = get_llm_with_fallback(temperature=0.5, max_tokens=200)
    resp = llm.invoke([
        SystemMessage(content=f"Enterprise HR & IT Copilot. User role: {role}. "
                              f"Help with policies, leave, IT support, assets. "
                              f"Context: {_history_str(state['messages'], 4)}. "
                              f"Be warm, max 3 sentences."),
        HumanMessage(content=state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=resp.content)], "response": resp.content, "sources": []}


# ── Routing ───────────────────────────────────────────────────────────────────
def _route(state: AgentState) -> str:
    intent = state.get("intent", "general")
    if intent == "hr_policy":                                          return "hr_rag"
    if intent in {"leave_balance","leave_apply","leave_status",
                  "leave_cancel","leave_approve"}:                     return "hr_leave"
    if intent in {"it_ticket","it_asset","it_ticket_update"}:          return "it_support"
    return "general"


# ── Graph ─────────────────────────────────────────────────────────────────────
_compiled_graph = None

def _build_graph():
    import os, sqlite3
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/graph_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    g = StateGraph(AgentState)
    g.add_node("router",     router_node)
    g.add_node("hr_rag",     hr_rag_node)
    g.add_node("hr_leave",   hr_leave_node)
    g.add_node("it_support", it_support_node)
    g.add_node("general",    general_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route,
        {"hr_rag":"hr_rag","hr_leave":"hr_leave","it_support":"it_support","general":"general"})
    g.add_edge("hr_rag",     END)
    g.add_edge("hr_leave",   END)
    g.add_edge("it_support", END)
    g.add_edge("general",    END)

    compiled = g.compile(checkpointer=memory)
    logger.info("LangGraph compiled — LLM receives last 10 msgs, full history in checkpoint")
    return compiled


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


def invoke_graph(message: str, user_id: int, employee_id: str,
                 role: str, department: str = "") -> dict:
    try:
        result = get_graph().invoke(
            {"messages": [HumanMessage(content=message)],
             "user_id": user_id, "employee_id": employee_id,
             "role": role, "department": department,
             "intent": "", "response": "", "sources": [], "confidence": 0.0},
            config={"configurable": {"thread_id": employee_id}}
        )
        return {
            "response":   result.get("response", ""),
            "intent":     result.get("intent", "general"),
            "sources":    result.get("sources", []),
            "confidence": result.get("confidence", 0.0)
        }
    except Exception as e:
        logger.error(f"invoke_graph error: {type(e).__name__}: {e}", exc_info=True)
        raise