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

# ── Date/Time Helper ────────────────────────────────────────────────────────────
def _get_current_time_ist():
    """Get current date and time in IST format."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ist = ZoneInfo('Asia/Kolkata')
    now = datetime.now(ist)
    return now.strftime("%A, %d %B %Y, %I:%M %p IST")

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
Today is {current_date}.

Conversation so far:
{history}

Collect: leave_type (casual|sick|privilege|maternity|paternity), start_date (YYYY-MM-DD),
end_date (YYYY-MM-DD), reason (optional).

Rules:
- Scan the conversation for fields already provided
- Ask for ONE missing field at a time, naturally
- Convert natural language dates (e.g. "next Monday", "June 15") to YYYY-MM-DD
- If they say "3 days", compute end_date from start_date
- If user says "cancel", "never mind", "changed my mind", or "don't want to apply", reply: CANCEL
- When leave_type + start_date + end_date are all confirmed, output EXACTLY:
  SUBMIT|leave_type|start_date|end_date|reason
  (use empty string for reason if not given. No other text.)

Current message: {msg}"""

_TICKET_FORM = """You are an IT Support assistant raising a support ticket.
Today is {current_date}.

Conversation so far:
{history}

Collect: issue_type (VPN|Laptop|Email/Outlook|Printer|Network|Software|Monitor or similar),
description (clear explanation of the problem, at least one sentence).

Rules:
- Scan conversation for what's been provided
- If user says "cancel", "never mind", or "changed my mind", reply: CANCEL
- If both fields are clear, output EXACTLY:
  SUBMIT|issue_type|description
  (no other text)
- Otherwise ask for the missing info naturally, one field at a time

Current message: {msg}"""

_ASSET_FORM = """You are an IT assistant processing an asset request.
Today is {current_date}.

Conversation so far:
{history}

Collect: asset_type (Laptop|Monitor|Keyboard|Mouse|VPN Token|Software License),
justification (business reason, at least one sentence).

Rules:
- Scan conversation for what's been provided
- If user says "cancel", "never mind", or "changed my mind", reply: CANCEL
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
        msg=state["messages"][-1].content,
        current_date=_get_current_time_ist()
    ))]).content.strip()

    # Handle cancellation
    if resp.strip().upper() == "CANCEL" or "CANCEL" in resp.strip().upper():
        cancel_msg = "No problem! I've cancelled the leave application. Feel free to ask if you need help with anything else."
        return {"messages": [AIMessage(content=cancel_msg)], "response": cancel_msg, "sources": []}

    # Extract SUBMIT line from response (handle extra text before/after)
    submit_line = None
    for line in resp.split('\n'):
        line = line.strip()
        if line.startswith("SUBMIT|"):
            submit_line = line
            break

    if submit_line:
        parts = submit_line.split("|")
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
        msg=state["messages"][-1].content,
        current_date=_get_current_time_ist()
    ))]).content.strip()

    # Handle cancellation
    if resp.strip().upper() == "CANCEL" or "CANCEL" in resp.strip().upper():
        cancel_msg = "No problem! I've cancelled the ticket. Let me know if you need help with anything else."
        return {"messages": [AIMessage(content=cancel_msg)], "response": cancel_msg, "sources": []}

    # Extract SUBMIT line from response (handle extra text before/after)
    submit_line = None
    for line in resp.split('\n'):
        line = line.strip()
        if line.startswith("SUBMIT|"):
            submit_line = line
            break

    if submit_line:
        parts = submit_line.split("|", 2)
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
        msg=state["messages"][-1].content,
        current_date=_get_current_time_ist()
    ))]).content.strip()

    # Handle cancellation
    if resp.strip().upper() == "CANCEL" or "CANCEL" in resp.strip().upper():
        cancel_msg = "No problem! I've cancelled the asset request. Let me know if you need help with anything else."
        return {"messages": [AIMessage(content=cancel_msg)], "response": cancel_msg, "sources": []}

    # Extract SUBMIT line from response (handle extra text before/after)
    submit_line = None
    for line in resp.split('\n'):
        line = line.strip()
        if line.startswith("SUBMIT|"):
            submit_line = line
            break

    if submit_line:
        parts = submit_line.split("|", 2)
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

    # For leave_list, leave_balance, leave_status - actually fetch and return data
    if intent in ["leave_list", "leave_balance", "leave_status"]:
        from app.database import SessionLocal
        from app.tools.leave_tools import get_my_leave_requests, get_leave_balance
        from datetime import date
        db = SessionLocal()
        try:
            if intent == "leave_list":
                data = get_my_leave_requests(state["user_id"], db)
                if data["total"] == 0:
                    msg = "You don't have any leave requests yet. You can apply for leave by saying something like 'I want to apply for casual leave from tomorrow for 3 days'."
                else:
                    msg = f"**📋 Your Leave History**\n\n"
                    msg += f"**Total:** {data['total']} leave requests\n"
                    msg += f"- ✅ Approved: {data['approved']}\n"
                    msg += f"- ⏳ Pending: {data['pending']}\n"
                    msg += f"- ❌ Cancelled: {data['cancelled']}\n\n"
                    msg += "**Recent Leaves:**\n"
                    for leave in data['leaves'][:5]:  # Show last 5
                        status_emoji = {"approved": "✅", "pending": "⏳", "cancelled": "❌", "rejected": "🚫"}
                        emoji = status_emoji.get(leave['status'], "📋")
                        msg += f"\n{emoji} **{leave['request_id']}** - {leave['leave_type'].title()}\n"
                        msg += f"   📅 {leave['start_date']} → {leave['end_date']} ({leave['num_days']} days)\n"
                        msg += f"   Status: {leave['status'].title()}\n"
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}

            elif intent == "leave_balance":
                try:
                    balance = get_leave_balance(state["user_id"], date.today().year, db)
                    msg = f"**💼 Your Leave Balance ({balance['year']})**\n\n"
                    for leave_type, info in balance.items():
                        if leave_type != "year":
                            msg += f"**{leave_type.title()} Leave:**\n"
                            msg += f"  - Available: **{info['available']}** days\n"
                            msg += f"  - Used: {info['used']} / {info['total']} days\n\n"
                except Exception:
                    msg = "I couldn't find your leave balance record. Please contact HR to set up your leave balance."
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}

            elif intent == "leave_status":
                data = get_my_leave_requests(state["user_id"], db)
                pending = [l for l in data['leaves'] if l['status'] == 'pending']
                if pending:
                    msg = f"**⏳ Your Pending Leave Requests ({len(pending)})**\n\n"
                    for leave in pending:
                        msg += f"🔸 **{leave['request_id']}** - {leave['leave_type'].title()}\n"
                        msg += f"   📅 {leave['start_date']} → {leave['end_date']} ({leave['num_days']} days)\n"
                        msg += f"   📝 Reason: {leave['reason']}\n\n"
                else:
                    msg = "You don't have any pending leave requests. All your leaves are either approved, cancelled, or you haven't applied yet."
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        except Exception as e:
            msg = f"❌ Error fetching leave data: {str(e)}"
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        finally:
            db.close()

    # Role-based: HR/Admin can see company stats, others get their own data
    if intent in ["leave_statistics", "pending_approvals"]:
        from app.database import SessionLocal
        from app.tools.leave_tools import get_company_leave_stats, get_pending_approvals
        db = SessionLocal()
        try:
            role = state.get("role", "employee")
            user_id = state.get("user_id", 0)

            if intent == "leave_statistics":
                # HR/Admin can see company-wide stats
                if role in ["hr_team", "admin"]:
                    data = get_company_leave_stats(db)
                    stats = data["summary"]
                    msg = f"**📊 Company Leave Statistics**\n\n"
                    msg += f"**Total Leave Requests:** {stats['total']}\n\n"
                    msg += f"📈 **Status Breakdown:**\n"
                    msg += f"- ✅ Approved: {stats['approved']}\n"
                    msg += f"- ⏳ Pending: {stats['pending']}\n"
                    msg += f"- ❌ Cancelled: {stats['cancelled']}\n"
                    msg += f"- 🚫 Rejected: {stats['rejected']}\n\n"
                    if data["by_type"]:
                        msg += "**By Leave Type:**\n"
                        for lt, info in data["by_type"].items():
                            msg += f"- {lt.title()}: {info['total']} (✅ {info['approved']}, ⏳ {info['pending']})\n"
                else:
                    # Employees see their own stats only
                    personal_data = get_my_leave_requests(state["user_id"], db)
                    msg = f"**📊 Your Leave Statistics**\n\n"
                    msg += f"**Total Requests:** {personal_data['total']}\n"
                    msg += f"- ✅ Approved: {personal_data['approved']}\n"
                    msg += f"- ⏳ Pending: {personal_data['pending']}\n"
                    msg += f"- ❌ Cancelled: {personal_data['cancelled']}\n\n"
                    msg += "💡 Tip: HR team members can see company-wide stats. To view your personal leave history, try 'show my leaves'."
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}

            elif intent == "pending_approvals":
                # Manager/HR/Admin can see pending approvals
                if role in ["hr_team", "admin", "manager"]:
                    data = get_pending_approvals(user_id, role, db)
                    if data["total"] == 0:
                        msg = "✅ **No pending approvals!** There are no leave requests waiting for your approval."
                    else:
                        msg = f"**⏳ Pending Approvals ({data['total']})**\n\n"
                        msg += "**Requests awaiting your review:**\n\n"
                        for req in data["approvals"][:10]:  # Show first 10
                            msg += f"🔸 **{req['request_id']}** - {req['requester']} ({req['requester_dept']})\n"
                            msg += f"   📅 {req['leave_type'].title()} leave: {req['start_date']} → {req['end_date']} ({req['num_days']} days)\n"
                            msg += f"   📝 Reason: {req['reason']}\n\n"
                        msg += "💡 To approve/reject, say 'approve leave LVXXXX' or use the approval panel."
                else:
                    # Employees can't see others' pending approvals
                    msg = "🔒 **Access Denied**\n\nYou don't have permission to view pending approvals. This is restricted to managers and HR team members.\n\nIf you're looking for your own leave requests, try 'show my leaves' or 'check my leave status'."
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        except Exception as e:
            msg = f"❌ Error: {str(e)}"
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        finally:
            db.close()

    # Handle actual leave approval/rejection when request_id is provided
    if intent == "leave_approve":
        msg_text = state["messages"][-1].content.lower()
        import re
        request_id = None
        
        # Try 1: Extract request_id (LV followed by 8 alphanumeric chars)
        match = re.search(r'(lv[a-z0-9]{8})', msg_text)
        if match:
            request_id = match.group(1).upper()
        else:
            # Try 2: Use LLM to understand natural language ordinal references
            # Extract IDs from conversation history
            available_ids = []
            for prev_msg in reversed(state["messages"][-5:]):
                if hasattr(prev_msg, 'content'):
                    ids_found = re.findall(r'(LV[A-Z0-9]{8})', prev_msg.content)
                    if ids_found:
                        available_ids = ids_found
                        break
            
            if available_ids:
                # Ask LLM to parse natural language and return the index
                llm = get_llm_with_fallback(temperature=0.1, max_tokens=50)
                user_msg = state["messages"][-1].content
                prompt = f"""Available requests: {available_ids}
User said: "{user_msg}"
Which request is the user referring to? Reply with ONLY the 0-based index (0 for first, 1 for second, -1 for last, -2 for second to last, etc.) or "unknown" if unclear."""
                try:
                    llm_response = llm.invoke([SystemMessage(content=prompt)]).content.strip()
                    # Extract number from response
                    idx_match = re.search(r'-?\d+', llm_response)
                    if idx_match:
                        idx = int(idx_match.group())
                        if 0 <= idx < len(available_ids):
                            request_id = available_ids[idx]
                        elif idx < 0 and abs(idx) <= len(available_ids):
                            request_id = available_ids[idx]
                except Exception:
                    pass
        
        if request_id:
            # Determine action (approve or reject)
            action = "reject" if any(w in msg_text for w in ["reject", "decline", "deny"]) else "approve"
            
            from app.database import SessionLocal
            from app.tools.leave_tools import approve_leave
            from fastapi import HTTPException
            db = SessionLocal()
            try:
                result = approve_leave(
                    request_id=request_id,
                    approver_id=state["user_id"],
                    action=action,
                    notes="",
                    db=db
                )
                status_word = "approved" if action == "approve" else "rejected"
                msg = f"✅ **Leave request {request_id} has been {status_word}!**\n\nThe employee will be notified of your decision."
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
            except HTTPException as e:
                msg = f"❌ **Could not {action}:** {e.detail}"
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
            except Exception as e:
                msg = f"❌ Error: {str(e)}"
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
            finally:
                db.close()
        else:
            # User said approve/reject but didn't specify which request - show pending approvals
            from app.database import SessionLocal
            from app.tools.leave_tools import get_pending_approvals
            db = SessionLocal()
            try:
                role = state.get("role", "employee")
                user_id = state.get("user_id", 0)
                data = get_pending_approvals(user_id, role, db)
                if data["total"] == 0:
                    msg = "✅ **No pending approvals!** There are no leave requests waiting for your approval."
                else:
                    action_word = "reject" if any(w in msg_text for w in ["reject", "decline", "deny"]) else "approve"
                    msg = f"**📋 Pending Approvals ({data['total']})**\n\nWhich request would you like to {action_word}?\n\n"
                    for i, req in enumerate(data["approvals"][:5], 1):
                        msg += f"{i}. 🔸 **{req['request_id']}** - {req['requester']} ({req['requester_dept']})\n"
                        msg += f"   📅 {req['leave_type'].title()} leave: {req['start_date']} → {req['end_date']} ({req['num_days']} days)\n"
                        msg += f"   📝 Reason: {req['reason']}\n\n"
                    msg += f"💡 Say 'approve 1st one', 'approve second', or 'approve LVXXXX' to proceed."
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
            except Exception as e:
                msg = f"❌ Error fetching pending approvals: {str(e)}"
                return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
            finally:
                db.close()

    # For other leave intents (leave_cancel), use LLM with guidance
    guidance = {
        "leave_cancel":   "Help user cancel their leave request. Ask for request_id if not provided. Check their leave list if needed.",
    }
    hint = guidance.get(intent, "Help with leave questions - check balance, list leaves, or apply.")
    llm = get_llm_with_fallback(temperature=0.3, max_tokens=250)
    resp = llm.invoke([
        SystemMessage(content=f"HR Leave assistant. Today is {_get_current_time_ist()}. {hint} "
                              f"Context: {_history_str(state['messages'], 4)}. "
                              f"Be warm, brief (2-3 sentences). Never mention API endpoints or POST requests."),
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
    if intent == "it_ticket_list":
        from app.database import SessionLocal
        from app.tools.ticket_tools import get_all_open_tickets, get_my_tickets
        db = SessionLocal()
        try:
            role = state.get("role", "employee")
            # Role-based access: IT team sees ALL open tickets, employees see only their own
            if role in ["it_team", "admin"]:
                data = get_all_open_tickets(db)
                if data["total"] == 0:
                    msg = "✅ **No open tickets!** All IT tickets have been resolved."
                else:
                    msg = f"**🎫 All Open IT Tickets ({data['total']})**\n\n"
                    msg += f"- 🔴 Open: {data['open']}\n"
                    msg += f"- 🟡 In Progress: {data['in_progress']}\n\n"
                    msg += "**Tickets:**\n"
                    for ticket in data['tickets'][:10]:  # Show last 10
                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        p_emoji = priority_emoji.get(ticket['priority'], "⚪")
                        status_emoji = {"open": "📥", "in_progress": "🔧", "resolved": "✅"}
                        s_emoji = status_emoji.get(ticket['status'], "📋")
                        msg += f"\n{p_emoji} **{ticket['ticket_id']}** - {ticket['issue_type']}\n"
                        msg += f"   {s_emoji} Status: {ticket['status'].title()} | By: {ticket['requester']}\n"
                        msg += f"   📝 {ticket['description']}\n"
            else:
                # Regular employees see their own tickets only
                data = get_my_tickets(state["user_id"], db)
                if data["total"] == 0:
                    msg = "You haven't created any IT tickets yet. You can create one by saying something like 'My VPN is not working' or 'I need help with my laptop'."
                else:
                    msg = f"**🎫 Your IT Tickets ({data['total']})**\n\n"
                    msg += f"- 📥 Open: {data['open']}\n"
                    msg += f"- 🔧 In Progress: {data['in_progress']}\n"
                    msg += f"- ✅ Resolved: {data['resolved']}\n\n"
                    msg += "**Recent Tickets:**\n"
                    for ticket in data['tickets'][:5]:  # Show last 5
                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                        p_emoji = priority_emoji.get(ticket['priority'], "⚪")
                        status_emoji = {"open": "📥", "in_progress": "🔧", "resolved": "✅"}
                        s_emoji = status_emoji.get(ticket['status'], "📋")
                        msg += f"\n{p_emoji} **{ticket['ticket_id']}** - {ticket['issue_type']}\n"
                        msg += f"   {s_emoji} Status: {ticket['status'].title()}\n"
                        msg += f"   📝 {ticket['description']}\n"
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        except Exception as e:
            msg = f"❌ Error fetching tickets: {str(e)}"
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        finally:
            db.close()

    if intent == "it_inventory":
        from app.database import SessionLocal
        from app.tools.ticket_tools import get_inventory_status
        db = SessionLocal()
        try:
            role = state.get("role", "employee")
            # Role-based access: Only IT team and admin can view inventory
            if role in ["it_team", "admin"]:
                data = get_inventory_status(db)
                summary = data["summary"]
                pending = data["pending_requests"]

                msg = f"**📦 Asset Inventory Status**\n\n"
                msg += f"**Total Requests:** {summary['total']}\n\n"
                msg += f"📊 **Breakdown:**\n"
                msg += f"- ⏳ Pending Manager Approval: {summary['pending_manager']}\n"
                msg += f"- 🔧 Pending IT Approval: {summary['pending_it']}\n"
                msg += f"- ✅ Approved: {summary['approved']}\n"
                msg += f"- ❌ Rejected: {summary['rejected']}\n"
                msg += f"- 📦 Fulfilled: {summary['fulfilled']}\n\n"

                if pending:
                    msg += f"**🚨 Action Required ({len(pending)} pending):**\n"
                    for req in pending[:10]:  # Show first 10
                        status_emoji = "⏳" if req['status'] == 'pending_manager' else "🔧"
                        msg += f"\n{status_emoji} **{req['request_id']}** - {req['asset_type']}\n"
                        msg += f"   👤 {req['requester']} | Status: {req['status']}\n"
                else:
                    msg += "✅ **No pending asset requests requiring action.**"
            else:
                # Employees don't have access to inventory view
                msg = "🔒 **Access Denied**\n\nYou don't have permission to view the asset inventory. This is restricted to IT team members only.\n\nIf you need to request an asset, please say something like 'I want to request a laptop' or visit the asset request section."
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        except Exception as e:
            msg = f"❌ Error fetching inventory: {str(e)}"
            return {"messages": [AIMessage(content=msg)], "response": msg, "sources": []}
        finally:
            db.close()

    llm = get_llm_with_fallback(temperature=0.3, max_tokens=250)
    resp = llm.invoke([
        SystemMessage(content=f"IT Support assistant. Today is {_get_current_time_ist()}. "
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
        SystemMessage(content=f"Enterprise HR & IT Copilot. Today is {_get_current_time_ist()}. User role: {role}. "
                              f"Help with policies, leave, IT support, assets. "
                              f"Context: {_history_str(state['messages'], 4)}. "
                              f"Be warm, max 3 sentences."),
        HumanMessage(content=state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=resp.content)], "response": resp.content, "sources": []}


# ── Routing ───────────────────────────────────────────────────────────────────
def _route(state: AgentState) -> str:
    intent = state.get("intent", "general")
    if intent == "hr_policy":                                                  return "hr_rag"
    if intent in {"leave_balance","leave_apply","leave_status",
                  "leave_cancel","leave_approve","leave_list",
                  "leave_statistics","pending_approvals"}:                   return "hr_leave"
    if intent in {"it_ticket","it_asset","it_ticket_update","it_ticket_list","it_inventory"}: return "it_support"
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