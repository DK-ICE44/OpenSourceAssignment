"""
Intent classifier rebuilt with LangChain.
Uses Groq llama-3.1-8b-instant (fastest free model) for low-latency classification.
Falls back to keyword matching if LLM call fails.
"""
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.agents.llm_factory import get_fast_llm

logger = logging.getLogger(__name__)

# ── Date/Time Helper ────────────────────────────────────────────────────────────
def _get_current_time_ist():
    """Get current date and time in IST format (UTC+5:30)."""
    from datetime import datetime, timedelta, timezone
    ist_offset = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist_offset)
    return now.strftime("%A, %d %B %Y, %I:%M %p IST")

# ── Prompt ────────────────────────────────────────────────────────────────────
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for an enterprise HR & IT assistant.
Current date and time: {current_time}

Classify the user query into exactly ONE of these intents:
- hr_policy       : company policies, rules, benefits, handbook questions
- leave_apply     : applying for leave
- leave_balance   : checking leave balance remaining
- leave_list      : listing/viewing all my leave requests or history
- leave_approve   : approving or rejecting leave (manager action)
- leave_status    : checking status of a specific leave request
- leave_cancel    : cancelling a leave request
- leave_statistics : viewing company-wide leave stats (HR/admin only)
- pending_approvals : viewing pending approvals to action (manager/HR/admin)
- it_ticket       : raising or tracking IT support tickets
- it_ticket_list  : listing/showing IT tickets (my tickets or all open for IT team)
- it_inventory    : viewing asset inventory status (IT team/admin only)
- it_asset        : requesting IT assets (laptop, monitor, etc.)
- it_ticket_update: updating ticket status (IT team action)
- general         : greetings, thanks, or unclear intent

Rules:
- Respond with ONLY a raw JSON object. No markdown. No extra text.
- Example: {{"intent": "hr_policy", "confidence": 0.95}}"""),
    ("human", "{message}")
])

# ── Chain singleton ───────────────────────────────────────────────────────────
_classifier_chain = None

def _get_classifier_chain():
    global _classifier_chain
    if _classifier_chain is None:
        llm = get_fast_llm(temperature=0.1, max_tokens=60)
        _classifier_chain = INTENT_PROMPT | llm | StrOutputParser()
        logger.info("Intent classifier chain initialized (Groq llama-3.1-8b-instant)")
    return _classifier_chain

# ── Keyword fallback ──────────────────────────────────────────────────────────
def _keyword_fallback(message: str) -> dict:
    msg = message.lower()
    if any(w in msg for w in ["policy", "rule", "handbook", "notice period",
                               "wfh", "work from home", "maternity", "paternity"]):
        return {"intent": "hr_policy", "confidence": 0.7}
    if any(w in msg for w in ["apply leave", "take leave", "want leave", "need leave"]):
        return {"intent": "leave_apply", "confidence": 0.7}
    if any(w in msg for w in ["balance", "remaining leave", "leave left"]):
        return {"intent": "leave_balance", "confidence": 0.7}
    if any(w in msg for w in ["how many leave", "how many leaves", "leaves have i applied", "leaves i applied", "my leave requests", "list my leaves", "show my leaves", "what are my leaves", "view my leaves"]) or "leave history" in msg or ("my pending" in msg and "approv" not in msg) or ("show my pending" in msg and "approv" not in msg):
        return {"intent": "leave_list", "confidence": 0.8}
    if any(w in msg for w in ["leave status", "my leave request", "pending leave", "my pending leaves", "leaves waiting for approval", "check my leave status"]):
        return {"intent": "leave_status", "confidence": 0.7}
    if any(w in msg for w in ["approve leave", "reject leave", "pending approval"]):
        return {"intent": "leave_approve", "confidence": 0.7}
    if any(w in msg for w in ["leave statistics", "company leave stats", "team leave stats", "leave overview"]):
        return {"intent": "leave_statistics", "confidence": 0.8}
    if any(w in msg for w in ["pending approvals", "show approvals", "approvals to review", "what needs my approval"]):
        return {"intent": "pending_approvals", "confidence": 0.8}
    if any(w in msg for w in ["show all tickets", "show my tickets", "list tickets", "all open tickets", "view tickets", "open tickets"]):
        return {"intent": "it_ticket_list", "confidence": 0.8}
    if any(w in msg for w in ["inventory", "asset inventory", "stock status", "inventory status"]):
        return {"intent": "it_inventory", "confidence": 0.85}
    if any(w in msg for w in ["create ticket", "raise ticket", "new ticket", "ticket for", "vpn issue", "laptop issue", "printer issue",
                               "network issue", "software install", "outlook issue"]):
        return {"intent": "it_ticket", "confidence": 0.7}
    if any(w in msg for w in ["asset", "request laptop", "request monitor",
                               "need equipment", "new laptop"]):
        return {"intent": "it_asset", "confidence": 0.7}
    return {"intent": "general", "confidence": 0.5}

# ── Public API ────────────────────────────────────────────────────────────────
def classify_intent(user_message: str) -> dict:
    """
    Classify intent using LangChain LCEL chain (Groq).
    Falls back to keyword matching on any failure.
    """
    try:
        chain = _get_classifier_chain()
        raw = chain.invoke({"message": user_message, "current_time": _get_current_time_ist()}).strip()
        result = json.loads(raw)
        logger.info(f"Intent classified: {result}")
        return result
    except ValueError as e:
        # API key validation error
        logger.error(f"Configuration error in intent classification: {e}")
        raise
    except Exception as e:
        logger.warning(f"LLM intent classification failed ({e}), using keyword fallback")
        return _keyword_fallback(user_message)