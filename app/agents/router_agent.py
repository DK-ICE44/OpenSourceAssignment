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

# ── Prompt ────────────────────────────────────────────────────────────────────
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for an enterprise HR & IT assistant.
Classify the user query into exactly ONE of these intents:
- hr_policy       : company policies, rules, benefits, handbook questions
- leave_apply     : applying for leave
- leave_balance   : checking leave balance or history
- leave_approve   : approving or rejecting leave (manager action)
- leave_status    : checking status of a leave request
- leave_cancel    : cancelling a leave request
- it_ticket       : raising or tracking IT support tickets
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
    if any(w in msg for w in ["balance", "how many leaves", "remaining leave", "leave left"]):
        return {"intent": "leave_balance", "confidence": 0.7}
    if any(w in msg for w in ["leave status", "my leave request", "pending leave"]):
        return {"intent": "leave_status", "confidence": 0.7}
    if any(w in msg for w in ["approve leave", "reject leave", "pending approval"]):
        return {"intent": "leave_approve", "confidence": 0.7}
    if any(w in msg for w in ["ticket", "vpn", "laptop issue", "printer",
                               "network", "software install", "outlook"]):
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
        raw = chain.invoke({"message": user_message}).strip()
        result = json.loads(raw)
        logger.info(f"Intent classified: {result}")
        return result
    except Exception as e:
        logger.warning(f"LLM intent classification failed ({e}), using keyword fallback")
        return _keyword_fallback(user_message)