"""
Intent classifier: routes a user message to the correct agent.
Uses a lightweight LLM call (Groq llama-3.1-8b-instant for speed).
"""
import json
from app.config import get_settings

settings = get_settings()

INTENT_PROMPT = """You are an intent classifier for an enterprise HR & IT assistant.
Classify the user query into exactly ONE of these intents:
- hr_policy: questions about company policies, rules, benefits, handbook
- leave_apply: applying for leave
- leave_balance: checking leave balance or history  
- leave_approve: approving or rejecting leave (manager only)
- leave_status: checking status of a leave request
- leave_cancel: cancelling a leave request
- it_ticket: raising or checking IT support tickets
- it_asset: requesting IT assets (laptop, monitor, etc.)
- it_ticket_update: updating ticket status (IT team only)
- general: greetings, thanks, unclear intent

Respond with ONLY a JSON object like: {"intent": "hr_policy", "confidence": 0.95}
"""

def classify_intent(user_message: str) -> dict:
    """Returns intent classification."""
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast, free, great for classification
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=60,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception:
        # Fallback: keyword-based classification
        msg = user_message.lower()
        if any(w in msg for w in ["policy", "rule", "handbook", "leave policy",
                                    "notice period", "wfh", "work from home"]):
            return {"intent": "hr_policy", "confidence": 0.7}
        elif any(w in msg for w in ["apply leave", "take leave", "want leave"]):
            return {"intent": "leave_apply", "confidence": 0.7}
        elif any(w in msg for w in ["balance", "how many leaves", "remaining"]):
            return {"intent": "leave_balance", "confidence": 0.7}
        elif any(w in msg for w in ["ticket", "vpn", "laptop issue", "printer",
                                     "network", "software"]):
            return {"intent": "it_ticket", "confidence": 0.7}
        elif any(w in msg for w in ["asset", "request laptop", "request monitor"]):
            return {"intent": "it_asset", "confidence": 0.7}
        return {"intent": "general", "confidence": 0.5}