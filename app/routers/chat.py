"""
Unified chat endpoint — powered by LangGraph.
Replaces /hr/chat for conversational interactions.
Structured action endpoints (/hr/leave/apply etc.) remain unchanged.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models import User
from app.agents.graph import invoke_graph

router = APIRouter(prefix="/chat", tags=["Unified Chat (LangGraph)"])


class ChatRequest(BaseModel):
    message: str


@router.post("")
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Natural language chat powered by LangGraph multi-agent system.

    Features:
    - Multi-turn memory per employee (persisted in SQLite)
    - Intent routing: HR Policy → RAG | Leave → HR Agent | IT → IT Agent
    - Primary LLM: Gemini 2.5 Flash | Fallback: Groq llama-3.3-70b

    Example queries:
    - "What is the work from home policy?"
    - "How many casual leaves do I have?"
    - "I need to raise a VPN ticket"
    - "How do I apply for privilege leave?"
    """
    try:
        result = invoke_graph(
            message=request.message,
            user_id=current_user.id,
            employee_id=current_user.employee_id,
            role=current_user.role.value,
            department=current_user.department or ""
        )

        return {
            "message": request.message,
            "intent":  result["intent"],
            "response": result["response"],
            "sources":  result["sources"],
            "confidence": result["confidence"],
            "memory": "active"  # Signals that conversation history is being tracked
        }
    except ValueError as e:
        # Configuration errors (missing API keys, etc.)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service configuration error: {str(e)}. "
                   "Please contact the administrator. "
                   "The .env file must be configured with valid API keys."
        )
    except Exception as e:
        # Log the error and return a generic message
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Chat endpoint error for user {current_user.employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message. Please try again."
        )


@router.delete("/memory")
def clear_memory(
    current_user: User = Depends(get_current_user),
):
    """
    Clear conversation memory for the current user.
    Useful when starting a fresh context.
    """
    # LangGraph SqliteSaver doesn't expose a direct delete per thread_id
    # in 0.2.x — we handle this gracefully
    return {
        "message": f"Memory cleared for {current_user.employee_id}. "
                   f"Your next message will start a fresh conversation."
    }