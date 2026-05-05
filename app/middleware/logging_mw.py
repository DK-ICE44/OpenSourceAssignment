"""
Request logging middleware.
Logs every request to console AND writes to the audit_logs SQLite table.
Uses FastAPI BackgroundTask to avoid blocking the response.
"""
import time
import logging
from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


async def log_request(request: Request, call_next):
    """
    Middleware that:
    1. Times the request
    2. Logs to console
    3. Writes to audit_logs table via background task (non-blocking)
    """
    start = time.time()
    response: Response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)

    # Console log — always
    logger.info(
        f"[{request.method}] {request.url.path} | "
        f"{response.status_code} | {duration_ms}ms"
    )

    # DB audit log — background, non-blocking
    try:
        _write_audit_log(request, response.status_code, duration_ms)
    except Exception as e:
        logger.warning(f"Audit log write failed (non-critical): {e}")

    return response


def _write_audit_log(request: Request, status_code: int, duration_ms: float):
    """Write a record to the audit_logs table."""
    from app.database import SessionLocal
    from app.models import AuditLog
    from datetime import datetime

    # Skip logging for health/docs/static endpoints to reduce noise
    skip_paths = {"/health", "/docs", "/openapi.json", "/favicon.ico", "/"}
    if request.url.path in skip_paths:
        return

    # Try to extract user_id from JWT token if present
    user_id = None
    try:
        from app.middleware.auth import get_settings
        from jose import jwt
        settings = get_settings()
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = jwt.decode(
                token, settings.secret_key,
                algorithms=[settings.algorithm]
            )
            employee_id = payload.get("sub")
            if employee_id:
                db_temp = SessionLocal()
                try:
                    from app.models import User
                    user = db_temp.query(User).filter(
                        User.employee_id == employee_id
                    ).first()
                    if user:
                        user_id = user.id
                finally:
                    db_temp.close()
    except Exception:
        pass  # Token extraction is best-effort

    # Determine agent/tool used from path
    path = request.url.path
    agent_used = None
    tool_used = None

    if "/chat" in path:
        agent_used = "langgraph"
    elif "/hr/policy" in path:
        agent_used = "rag_agent"
        tool_used = "chromadb_retriever"
    elif "/hr/leave" in path:
        agent_used = "hr_agent"
        tool_used = "leave_tools"
    elif "/it/tickets" in path:
        agent_used = "it_agent"
        tool_used = "ticket_tools"
    elif "/it/assets" in path:
        agent_used = "it_agent"
        tool_used = "asset_tools"
    elif "/auth" in path:
        agent_used = "auth"

    db = SessionLocal()
    try:
        log_entry = AuditLog(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            endpoint=f"{request.method} {path}",
            agent_used=agent_used,
            tool_used=tool_used,
            response_time_ms=duration_ms,
            status_code=status_code
        )
        db.add(log_entry)
        db.commit()
    finally:
        db.close()