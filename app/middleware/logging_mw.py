import time
from fastapi import Request
from sqlalchemy.orm import Session
from app.models import AuditLog

async def log_request(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    # Log to console (DB logging can be added via background task)
    print(f"[{request.method}] {request.url.path} | "
          f"{response.status_code} | {duration_ms:.1f}ms")

    return response