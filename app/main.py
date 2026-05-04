from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings, configure_langsmith
from app.database import init_db
from app.routers import auth, hr, it
from app.routers.chat import router as chat_router
from app.middleware.logging_mw import log_request

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise HR & IT AI Copilot — LangGraph + LangChain + RAG + RBAC",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(log_request)

# Routers
app.include_router(auth.router)
app.include_router(hr.router)
app.include_router(it.router)
app.include_router(chat_router)   # ← NEW: LangGraph-powered chat

@app.on_event("startup")
def startup():
    init_db()
    
    # Validate configuration
    if not settings.gemini_api_key:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "GEMINI_API_KEY not configured. Chat features will be limited. "
            "Add GEMINI_API_KEY to .env file. Get it from: https://ai.google.dev"
        )
    if not settings.groq_api_key:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "GROQ_API_KEY not configured. Intent classification will use keyword fallback. "
            "Add GROQ_API_KEY to .env file. Get it from: https://console.groq.com"
        )
    
    # Configure LangSmith tracing if key is present
    tracing = configure_langsmith()
    print(f"{settings.app_name} v2.0 started!")
    print(f"   LangSmith tracing: {'enabled' if tracing else 'disabled'}")

@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name, "version": "2.0.0"}

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": "2.0.0",
        "docs": "/docs",
        "chat_endpoint": "/chat",
        "health": "/health"
    }