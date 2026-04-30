from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import init_db
from app.routers import auth, hr, it
from app.middleware.logging_mw import log_request

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise HR & IT AI Copilot with RAG, RBAC, and LLM routing",
    version="1.0.0"
)

# CORS (allow all for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
app.middleware("http")(log_request)

# Routers
app.include_router(auth.router)
app.include_router(hr.router)
app.include_router(it.router)

@app.on_event("startup")
def startup():
    init_db()
    print(f"✅ {settings.app_name} started!")

@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/health"
    }