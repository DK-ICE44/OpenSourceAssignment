import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "HR-IT Copilot"
    debug: bool = False
    database_url: str = "sqlite:///./db/copilot.db"
    secret_key: str = "changeme"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    # LLM Keys
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # LangSmith Tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "hr-it-copilot"

    # Power Automate
    power_automate_email_webhook: str = ""

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

def configure_langsmith():
    """Set LangSmith environment variables if tracing is enabled."""
    settings = get_settings()
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        return True
    return False