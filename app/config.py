from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "HR-IT Copilot"
    debug: bool = False
    database_url: str = "sqlite:///./db/copilot.db"
    secret_key: str = "changeme"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    gemini_api_key: str = ""
    groq_api_key: str = ""
    power_automate_email_webhook: str = ""

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()