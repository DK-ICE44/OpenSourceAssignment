"""
Centralized LangChain LLM factory.
Primary  : Gemini 2.5 Flash  (langchain-google-genai)
Fast     : Groq llama-3.1-8b-instant  (classification, low latency)
Fallback : Groq llama-3.3-70b-versatile (when Gemini fails)
"""
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_primary_llm(temperature: float = 0.2, max_tokens: int = 1024):
    """Gemini 2.5 Flash — best quality, used for RAG and complex reasoning."""
    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY not configured. Please add it to .env file. "
            "Get your key from: https://ai.google.dev"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def get_fast_llm(temperature: float = 0.1, max_tokens: int = 128):
    """Groq llama-3.1-8b-instant — fastest, used only for intent classification."""
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. Please add it to .env file. "
            "Get your key from: https://console.groq.com"
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_fallback_llm(temperature: float = 0.2, max_tokens: int = 1024):
    """Groq llama-3.3-70b — fallback when Gemini is unavailable or fails."""
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. Please add it to .env file. "
            "Get your key from: https://console.groq.com"
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_with_fallback(temperature: float = 0.2, max_tokens: int = 1024):
    """
    Returns Gemini with automatic Groq fallback via LangChain's .with_fallbacks().
    This is the recommended LLM for most agent nodes.
    """
    primary = get_primary_llm(temperature, max_tokens)
    fallback = get_fallback_llm(temperature, max_tokens)
    return primary.with_fallbacks([fallback])