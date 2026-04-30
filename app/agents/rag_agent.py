"""
RAG pipeline for HR policy Q&A.
- Retrieves relevant handbook chunks
- Uses Gemini as primary LLM
- Falls back to Groq if needed
"""

import logging
from typing import Dict, Any, List

from app.rag.retriever import retrieve
from app.config import get_settings

settings = get_settings()

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an HR Policy Assistant for a company. 
Answer employee questions using ONLY the provided context from the employee handbook.
If the answer is not in the context, say:
"I couldn't find this in the handbook. Please contact HR directly."
Always cite the page number when available.
Be concise, professional, and friendly.
"""

# ------------------------------------------------------------------------------
# LLM Calls
# ------------------------------------------------------------------------------

def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.2
        )
    )
    return response.text


def _call_groq(prompt: str) -> str:
    from groq import Groq

    if not settings.groq_api_key:
        raise RuntimeError("Groq API key not configured")

    client = Groq(api_key=settings.groq_api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    )

    if not response or not response.choices:
        raise ValueError("Empty response from Groq")

    return response.choices[0].message.content.strip()


# ------------------------------------------------------------------------------
# Core Logic
# ------------------------------------------------------------------------------

def _build_prompt(context: str, question: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context from Employee Handbook:
{context}

Employee Question: {question}

Answer strictly using the context above.
"""


def _try_gemini(prompt: str) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini not configured")

    logger.info("Trying Gemini...")
    return _call_gemini(prompt)


def _try_groq(prompt: str) -> str:
    logger.info("Trying Groq fallback...")
    return _call_groq(prompt)


def answer_policy_question(question: str, user_role: str = "employee") -> Dict[str, Any]:
    """
    RAG pipeline: retrieve → generate → fallback if needed
    """

    # --------------------------------------------------------------------------
    # Step 1: Retrieve
    # --------------------------------------------------------------------------
    chunks: List[Dict] = retrieve(question, top_k=5)

    if not chunks:
        return {
            "answer": "I couldn't find relevant policy information. Please contact HR.",
            "sources": [],
            "retrieved_chunks": 0
        }

    # --------------------------------------------------------------------------
    # Step 2: Build context
    # --------------------------------------------------------------------------
    context = "\n\n".join([
        f"[Page {c['page']}] {c['text']}" for c in chunks
    ])

    prompt = _build_prompt(context, question)

    # --------------------------------------------------------------------------
    # Step 3: Generate with fallback
    # --------------------------------------------------------------------------
    answer = None
    errors = []

    # Try Gemini first
    try:
        answer = _try_gemini(prompt)

        # Validate output
        if not answer or len(answer.strip()) == 0:
            raise ValueError("Gemini returned empty answer")

    except Exception as e:
        logger.warning(f"Gemini failed: {str(e)}")
        errors.append(f"Gemini: {str(e)}")

        # Fallback to Groq
        try:
            answer = _try_groq(prompt)

            if not answer or len(answer.strip()) == 0:
                raise ValueError("Groq returned empty answer")

        except Exception as e2:
            logger.error(f"Groq failed: {str(e2)}")
            errors.append(f"Groq: {str(e2)}")

            answer = "Both LLM providers failed. Please contact HR or try again later."

    # --------------------------------------------------------------------------
    # Step 4: Sources
    # --------------------------------------------------------------------------
    sources = [
        {
            "page": c["page"],
            "source": c["source"],
            "score": c["relevance_score"]
        }
        for c in chunks
    ]

    # --------------------------------------------------------------------------
    # Step 5: Response
    # --------------------------------------------------------------------------
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(chunks),
        "errors": errors  # useful for debugging, remove in prod if needed
    }

