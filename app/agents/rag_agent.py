"""
RAG pipeline rebuilt with LangChain LCEL.
Flow: question → ChromaDB retrieval → context formatting → LLM chain → answer
"""
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.rag.retriever import retrieve
from app.agents.llm_factory import get_llm_with_fallback

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────
HR_POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an HR Policy Assistant for a company.
PRIORITY 1: Answer using the provided employee handbook context when available. Always cite page numbers (e.g., "According to page 5...").
PRIORITY 2: If the handbook doesn't cover the topic, provide a helpful general HR response based on common practices, but clearly state: "This specific information isn't in our handbook, but generally..."
PRIORITY 3: For company-specific questions not in the handbook (e.g., specific deadlines, forms, contacts), say: "I don't see this in the handbook. Please contact HR directly for accurate information."
Be concise, professional, and friendly."""),
    ("human", "Context from Employee Handbook:\n{context}\n\nEmployee Question: {question}")
])

# ── Chain singleton ───────────────────────────────────────────────────────────
_rag_chain = None

def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        llm = get_llm_with_fallback(temperature=0.2, max_tokens=1024)
        _rag_chain = HR_POLICY_PROMPT | llm | StrOutputParser()
        logger.info("RAG chain initialized (Gemini → Groq fallback)")
    return _rag_chain

# ── Public API ────────────────────────────────────────────────────────────────
def answer_policy_question(question: str, user_role: str = "employee") -> Dict[str, Any]:
    """
    Full LangChain RAG pipeline:
    1. Retrieve top-5 chunks from ChromaDB (existing retriever, unchanged)
    2. Format chunks into context string
    3. Invoke LCEL chain: prompt | gemini-with-groq-fallback | str_parser
    """
    # Step 1: Retrieve
    chunks = retrieve(question, top_k=5)

    if not chunks:
        return {
            "answer": "I couldn't find relevant policy information. Please contact HR.",
            "sources": [],
            "retrieved_chunks": 0
        }

    # Step 2: Format context
    context = "\n\n".join([
        f"[Page {c['page']}] {c['text']}" for c in chunks
    ])

    # Step 3: Generate via LangChain LCEL chain
    try:
        chain = _get_rag_chain()
        answer = chain.invoke({"context": context, "question": question})
    except Exception as e:
        logger.error(f"RAG chain failed: {e}")
        answer = "I'm unable to retrieve policy information right now. Please contact HR directly."

    # Step 4: Build sources
    sources = [
        {"page": c["page"], "source": c["source"], "score": c["relevance_score"]}
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(chunks)
    }