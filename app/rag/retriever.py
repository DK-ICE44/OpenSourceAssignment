import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "app/rag/chroma_db"
COLLECTION_NAME = "employee_handbook"

_client = None
_collection = None

def _get_collection():
    global _client, _collection
    if _collection is None:
        emb_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
    return _collection

def retrieve(query: str, top_k: int = 5, department_filter: str = None) -> list[dict]:
    """Retrieve top-K relevant chunks for a query."""
    collection = _get_collection()

    where = {"department": department_filter} if department_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "handbook"),
            "page": meta.get("page", 0),
            "relevance_score": round(1 - dist, 4)
        })
    return chunks