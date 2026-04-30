"""
Run once (or whenever the handbook changes) to build the vector store.
Usage: python -m app.rag.ingest
"""
import os
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.config import get_settings

CHROMA_PATH = "app/rag/chroma_db"
COLLECTION_NAME = "employee_handbook"
CHUNK_SIZE = 600      # characters per chunk
CHUNK_OVERLAP = 100

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF, preserving page metadata."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also grab tables
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    if row:
                        table_text += " | ".join(
                            [str(c) if c else "" for c in row]
                        ) + "\n"
            full_text = text + ("\n[TABLE]\n" + table_text if table_text else "")
            if full_text.strip():
                pages.append({
                    "text": full_text.strip(),
                    "page_num": i + 1,
                    "source": os.path.basename(pdf_path)
                })
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest():
    os.makedirs(CHROMA_PATH, exist_ok=True)

    pdf_path = "data/employee_handbook.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at {pdf_path}. Place your handbook there first.")
        return

    print(f"📄 Extracting text from {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"   Found {len(pages)} pages with content.")

    # Build chunks
    all_chunks, all_ids, all_metas = [], [], []
    chunk_idx = 0
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            if chunk.strip():
                all_chunks.append(chunk)
                all_ids.append(f"chunk_{chunk_idx}")
                all_metas.append({
                    "page": page["page_num"],
                    "source": page["source"],
                    "department": "HR"  # All handbook docs are HR-accessible
                })
                chunk_idx += 1

    print(f"   Created {len(all_chunks)} chunks.")

    # Embed and store
    print("🔢 Embedding and storing in ChromaDB...")
    emb_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Free, local, fast (~80MB)
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to re-ingest fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Batch insert (ChromaDB has batch limits)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
            metadatas=all_metas[i:i+batch_size]
        )
        print(f"   Stored batch {i//batch_size + 1}/{-(-len(all_chunks)//batch_size)}")

    print(f"✅ Ingestion complete! {len(all_chunks)} chunks stored.")

if __name__ == "__main__":
    ingest()