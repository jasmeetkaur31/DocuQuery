"""
Embedding generation (sentence-transformers) and vector storage (ChromaDB).
"""
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    TOP_K,
)

# Loaded once at import time — avoids reloading the model on every request.
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeds a list of strings into vectors."""
    return _embedding_model.encode(texts, show_progress_bar=False).tolist()


def add_chunks_to_store(chunks: List[Dict]) -> int:
    """
    Embeds and stores a list of chunk dicts (from ingestion.process_pdf)
    into ChromaDB. Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    embeddings = embed_texts(texts)

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def query_store(question: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Embeds the question and retrieves the top_k most similar chunks.
    Returns a list of dicts: [{"text":..., "source":..., "page":..., "distance":...}, ...]
    """
    query_embedding = embed_texts([question])[0]
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        retrieved.append({
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "distance": dist,
        })
    return retrieved


def get_indexed_document_count() -> int:
    """Returns the total number of chunks currently stored."""
    return _collection.count()


def reset_store():
    """Deletes all vectors — useful for testing/reset endpoint."""
    global _collection
    _chroma_client.delete_collection(COLLECTION_NAME)
    _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
