"""
Central configuration for the RAG PDF Q&A Assistant.
Loads environment variables and defines tunable constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Model settings ---
GEMINI_MODEL = "gemini-3.6-flash"  # fast + on the free tier
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, 384-dim, good baseline

# --- Chunking settings ---
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# --- Retrieval settings ---
TOP_K = 4                 # number of chunks retrieved per query

# --- Storage paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "pdf_documents"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
