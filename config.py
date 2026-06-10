"""
Configuration module for the RAG Evaluation Framework.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for all framework components."""

    # ── OpenAI ──────────────────────────────────────────────
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ── RAG Pipeline ────────────────────────────────────────
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

    # ── Evaluation ──────────────────────────────────────────
    TESTSET_SIZE = int(os.getenv("TESTSET_SIZE", "50"))

    # ── Paths ───────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

    # ── Flask ───────────────────────────────────────────────
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5050"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # ── Wikipedia Topics ────────────────────────────────────
    DEFAULT_TOPICS = [
        "Artificial intelligence",
        "Machine learning",
        "Natural language processing",
        "Deep learning",
        "Neural network",
        "Transformer (deep learning architecture)",
        "Large language model",
        "Retrieval-augmented generation",
    ]

    @classmethod
    def ensure_dirs(cls):
        """Create required directories if they don't exist."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.CHROMA_DIR, exist_ok=True)
