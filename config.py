"""Centralised configuration for the Real Estate RAG Assistant.

Reads environment variables from .env via python-dotenv and exposes them
as module-level constants with sensible defaults. All pipeline components
import this module rather than reading env vars directly, making
configuration changes a single edit in .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.json"
GROUND_TRUTH_PATH = PROCESSED_DIR / "ground_truth.json"
MONITORING_DB_PATH = PROJECT_ROOT / "data" / "monitoring.db"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

LLM_JUDGE_PROVIDER = os.getenv("LLM_JUDGE_PROVIDER", "groq")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "openai/gpt-oss-120b")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "hybrid")

PROVIDER_CONFIGS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": CEREBRAS_API_KEY,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY,
    },
}
