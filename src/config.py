import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Data Paths ──
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = os.getenv("INPUT_DIR", str(BASE_DIR / "data" / "markdown"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(BASE_DIR / "bm25_index.json"))
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / ".cache" / "models"))

# ── Embedding & Retrieval ──
EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "50"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "20"))
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "4"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
EXACT_LOOKUP_SHORT_CIRCUIT = os.getenv("EXACT_LOOKUP_SHORT_CIRCUIT", "true").lower() not in {
    "0",
    "false",
    "no",
}

# Reranker: Cohere API (no local model)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
HYBRID_VECTOR_WEIGHT = 0.5
HYBRID_BM25_WEIGHT = 0.5

# Generation context budget
MAX_CONTEXT_ARTICLES = int(os.getenv("MAX_CONTEXT_ARTICLES", "6"))
MAX_REFERENCE_ARTICLES = int(os.getenv("MAX_REFERENCE_ARTICLES", "2"))
MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "4000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "18000"))

# ── System ──
MAX_CPU_CORES = int(os.getenv("MAX_CPU_CORES", "12"))

# ── Multi-Agent LLM Config ──
# Orchestrator: Groq by default; optionally Google with a separate key.
ORCHESTRATOR_PROVIDER = os.getenv("ORCHESTRATOR_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GOOGLE_ORCHESTRATOR_API_KEY = os.getenv("GOOGLE_ORCHESTRATOR_API_KEY", "")
GOOGLE_ORCHESTRATOR_MODEL = os.getenv("GOOGLE_ORCHESTRATOR_MODEL", "gemini-3.5-flash")

# Generator: Google AI Studio / Gemini (large context, accurate synthesis)
GENERATOR_PROVIDER = "google"
GOOGLE_GENERATOR_API_KEY = os.getenv("GOOGLE_GENERATOR_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GOOGLE_GENERATOR_MODEL = os.getenv("GOOGLE_GENERATOR_MODEL", os.getenv("GOOGLE_MODEL", "gemini-3.5-flash"))

# Backward-compatible aliases used by the current app code.
GOOGLE_API_KEY = GOOGLE_GENERATOR_API_KEY
GOOGLE_MODEL = GOOGLE_GENERATOR_MODEL

# ── Legacy single-model config (for index_builder, etc.) ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
