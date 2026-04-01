import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Data Paths ──
INPUT_DIR = "data/markdown"
CHROMA_DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index.json"

# ── Embedding & Retrieval ──
EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
TOP_K_RETRIEVE = 50
TOP_K_RERANK = 5

# Reranker: Cohere API (no local model)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
HYBRID_VECTOR_WEIGHT = 0.5
HYBRID_BM25_WEIGHT = 0.5

# ── System ──
MAX_CPU_CORES = 12

# ── Multi-Agent LLM Config ──
# Orchestrator: Groq (ultra-fast query analysis & keyword extraction)
ORCHESTRATOR_PROVIDER = "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Generator: Google AI Studio / Gemini (large context, accurate synthesis)
GENERATOR_PROVIDER = "google"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

# ── Legacy single-model config (for index_builder, etc.) ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
