import argparse
import os
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "jasongyn/viet-legal-rag"

SECRET_KEYS = (
    "GROQ_API_KEY",
    "GOOGLE_GENERATOR_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_ORCHESTRATOR_API_KEY",
    "COHERE_API_KEY",
)

VARIABLE_KEYS = (
    "ORCHESTRATOR_PROVIDER",
    "GROQ_MODEL",
    "GOOGLE_GENERATOR_MODEL",
    "GOOGLE_MODEL",
    "GOOGLE_ORCHESTRATOR_MODEL",
    "COHERE_RERANK_MODEL",
    "TOP_K_RETRIEVE",
    "TOP_K_RERANK",
    "MAX_SEARCH_QUERIES",
    "EXACT_LOOKUP_SHORT_CIRCUIT",
    "MAX_CONTEXT_ARTICLES",
    "MAX_REFERENCE_ARTICLES",
    "MAX_ARTICLE_CHARS",
    "MAX_CONTEXT_CHARS",
    "EMBED_BATCH_SIZE",
    "MAX_CPU_CORES",
)

ALLOW_PATTERNS = (
    "README.md",
    "web_app.py",
    "requirements.txt",
    "src/**",
    "data/markdown/**",
    "bm25_index.json",
    "chroma_db/**",
)

IGNORE_PATTERNS = (
    ".env",
    ".env.*",
    ".git/**",
    ".cache/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "*.log",
    "data/raw/**",
)


def _is_real_value(value: str | None) -> bool:
    return bool(value and value.strip() and not value.lower().startswith("your_"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy VietLegalRAG to a Hugging Face Space.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-secrets", action="store_true")
    args = parser.parse_args()

    env = dotenv_values(ROOT / ".env")
    api = HfApi(token=os.getenv("HF_TOKEN") or None)

    if not args.skip_secrets:
        for key in SECRET_KEYS:
            value = env.get(key) or os.getenv(key)
            if _is_real_value(value):
                api.add_space_secret(args.repo_id, key, value)
                print(f"Set Space secret: {key}")

        for key in VARIABLE_KEYS:
            value = env.get(key) or os.getenv(key)
            if _is_real_value(value):
                api.add_space_variable(args.repo_id, key, value)
                print(f"Set Space variable: {key}")

    if not args.skip_upload:
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=ROOT,
            allow_patterns=list(ALLOW_PATTERNS),
            ignore_patterns=list(IGNORE_PATTERNS),
            commit_message="Deploy VietLegalRAG Streamlit app",
        )
        print(f"Uploaded Space commit: {commit.oid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
