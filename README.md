---
title: VietLegal RAG
sdk: streamlit
app_file: web_app.py
pinned: false
---

# VietLegal Agentic RAG Platform

Vietnamese legal RAG demo for laws, decrees, appendices, and table-heavy legal documents.

## What It Does

- Converts Vietnamese legal markdown into structured legal chunks.
- Indexes chunks into ChromaDB with `AITeamVN/Vietnamese_Embedding`.
- Combines vector retrieval and BM25 with reciprocal-rank fusion.
- Prioritizes exact legal lookup such as `Điều 17 Luật Doanh nghiệp 2020`.
- Optionally reranks with Cohere when `COHERE_API_KEY` is set.
- Generates legal answers with Groq orchestrator and Gemini generator when API keys are set.
- Provides a Streamlit demo UI with retrieval trace and timings.

## Setup

```powershell
cd D:\Projects\Personal\VietLegalRAG\vietlegal-agentic-rag-platform
& 'D:\Projects\Personal\VietLegalRAG\.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill these later in `.env` for full generation:

```env
GROQ_API_KEY=...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
```

Without keys, retrieval-only mode still works.

## Build Index

For a clean rebuild after changing documents or chunking:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m src.index_builder --rebuild
```

For resume mode:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m src.index_builder
```

CPU defaults are tuned for Ryzen-class laptops:

```env
MAX_CPU_CORES=12
EMBED_BATCH_SIZE=8
```

If the machine gets hot or slow, use `EMBED_BATCH_SIZE=4`.

## Run CLI

Full pipeline after adding API keys:

```powershell
python -m src.agent "Điều 17 Luật Doanh nghiệp 2020 quy định về điều gì?"
```

Retrieval-only smoke test:

```powershell
python -m scripts.evaluate_retrieval --top-k 5
```

## Run Web Demo

```powershell
streamlit run web_app.py
```

Keep `Retrieval only` checked until API keys are configured. After adding keys, uncheck it to run answer generation.

## Data Notes

- `chroma_db/`, `bm25_index.json`, `.cache/`, and `.env` are local artifacts and are ignored by git.
- If retrieval seems stale after adding documents, run `python -m src.index_builder --rebuild`.
- The system is a technical RAG demo, not legal advice.

## Latency Tuning

Defaults are tuned for faster interactive demos:

```env
TOP_K_RETRIEVE=30
TOP_K_RERANK=10
MAX_SEARCH_QUERIES=4
EXACT_LOOKUP_SHORT_CIRCUIT=true
MAX_CONTEXT_ARTICLES=6
MAX_REFERENCE_ARTICLES=2
MAX_ARTICLE_CHARS=4000
MAX_CONTEXT_CHARS=18000
```

- `EXACT_LOOKUP_SHORT_CIRCUIT=true` skips hybrid retrieval and Cohere rerank for precise queries such as `Điều 17 Luật Doanh nghiệp 2020`.
- Reduce `MAX_CONTEXT_*` values to lower Gemini latency and token cost.
- Increase `TOP_K_*` or `MAX_CONTEXT_*` when broad consultation queries need more recall.
