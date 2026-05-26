---
title: VietLegal RAG
sdk: streamlit
app_file: web_app.py
pinned: false
---

# VietLegal Agentic RAG Platform

VietLegalRAG is a Vietnamese legal RAG demo for enterprise, investment, decree, appendix, and table-heavy legal documents.

Live demo: https://huggingface.co/spaces/jasongyn/viet-legal-rag

The app is a technical demo, not legal advice.

## Highlights

- Agentic query planning: an orchestrator turns user questions into focused legal search queries.
- Hybrid retrieval: vector search with `AITeamVN/Vietnamese_Embedding` plus BM25 and reciprocal-rank fusion.
- Exact lookup shortcut: direct retrieval for queries such as `Dieu 17 Luat Doanh nghiep 2020`.
- Optional reranking: Cohere reranks fused retrieval candidates when `COHERE_API_KEY` is available.
- Answer generation: Gemini synthesizes cited legal answers from retrieved context.
- Streamlit UI: sample questions, streaming answers, retrieval trace, and timing metrics.
- Hugging Face Space deployment script with API keys stored as Space secrets.

## Current Corpus

The repository includes processed Markdown legal documents for:

- Luat Doanh nghiep 2020 and amendment materials.
- Luat Dau tu 2025.
- Nghi dinh 31/2021/ND-CP.
- Nghi dinh 47/2021/ND-CP.
- Nghi dinh 153/2020/ND-CP.
- Nghi dinh 168/2025/ND-CP.
- Appendices and table-derived legal entries used by the retriever.

Raw ODT documents are kept under `data/raw/`. Processed Markdown documents are under `data/markdown/`.

## Architecture

```text
User question
  -> Orchestrator Agent
     -> query classification
     -> search query decomposition
  -> LegalRetriever
     -> exact article/appendix lookup
     -> vector retrieval + BM25
     -> reciprocal-rank fusion
     -> optional Cohere rerank
     -> cross-reference expansion
  -> Generator Agent
     -> Gemini answer synthesis
     -> streamed response in Streamlit
```

## Setup

```powershell
cd D:\Projects\Personal\VietLegalRAG\vietlegal-agentic-rag-platform
& 'D:\Projects\Personal\VietLegalRAG\.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your own keys:

```env
GROQ_API_KEY=...
GOOGLE_GENERATOR_API_KEY=...
COHERE_API_KEY=...
```

Backward-compatible `GOOGLE_API_KEY` also works for the generator.

## Run The App

```powershell
$env:PYTHONIOENCODING='utf-8'
streamlit run web_app.py
```

The web app always runs the full pipeline: orchestrator, retriever, and generator.

## CLI Query

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m src.agent "Dieu 17 Luat Doanh nghiep 2020 quy dinh ve dieu gi?"
```

## Convert Raw ODT Documents

The ODT conversion script uses Pandoc and BeautifulSoup to:

- convert ODT to HTML/Markdown;
- extract basic document metadata;
- linearize legal tables into searchable prose;
- write Markdown files with frontmatter.

Install Pandoc first, then run:

```powershell
python -m scripts.convert_odt_to_md --input-dir data/raw --output-dir data/markdown
```

After converting or editing documents, rebuild the retrieval indexes.

## Build Indexes

Resume indexing:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m src.index_builder
```

Clean rebuild:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m src.index_builder --rebuild
```

Local artifacts are intentionally ignored by Git:

- `chroma_db/`
- `bm25_index.json`
- `.cache/`
- `.env`

The Hugging Face Space deployment script uploads `chroma_db/` and `bm25_index.json` separately so the Space can run without rebuilding indexes at startup.

## Evaluate Retrieval

Run a small golden-query retrieval check:

```powershell
python -m scripts.evaluate_retrieval --top-k 5
```

The script checks whether expected legal document numbers and article titles appear in the top-k retrieval results.

## Deploy To Hugging Face Spaces

Login first:

```powershell
hf auth login --add-to-git-credential
```

Deploy:

```powershell
python -m scripts.deploy_hf_space --repo-id jasongyn/viet-legal-rag
```

The deploy script:

- uploads app/runtime files to the Space;
- uploads local indexes needed by the Space;
- reads local `.env`;
- stores API keys as Hugging Face Space secrets;
- does not upload `.env`.

## Configuration

Useful `.env` knobs:

```env
TOP_K_RETRIEVE=50
TOP_K_RERANK=20
MAX_SEARCH_QUERIES=4
EXACT_LOOKUP_SHORT_CIRCUIT=true
MAX_CONTEXT_ARTICLES=6
MAX_REFERENCE_ARTICLES=2
MAX_ARTICLE_CHARS=4000
MAX_CONTEXT_CHARS=18000
EMBED_BATCH_SIZE=8
MAX_CPU_CORES=12
```

- Increase `TOP_K_RETRIEVE` for more recall.
- Increase `TOP_K_RERANK` when you want more reranked articles kept.
- Keep `MAX_SEARCH_QUERIES` bounded to avoid too many reranker calls per user query.
- Reduce `MAX_CONTEXT_*` values to lower generator latency and token cost.

## Screenshots

Screenshots and example answers are useful for a portfolio README, but they should be added only after the public Space UI is stable. A good next step is to add `assets/screenshot-home.png` and one short example answer section.
