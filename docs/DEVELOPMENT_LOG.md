# Development Log

Chronological record of what was built, decisions made, and blockers encountered.

---

## 2026-07-21

### Phase 0 — Foundation

- Initialised `uv` project with `pyproject.toml` + `uv.lock`
- Created directory structure: `data/raw/`, `data/processed/`, `ingest/`, `rag/`, `evaluation/`, `monitoring/`, `app/`
- Added dependencies: `lxml`, `plotly`
- Created `config.py` — centralised env-var loading with `python-dotenv`
- Updated `.env.example` with all configurable vars (provider, model, chunking, retrieval)
- **Decision:** `dlt` skipped for MVP — 16 static sources don't need scheduled ingestion

### Phase 1 — Data Ingestion

**T1.1 Scraper (`ingest/scraper.py`):**
- Built per-domain extractors for 12 source sites using `BeautifulSoup` + `requests`
- Implemented exponential-backoff retry for 429 rate limits
- **Blocker:** `propertygenie.com.my` returned 429 consistently even after retries — excluded
- **Blocker:** `propcashflow/tools/legal-fee-calculator` and `suppiah_law` calculator pages had <50 chars of article text — excluded as expected (pure calculator widgets)
- **Result:** 16/18 sources scraped, ~235K chars total

**T1.2 Cleaner (`ingest/cleaner.py`):**
- Strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, `<noscript>` tags
- Removes promotional phrases (share prompts, newsletter CTAs)
- Normalises whitespace
- **Result:** 13.3% size reduction (235K → 204K chars)

**T1.3 Chunker (`ingest/chunker.py`):**
- Sentence-boundary splitting (not arbitrary token count)
- Configurable chunk_size (default 512) and overlap (default 50)
- Rough token estimation via whitespace split
- UUID-based IDs for unambiguous addressing
- **Result:** 81 chunks from 16 documents

**T1.4 Orchestrator (`ingest/ingest.py`):**
- Chains scrape → clean → chunk in a single `uv run python -m ingest.ingest`
- Prints per-category summary
- **Result:** 28s total runtime

### Phase 2 — RAG Pipeline

**T2.1/T2.2 Index builders (`rag/build_index.py`):**
- MinSearch keyword index over `title` + `text` fields
- Sentence-transformers (`all-MiniLM-L6-v2`) vector embeddings → shape (81, 384)
- Indexes saved to `data/processed/` for reuse

**T2.3 Search layer (`rag/search.py`):**
- 4 retrievers: `keyword`, `vector`, `hybrid` (alpha=0.5), `reranked` (cross-encoder)
- Min-max normalisation for hybrid score fusion
- Cross-encoder re-ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`

**T2.4 Prompts (`rag/prompts.py`):**
- 2 system prompt variants: `default` (detailed) and `concise`
- Query rewrite prompt for abbreviation expansion
- `build_prompt()` assembles chat message list

**T2.5 RAG orchestrator (`rag/rag_pipeline.py`):**
- `ask()` function with full pipeline: retrieve → prompt → LLM → answer
- Lazy index loading (cached as function attributes)
- Configurable retriever type, K, rerank, prompt template, model
- Query rewriting via LLM before retrieval
- **Blocker:** First run timed out downloading sentence-transformers model (100MB+). Cached on second run.

### Phase 3 — Evaluation

**T3.1 Ground truth (`evaluation/generate_ground_truth.py`):**
- Groups 81 chunks back into 16 articles, generates 3 Q&A pairs per article
- **Blocker:** Initial version generated per-chunk (81 API calls = timeout). Switched to per-article generation (16 calls).
- **Blocker:** Some articles exceeded Groq free-tier TPM limit (8K). Truncated input text to 3000 chars.
- **Blocker:** API calls hung indefinitely — added 30s timeout to OpenAI client.
- **Result:** 48 ground-truth pairs across 4 categories

**T3.4 Judge prompt (`evaluation/judge_prompt.txt`):**
- Scores 0-10 on faithfulness, relevance, completeness, citation correctness
- Returns JSON for programmatic parsing

**T3.2 Retrieval evaluation (`evaluation/evaluate_retrieval.py`):**
- Evaluated all 4 retrievers against 48 ground-truth pairs
- **Blocker:** `transformers==4.17.0` couldn't download cross-encoder model. Upgraded to 4.45.2, downgraded `tokenizers` from 0.23.1 to 0.20.3 to resolve dependency conflict.
- **Results:**
  - `keyword`: HR@1=0.3125, MRR=0.7309
  - `vector`: HR@1=0.5625, MRR=0.8171
  - `hybrid`: HR@1=0.5833, MRR=0.8069 ← **winner**
  - `reranked`: identical to hybrid (cross-encoder agreed with embeddings)
  - `hybrid+rewrite`: HR@1=0.6042, marginal improvement

**T3.5 Query rewriting evaluation:**
- Query rewrite improves HR@1 from 0.5833 → 0.6042 and finds 1 more relevant result (42/48 vs 41/48)
- **Decision:** Enable query rewriting — low cost, small but real improvement

**T3.3 Generation evaluation (`evaluation/evaluate_answers.py`):**
- Tested 3 configs on 5 Q&A pairs each:
  - `hybrid_k3_default`: **overall 7.45** ← **winner**
  - `hybrid_k5_default`: overall 6.30 (more noise)
  - `hybrid_k3_concise`: overall 6.35 (less faithful)
- **Decision:** `k=3` + `default` prompt

**T3.6 Final config (`evaluation/results/final_config.md`):**
- Hybrid retriever, K=3, default prompt, query rewriting enabled

---

## Blocker Summary

| Date | Blocker | Resolution |
|------|---------|------------|
| Jul 21 | `propertygenie.com.my` returns 429 | Excluded source |
| Jul 21 | Calculator pages have no article text | Excluded (expected behaviour) |
| Jul 21 | Sentence-transformers download timeouts | Model cached after first run |
| Jul 21 | Ground truth: per-chunk generation = 81 API calls | Switched to per-article (16 calls) |
| Jul 21 | Groq TPM limit exceeded for large articles | Truncated input to 3000 chars |
| Jul 21 | API call hangs on large inputs | Added 30s timeout to OpenAI client |
| Jul 21 | `cached_download` missing in `huggingface-hub` | Downgraded to `<0.24` |
| Jul 21 | Cross-encoder model download fails | Upgraded `transformers` to 4.45.2, downgraded `tokenizers` to 0.20.3 |
| Jul 21 | Inline `#` comments in `.env` break auth | Documented in `.env.example` |
| Jul 21 | Notebook 401 errors (env not loaded) | Fixed `load_dotenv` path relative to `config.py` |
| Jul 22 | `sqlite3.Connection.lastrowid` removed in Python 3.13 | Switched to `Cursor.lastrowid` |
| Jul 22 | Groq 429 rate limit during testing | Expected on free tier; add retries in production |
| Jul 22 | Docker image ~2.5 GB from PyTorch/sentence-transformers | Migrated to ONNX Runtime — image reduced to ~400 MB |

---
## 2026-07-22

### Phase 4 — Application & Monitoring

**T4.1 SQLite logger (`monitoring/logger.py`):**
- Auto-creates `queries` table with schema: id, timestamp, question, answer, sources (JSON), latency_ms, retriever, k, model, rewrite, prompt_template, feedback
- `log_query()` returns row id for feedback linking
- `set_feedback(query_id, ±1/0)` for user ratings
- `get_stats()` for aggregate metrics
- **Blocker:** `sqlite3.Connection.lastrowid` deprecated in Python 3.13 — used `Cursor.lastrowid` instead

**T4.2 Monitoring dashboard (`monitoring/dashboard.py`):**
- 5 Plotly charts: query volume (daily bar), latency distribution (histogram), feedback ratio (pie), top cited sources (horizontal bar), latency trend (scatter)
- Streamlit-powered, reads directly from `monitoring.db`

**T4.3 Streamlit chat UI (`app/app.py`):**
- Conversational interface with chat history (st.session_state)
- Sidebar showing current config
- Sources shown in expandable panels with scores
- Latency badge on each response
- Calls `ask()` from `rag_pipeline.py`

**T4.4 User feedback loop:**
- 👍/👎 buttons on every assistant message
- Calls `set_feedback()` → updates `monitoring.db`
- Feedback visible on dashboard pie chart

### Phase 5 — Packaging

**T5.1 Dockerfile:**
- `python:3.13-slim` base with `uv`
- Two-stage `uv sync` for layer caching of dependencies

**T5.2 docker-compose.yml:**
- Single service `app` exposing port 8501
- Named volume `data` for persistent SQLite DB
- Passes `.env` file for secrets

**T5.3 README:**
- Architecture overview, quick start, project structure, config reference, eval results, deployment guide

**T5.4 Cloud deployment:**
- Instructions added to README for GCP Cloud Run and Azure Container Apps

### Phase 6 — ONNX Migration

**T6.1 ONNX embedder (`rag/embedder.py`):**
- Replaced `sentence-transformers/all-MiniLM-L6-v2` (torch, 1.8 GB) with ONNX Runtime via `Xenova/all-MiniLM-L6-v2`
- New class `Embedder` with `encode()` / `encode_batch()` using mean pooling
- New class `CrossEncoder` with `predict()` for reranking pairs
- Identical embeddings (cos=1.0 across all 489 chunks) vs torch version
- Docker image reduced from ~2.5 GB to ~400 MB

**T6.2 Config toggle (`EMBEDDING_BACKEND`):**
- Added `EMBEDDING_BACKEND=onnx|torch` in `config.py` for revert safety
- `rag/build_index.py` updated to select embedder class based on config

**T6.3 Slimmed dependencies:**
- Moved `torch`, `sentence-transformers`, `lxml`, `psycopg` to optional dev group
- Runtime deps reduced to 9 packages
- Regenerated `uv.lock` with 70 packages (was >200)

**T6.4 Evaluation re-run:**
- Re-ran LLM-as-a-Judge generation evaluation with updated ONNX pipeline
- Results were more balanced — default and concise both scored 7.0 overall
- K=5 (7.15) outperformed K=3 (7.0) in overall score — switched `TOP_K` to 5
- Cross-encoder reranking confirmed: HR@1 0.4583 → 0.6042 (+32%)
- `final_config.md` updated with new numbers and retriever/reranker explanation

