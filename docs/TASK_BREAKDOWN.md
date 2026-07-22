# Task Breakdown — Real Estate RAG Assistant

**Version:** 1.0 (all phases complete)
**Inspired by:** LLM Zoomcamp 2026 Capstone

---

# Phase Overview

| Phase | Name | Tasks | Est. Effort | Status |
|-------|------|-------|-------------|--------|
| 0 | Foundation | 3 | small | ✅ Complete |
| 1 | Data Ingestion | 4 + 1 opt | medium | ✅ Complete |
| 2 | RAG Pipeline | 5 | large | ✅ Complete |
| 3 | Evaluation | 6 | large | ✅ Complete |
| 4 | Application + Monitoring | 4 | medium | ✅ Complete |
| 5 | Reproducibility & Bonus | 4 | medium | ✅ Complete |

**Total tasks:** 26 (27 including optional)

---

# Task Dependency Map

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
                │            │            │
                ▼            ▼            ▼
             T1.1-T1.5   T2.1-T2.5   T3.1-T3.6
                              │
                              ▼
                          T4.1-T4.4
                              │
                              ▼
                          T5.1-T5.4
```

---

# Phase 0 — Foundation

## T0.1 — Initialize project structure

**Description:** Create all directories and placeholder files per the project structure in SDD.

**Files:**
- `data/raw/`
- `data/processed/`
- `ingest/`
- `rag/`
- `evaluation/`
- `monitoring/`
- `app/`

**Dependencies:** None

**Acceptance criteria:**
- Directories exist and are git-tracked via `.gitkeep`
- `main.py` stub is replaced with module entry points

**Effort:** small

---

## T0.2 — Create shared configuration module

**Description:** Implement `config.py` that reads environment variables and exposes model/embedding/retrieval settings.

**File:** `config.py`

**Dependencies:** T0.1

**Acceptance criteria:**
- Reads `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY` from `.env`
- Exposes `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`, `TOP_K`, `CHUNK_SIZE` as constants
- Falls back to sensible defaults when vars are unset

**Effort:** small

---

## T0.3 — Set up dependencies

**Description:** Create `requirements.txt` with pinned versions of all dependencies.

**File:** `requirements.txt`

**Dependencies:** T0.1

**Acceptance criteria:**
- Includes: `openai`, `streamlit`, `minsearch`, `sentence-transformers`, `dlt`, `sqlite3` (stdlib), `plotly`
- Versions are pinned

**Effort:** small

---

# Phase 1 — Data Ingestion

## T1.1 — Implement web scraper

**Description:** Extract article content (title, body text, metadata) from the source URLs listed in SDD §7.1. Handle multiple site formats gracefully.

**File:** `ingest/scraper.py`

**Dependencies:** T0.1, T0.2

**Acceptance criteria:**
- Accepts a list of URLs grouped by category
- Returns structured dicts: `{"url", "source", "category", "title", "raw_html", "text"}`
- Handles HTTP errors and rate-limiting gracefully
- Saves raw output to `data/raw/`

**Effort:** medium

---

## T1.2 — Implement document cleaner

**Description:** Clean extracted HTML: remove ads, navigation, boilerplate, scripts, styles. Normalize whitespace.

**File:** `ingest/cleaner.py`

**Dependencies:** T1.1

**Acceptance criteria:**
- Strips HTML tags, scripts, styles, nav elements
- Normalizes Unicode and whitespace
- Returns clean `{"url", "source", "category", "title", "text"}`

**Effort:** small

---

## T1.3 — Implement text chunker

**Description:** Split cleaned documents into chunks. Support configurable chunk size and overlap.

**File:** `ingest/chunker.py`

**Dependencies:** T1.2

**Acceptance criteria:**
- Splits by sentence boundaries (not mid-sentence)
- Configurable `chunk_size` (default 512 tokens) and `chunk_overlap` (default 50 tokens)
- Output schema matches SDD §7.3
- Each chunk has `id`, `source`, `category`, `title`, `url`, `text`

**Effort:** small

---

## T1.4 — Implement ingestion orchestrator

**Description:** Wire scraper → cleaner → chunker into a single `ingest()` function. Save output to `data/processed/documents.json`.

**File:** `ingest/ingest.py`

**Dependencies:** T1.1, T1.2, T1.3

**Acceptance criteria:**
- Running `python -m ingest.ingest` processes all sources end-to-end
- Output is valid JSON array of chunk objects
- Prints summary: total sources, total chunks, errors per source

**Effort:** small

---

## T1.5 — [OPTIONAL] Implement dlt ingestion pipeline

**Description:** Wrap the ingestion flow as a dlt pipeline for automated, scheduled ingestion. Enables the full 2pt Ingestion score on the rubric.

**File:** `ingest/dlt_pipeline.py`

**Dependencies:** T1.4

**Acceptance criteria:**
- Defines a dlt pipeline resource for each source category
- Can run with `python -m ingest.dlt_pipeline`
- Logs pipeline run status (extracted rows, load time, errors)
- Supports incremental runs (skips already-ingested URLs)

**Effort:** medium

---

# Phase 2 — RAG Pipeline

## T2.1 — Implement MinSearch index builder

**Description:** Build a MinSearch index from `documents.json` for keyword-based retrieval.

**File:** `rag/build_index.py`

**Dependencies:** T1.4

**Acceptance criteria:**
- Loads chunks from `data/processed/documents.json`
- Builds MinSearch index over `["title", "text"]` fields
- Saves index state so it can be reloaded without reprocessing
- Function signature: `build_minsearch_index(documents) -> minsearch.Index`

**Effort:** small

---

## T2.2 — Implement vector index

**Description:** Generate embeddings using sentence-transformers and store them for cosine-similarity search.

**File:** `rag/build_index.py` (same file, new function)

**Dependencies:** T1.4

**Acceptance criteria:**
- Loads sentence-transformers model (configurable, default `all-MiniLM-L6-v2`)
- Generates embedding for each chunk
- Stores embeddings in memory (list of numpy arrays)
- Function signature: `build_vector_index(documents) -> (embeddings, model)`

**Effort:** medium

---

## T2.3 — Implement search layer

**Description:** Unified search module with functions for keyword search, vector search, hybrid search, and re-ranking.

**File:** `rag/search.py`

**Dependencies:** T2.1, T2.2

**Acceptance criteria:**

**Keyword search:**
- `search_keyword(index, query, k=5) -> list[dict]`
- Returns chunks with relevance scores

**Vector search:**
- `search_vector(embeddings, model, query, k=5) -> list[dict]`
- Cosine similarity search over stored embeddings

**Hybrid search:**
- `search_hybrid(index, embeddings, model, query, k=5, alpha=0.5) -> list[dict]`
- Weighted fusion of keyword and vector scores

**Re-ranking:**
- `rerank(query, results, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[dict]`
- Re-scores top-K results using a cross-encoder
- Returns re-ranked results with updated scores

**Effort:** large

---

## T2.4 — Define prompts

**Description:** Create prompt templates: system prompt, user prompt template, and query rewriting prompt.

**File:** `rag/prompts.py`

**Dependencies:** T0.2, T2.3

**Acceptance criteria:**

- **System prompt:** instructs assistant to answer only from context, cite sources, avoid legal advice
- **User prompt template:** f-string with `{question}` and `{context}` placeholders
- **Query rewrite prompt:** expands abbreviations, reformulates vague queries
- At least 2 variations of system prompt for evaluation (concise vs detailed)
- Function signature: `build_prompt(question, context, template="default") -> str`

**Effort:** small

---

## T2.5 — Implement RAG pipeline orchestrator

**Description:** Wire retrieval → prompt building → LLM generation into a single `ask()` function.

**File:** `rag/rag_pipeline.py`

**Dependencies:** T2.3, T2.4

**Acceptance criteria:**
- `ask(question, retriever="hybrid", k=5) -> dict{"answer", "sources", "latency"}`
- Accepts optional `system_prompt_template` and `model` override
- Returns structured response with answer text and source citations
- Handles LLM API errors gracefully with retry logic

**Effort:** medium

---

# Phase 3 — Evaluation

## T3.1 — Generate ground-truth dataset

**Description:** Create question-answer pairs from documents for retrieval and generation evaluation.

**File:** `evaluation/generate_ground_truth.py`

**Dependencies:** T1.4

**Acceptance criteria:**
- For each document, generate 3-5 questions using an LLM
- Each record: `{"question", "answer", "relevant_doc_ids", "category"}`
- Minimum 50 question-answer pairs
- Saved as `data/processed/ground_truth.json`

**Effort:** medium

---

## T3.2 — Evaluate retrieval approaches

**Description:** Compute Hit Rate and MRR for each retriever variant.

**File:** `evaluation/evaluate_retrieval.py`

**Dependencies:** T2.3, T3.1

**Acceptance criteria:**
- Evaluates 4 retrievers: keyword, vector, hybrid, re-ranked
- Computes Hit Rate@K and MRR@K for K=1,3,5
- Outputs comparison table to console and saves to `evaluation/results/retrieval_results.json`
- Identifies best-performing approach

**Effort:** medium

---

## T3.3 — Evaluate generation (LLM-as-a-Judge)

**Description:** Score generated answers on faithfulness, relevance, completeness, and citation correctness.

**File:** `evaluation/evaluate_answers.py`

**Dependencies:** T2.5, T3.1

**Acceptance criteria:**
- For each ground-truth Q, runs RAG pipeline and scores answer against rubric
- Uses an LLM judge prompt (read from `evaluation/judge_prompt.txt`)
- Computes average scores per configuration (chunk size, prompt template, top-K)
- Outputs comparison table and saves to `evaluation/results/answer_eval_results.json`

**Effort:** medium

---

## T3.4 — Define judge prompt

**Description:** Write the LLM-as-a-Judge prompt that scores answers on the 4 criteria.

**File:** `evaluation/judge_prompt.txt`

**Dependencies:** None (can be done in parallel with T3.1)

**Acceptance criteria:**
- Asks judge to rate 0-10 on faithfulness, relevance, completeness, citation correctness
- Includes instructions to penalize hallucination
- Includes example scoring

**Effort:** small

---

## T3.5 — Evaluate query rewriting

**Description:** Measure whether query rewriting before retrieval improves Hit Rate/MRR vs raw queries.

**File:** `evaluation/evaluate_retrieval.py` (extend T3.2)

**Dependencies:** T2.4, T3.2

**Acceptance criteria:**
- For each ground-truth Q, generate rewritten version using LLM
- Compare retrieval metrics (Hit Rate, MRR) for raw vs rewritten queries
- Report improvement (or degradation)

**Effort:** small

---

## T3.6 — Determine best configuration

**Description:** Synthesize results from T3.2, T3.3, and T3.5 into a final configuration recommendation.

**File:** Documented in `evaluation/results/final_config.md`

**Dependencies:** T3.2, T3.3, T3.5

**Acceptance criteria:**
- States chosen retriever (keyword/vector/hybrid/re-ranked)
- States chosen chunk size, prompt template, top-K
- States whether query rewriting is enabled
- Includes metrics justifying each choice

**Effort:** small

---

# Phase 4 — Application & Monitoring

## T4.1 — Implement logger

**Description:** Log every query, answer, retrieved sources, latency, and timestamp to SQLite.

**File:** `monitoring/logger.py`

**Dependencies:** T0.1

**Acceptance criteria:**
- `log_query(query, answer, sources, latency, feedback=None) -> None`
- Creates SQLite DB at `data/monitoring.db` on first run
- Schema: `id, timestamp, query, answer, sources_json, latency_ms, feedback_score`
- Thread-safe writes

**Effort:** small

---

## T4.2 — Implement monitoring dashboard

**Description:** Streamlit dashboard with 5+ charts reading from SQLite.

**File:** `monitoring/dashboard.py`

**Dependencies:** T4.1

**Acceptance criteria:**

Minimum 5 charts:
1. Total queries over time (line chart)
2. Average latency over time (line chart)
3. Top-10 most popular questions (bar chart)
4. User feedback score distribution (histogram)
5. Token usage over time (area chart)

**Effort:** medium

---

## T4.3 — Build Streamlit UI

**Description:** User-facing chat interface with the RAG pipeline.

**File:** `app/app.py`

**Dependencies:** T2.5, T4.1

**Acceptance criteria:**
- Chat input box at bottom
- Message history displayed as chat bubbles
- Each answer shows source citations (expandable)
- Thumbs up/down feedback button per answer
- Sidebar with: model selector, retriever type selector, link to monitoring dashboard
- Runs with `streamlit run app/app.py`

**Effort:** medium

---

## T4.4 — Wire feedback loop

**Description:** Pass user feedback from the UI into the logger.

**File:** `app/app.py` (extend T4.3)

**Dependencies:** T4.3, T4.1

**Acceptance criteria:**
- Clicking thumbs up/down triggers `log_query(..., feedback=score)`
- Feedback is immediately queryable in the dashboard
- UI shows confirmation feedback was recorded

**Effort:** small

---

# Phase 5 — Reproducibility & Bonus

## T5.1 — Create Dockerfile

**Description:** Docker image for the Streamlit app with all dependencies.

**File:** `Dockerfile`

**Dependencies:** T4.3

**Acceptance criteria:**
- Based on `python:3.11-slim`
- Installs all dependencies from `requirements.txt`
- Exposes port 8501
- CMD runs `streamlit run app/app.py`

**Effort:** small

---

## T5.2 — Create docker-compose

**Description:** Single `docker-compose up` that runs the app + optional services.

**File:** `docker-compose.yml`

**Dependencies:** T5.1

**Acceptance criteria:**
- Single service entry for the Streamlit app
- Mounts `data/` volume for persistence
- Port 8501 mapped to host
- One-command startup: `docker-compose up`

**Effort:** small

---

## T5.3 — Write README

**Description:** Comprehensive project documentation.

**File:** `README.md`

**Dependencies:** All above

**Acceptance criteria:**
- Problem description
- Architecture diagram (from SDD)
- Setup instructions (clone, .env, docker-compose up)
- Usage examples with screenshots
- Evaluation results summary
- Link to SDD and task breakdown

**Effort:** medium

---

## T5.4 — Deploy to cloud (bonus)

**Description:** Deploy the application to Hugging Face Spaces or a similar free tier.

**File:** `README.md` (add deployment section) + any cloud config

**Dependencies:** T5.1

**Acceptance criteria:**
- App is publicly accessible via URL
- README includes deployment section with link
- Works with free-tier constraints

**Effort:** medium

---

# Execution Order (Recommended)

```
Phase 0 ──────────────────────────────────────────────────
T0.1 → T0.2 → T0.3

Phase 1 ──────────────────────────────────────────────────
T1.1 → T1.2 → T1.3 → T1.4
   └──→ T1.5 (optional, parallel with T1.2-T1.4)

Phase 2 ──────────────────────────────────────────────────
T1.4 ──→ T2.1 ──→ T2.3 ──→ T2.4 ──→ T2.5
  └──→ T2.2 ──┘

Phase 3 ──────────────────────────────────────────────────
T3.4 ──→ T3.1 ──→ T3.2 ──→ T3.3 ──→ T3.6
                └──→ T3.5 ──┘

Phase 4 ──────────────────────────────────────────────────
T4.1 ──→ T4.2
  └──→ T4.3 ──→ T4.4

Phase 5 ──────────────────────────────────────────────────
T5.1 ──→ T5.2 ──→ T5.3
  └──→ T5.4 (bonus, parallel)
```

---

# Evaluation Criteria Mapping

| Criterion | Max Pts | Relevant Tasks |
|-----------|---------|----------------|
| Problem description | 2 | T5.3 (README) |
| Retrieval flow | 2 | T2.1, T2.2, T2.3, T2.5 |
| Retrieval evaluation | 2 | T3.1, T3.2, T3.6 |
| LLM evaluation | 2 | T3.1, T3.3, T3.4, T3.6 |
| Interface | 2 | T4.3 |
| Ingestion pipeline | 1 (script) / 2 (dlt) | T1.4, T1.5 (optional) |
| Monitoring | 2 | T4.1, T4.2, T4.4 |
| Containerization | 2 | T5.1, T5.2 |
| Reproducibility | 2 | T5.1, T5.2, T5.3 |
| Hybrid search (best practice) | 1 | T2.3 |
| Re-ranking (best practice) | 1 | T2.3 |
| Query rewriting (best practice) | 1 | T3.5 |
| Cloud deployment (bonus) | 2 | T5.4 |
| **Total possible** | **23** | |
