# Implementation Decisions

Records the why behind architectural and code choices.

---

## 1. Project & Tooling

### Python 3.13 + uv

Chose `uv` over pip/poetry for dependency management because the project was initialized with it. `uv add <pkg>` handles transitive dependency resolution and lockfile management automatically.

### Module-level config (config.py)

All environment variables are read once at import time in a single `config.py` module. Every other component imports from it rather than calling `os.getenv()` directly. This means changing a model, chunk size, or API key is a single edit in `.env` with zero code changes.

### Why not pydantic-settings

Slimmer dependency. `os.getenv` + `load_dotenv` is sufficient for ~20 config vars.

---

## 2. Ingestion Pipeline

### Scraper: per-domain extractors

Each source site has different HTML structure. Rather than one fragile generic parser, each domain gets its own extractor function (`extract_iproperty`, `extract_propcashflow`, etc.) with hand-picked CSS selectors. A shared `_find_and_clean_article()` helper eliminates repetition across extractors.

Decided **against** a generic "grab the `<article>` tag" approach after discovering that some sites use `<div class="content">` or `<main>` while others nest content inside calculator widgets.

### Excluded sources

Two sources were excluded:
1. **propcashflow/tools/legal-fee-calculator** — calculator page with <50 chars of article text, no educational content
2. **suppiah_law/legal-fee-stamp-duty-calculator** — same reason, pure calculator widget
3. **propertygenie.com.my** — returned 429 on every request despite exponential backoff retries

These are safe to lose because other sources cover the same categories (stamp duty, insurance).

### Chunking: sentence-boundary + token budget + heading detection

Key decisions:
- **Split at sentence boundaries** (not arbitrary token count) — avoids mid-sentence cuts that break meaning
- **Heading detection**: `split_sentences()` splits on both `[.!?]` and `\n`. Combined with the cleaner inserting `\n` before headings, section titles become their own sentences and trigger new chunk groups. This produces 489 focused chunks (up from 81) with each topic section isolated.
- **Whitespace-split for token estimation** — not a full tokenizer. Fast and close enough for chunk boundary decisions
- **Overlap of 50 tokens** — prevents information loss at chunk seams, especially for list items or multi-sentence explanations
- **UUID-based IDs** instead of sequential integers — makes each chunk unambiguously addressable during evaluation

### Cleaner: structure preservation

- `clean_html()` inserts `\n` before `<h1>`–`<h6>`, `<p>`, `<li>`, `<br>` via BeautifulSoup before `get_text()`. Without this, all headings and paragraphs merge into one continuous string.
- `normalize_whitespace()` preserves newlines (collapses horizontal whitespace only). Regex patterns that depend on `\n` as a boundary work correctly.
- `remove_boilerplate_phrases()` removed `re.DOTALL` flag and shortened greedy `.*?` patterns. Without this fix, "Share this" at the start of an article consumed the entire 34K text body leaving only 322 chars.

### dlt pipeline (optional, skipped)

The design includes a `dlt_pipeline.py` wrapper for automated ingestion. Skipped for MVP because:
- 16 static sources rarely change — no recurring ingestion schedule needed
- Python scripts are simpler to debug during development
- Can add dlt later without changing any ingestion logic

---

## 3. RAG Pipeline

### Retrievers: four strategies

| Retriever | Technique | Rationale |
|-----------|-----------|-----------|
| `keyword` | MinSearch (TF-IDF) | Baseline — matches exact terms. Fast, zero dependencies beyond minsearch. |
| `vector` | all-MiniLM-L6-v2 + cosine similarity | Captures semantic meaning. Works when user asks conceptually (e.g. "what costs should I prepare?") without exact keyword overlap. |
| `hybrid` | Weighted fusion (alpha=0.5) | Combines both strengths. Default choice because property questions can be either keyword-heavy (stamp duty tiers) or semantic (MRTA vs MLTA). |
| `reranked` | Cross-encoder (ms-marco-MiniLM-L-6-v2) | Highest precision for evaluation. Slow (2-5s per query) so not the default, but useful for comparing against other retrievers in evaluation. |

### Normalisation for hybrid fusion

Keyword and vector scores live on different scales. When fusing, each score set is min-max normalised to [0, 1] so the `alpha` weight produces meaningful blends.

### Why 3 search results (k=3)

K=5 selected for best overall score (7.15), faithfulness (6.6), and completeness (8.4) in LLM-as-a-Judge evaluation. K is configurable via `TOP_K` in `.env`.

### Lazy index loading

The `ask()` function caches indexes as function attributes (`ask._docs`, `ask._kw_index`, etc.) on first call rather than requiring a separate initialisation step. This makes the function self-contained for notebooks and testing.

### LLM-as-a-Judge evaluation (Phase 3 design)

Generation quality uses an LLM judge prompt (in `evaluation/judge_prompt.txt`) that scores each answer on 4 axes: faithfulness, relevance, completeness, citation correctness. This follows the course's evaluation module approach.

---

## 4. Prompt Design

### Two system prompt variants

| Variant | Style | Use case |
|---------|-------|----------|
| `default` | Detailed instructions, asks for source citation | Production / user-facing |
| `concise` | Short instructions | Evaluation — isolates whether verbosity affects LLM-as-a-Judge scores |

### Query rewriting

Expands abbreviations (MRTA → Mortgage Reducing Term Assurance) before retrieval. Useful for keyword/hybrid retrievers where the exact expanded term exists in the knowledge base but the abbreviation doesn't.

---

## 5. Search Score Semantics

| Retriever | Scale | Meaning |
|-----------|-------|---------|
| keyword | Relative ranking | TF-IDF-like term overlap, only comparable within result set |
| vector | [0, ~1] | Cosine similarity — semantic closeness |
| hybrid | [0, 1] (normalised) | Weighted blend of the above two |
| reranked | Unbounded | Cross-encoder raw relevance logit |

Scores should never be compared across different queries — they only indicate which chunk was most relevant *for that specific question*.

---

## 6. Evaluation (design)

### Ground truth dataset

Generated by having the LLM produce 3-5 question-answer pairs per document (~50+ pairs total). Each pair includes `relevant_doc_ids` so both retrieval and generation can be evaluated against the same ground truth.

### Retrieval metrics

Hit Rate@K and MRR@K are computed for each retriever variant at K=1,3,5. The cross-encoder reranker is expected to outperform keyword/vector/hybrid in MRR, but the question is whether the quality gain justifies the ~2-5s latency cost.

### Generation metrics

LLM-as-a-Judge scores each answer on 4 criteria (0-10). Multiple configurations (chunk size, prompt template, K) are compared to select the optimal combination.

---

## 7. Monitoring

### Logging to SQLite

SQLite was chosen over a full observability stack (Grafana, etc.) because:
- Zero infrastructure — just a file on disk
- Sufficient for a single-user application
- Dashboard can read directly from it with simple SQL queries

### Token usage and cost tracking

Every LLM call captures `usage` from the OpenAI-compatible response, including prompt/completion/total tokens. Cost is calculated using provider-specific per-token rates (default: $0.15/1M input, $0.60/1M output for Groq). Stored alongside the query in SQLite for dashboard visualisation.

### 7 dashboard charts

1. Query volume over time
2. Latency distribution
3. User feedback ratio
4. Top cited sources
5. Token usage over time
6. Cost over time
7. Cost distribution per query

Plus 4 KPI cards: total queries, avg latency, total cost, avg tokens.

---

## 8. Reproducibility

### Docker + docker-compose

Single `docker-compose up` runs the Streamlit app. Volumes mount `data/` and the `.env` file so the knowledge base and configuration persist across container restarts.

### Dependencies pinned

All dependency versions are locked in `uv.lock` and installed via `uv sync --frozen` for deterministic builds.

---

## 9. Version Alignment Fix

### huggingface-hub downgrade

`uv.lock` initially resolved `huggingface-hub==0.36.2`, which no longer exports `cached_download`. But `sentence-transformers==2.2.2` still imports it. Pinned `huggingface-hub<0.24` to restore the removed API. A cleaner fix would be upgrading sentence-transformers, but version conflicts with `transformers==4.17.0` and `tokenizers==0.23.1` made that path more invasive.

### Groq model id format

Groq uses the full model ID `openai/gpt-oss-120b` (with slash prefix) rather than a plain name. This differs from conventional OpenAI-compatible providers where the model is just `gpt-4o` or similar. The `PROVIDER_CONFIGS` map keeps provider-specific variations isolated to `config.py`.

### Streamlit script imports

`streamlit run app/app.py` executes scripts from the script's directory, not the project root. Direct imports like `import config` fail because `app/` is added to `sys.path` instead of the project root. Fixed by inserting `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at the top of both `app/app.py` and `monitoring/dashboard.py`.

### Docker build with uv

The Dockerfile uses `uv sync --frozen` to skip dependency re-resolution and install directly from `uv.lock`. This avoids the re-resolution hang and produces deterministic builds. A single `COPY . .` + `uv sync` step is used (no two-stage dependency caching) because the heavy ML packages (torch, transformers) dominate install time regardless.
