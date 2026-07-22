# Real Estate RAG Assistant

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![Built with](<https://img.shields.io/badge/built%20with-uv-5832C4>)](https://docs.astral.sh/uv/)
[![LLM Zoomcamp](<https://img.shields.io/badge/LLM%20Zoomcamp-2026-green>)](https://github.com/DataTalksClub/llm-zoomcamp)

An **AI assistant** that focuses on assisting users especially the first-time buyers with understanding common questions related to the property purchasing process in Malaysia 🇲🇾.

---

## Table of Contents

- [Real Estate RAG Assistant](#real-estate-rag-assistant)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Problem Statement](#problem-statement)
  - [Solution](#solution)
  - [Scope](#scope)
  - [Demo](#demo)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Run](#run)
    - [Docker](#docker)
  - [Makefile Targets](#makefile-targets)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
  - [Example Questions](#example-questions)
  - [Retrieval Strategies](#retrieval-strategies)
  - [Evaluation Results](#evaluation-results)
    - [Retrieval Evaluation (48 ground-truth pairs)](#retrieval-evaluation-48-ground-truth-pairs)
    - [Generation Evaluation (LLM-as-a-Judge)](#generation-evaluation-llm-as-a-judge)
  - [Monitoring](#monitoring)
  - [Deployment](#deployment)
    - [Docker (local)](#docker-local)
    - [Google Cloud Run](#google-cloud-run)
    - [Azure Container Apps](#azure-container-apps)
  - [Troubleshooting](#troubleshooting)
  - [Development](#development)

---

## Overview

**Real Estate RAG Assistant** is an end-to-end **Retrieval-Augmented Generation (RAG)** application developed as the capstone project for [**LLM Zoomcamp 2026**](https://github.com/DataTalksClub/llm-zoomcamp).

Rather than relying solely on a LLM's internal knowledge, the assistant retrieves relevant information from a curated knowledge base before generating responses. This grounding process improves answer transparency, reduces hallucinations, enables source attribution, and allows the knowledge base to be updated independently of the underlying language model without requiring model retraining.

Malaysian residential property purchasing is selected as the demonstration domain because the relevant information spread across property guides, financial resources, legal references, insurance articles, etc. This makes it an ideal real-world use case for illustrating how RAG can consolidate scattered knowledge into a single conversational interface.

Although the current implementation focuses on Malaysian real estate, the overall architecture is domain-agnostic. By replacing the underlying document corpus, the same pipeline can be adapted to other knowledge-intensive domains such as healthcare, legal services, insurance, education, or enterprise knowledge management.

---

## Problem Statement

Buying a home is one of the biggest financial decisions most people will ever make.For first-time homebuyers in Malaysia, the purchasing journey involves understanding many interconnected topics, including:

- purchasing procedures
- financing options
- required documentation
- legal processes
- transaction costs
- insurance coverage

While this information is publicly available, it is distributed across many sources. Finding the right information often requires visiting multiple sources, comparing conflicting advice, and determining which information is still current and trustworthy.

This creates two key challenges:

1. **Information fragmentation** – users spend significant time searching across multiple websites before finding a complete answer.
2. **Trustworthiness** – not every search result comes from an authoritative or verified source, making it difficult for users to know which information they should rely on when making important financial decisions.

---

## Solution

A complete RAG pipeline that transforms curated Malaysian real estate resources into a searchable knowledge base for natural-language Q&A. It covers:

- Document ingestion and preprocessing
- Knowledge base construction
- Retrieval pipeline
- Prompt engineering
- LLM-based answer generation
- Retrieval evaluation (Hit Rate, MRR)
- Answer evaluation using LLM-as-a-Judge
- Usage monitoring and logging
- Interactive Streamlit application

Users can ask natural language questions such as:

- *What documents do I need before applying for a home loan?*
- *How is stamp duty calculated?*
- *What's the difference between MRTA and MLTA?*
- *What costs should I prepare before signing the SPA?*

Every answer cites the vetted resource(s) it was generated from, so the user can verify it rather than take it on faith.

---

## Scope

This project is implemented as an MVP covering four knowledge areas:

- **Buying Process** — purchasing workflow, required documents, SPA, ownership transfer
- **Financing** — home loan applications, DSR, LTV, affordability
- **Legal & Transaction Costs** — stamp duty, legal fees, Memorandum of Transfer (MOT)
- **Insurance** — MRTA, MLTA, fire insurance, homeowner protection

The knowledge base is built from a curated set of reputable Malaysian property education sites and financial/legal guidance articles, rather than exhaustive government publications. The ingestion pipeline is designed to be extensible, allowing additional sources to be incorporated with minimal changes to the overall architecture.

---

## Demo

![Chat Interface](image/README/1784696078806.png)

![Monitoring Dashboard](image/README/1784696294082.png)

---

## Architecture

```
User → Streamlit UI → RAG Pipeline → LLM (Groq)
                           ↓
                    Retrieval (hybrid keyword + vector)
                           ↓
                    MinSearch index + sentence-transformers
                           ↓
                    16 scraped property guides (489 chunks)
```

---

## Quick Start

### Prerequisites

- Python 3.13+ with [`uv`](https://docs.astral.sh/uv/)
- API key from a supported provider (default: Groq)

### Setup

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
uv sync

> The `.gitignore` excludes runtime data (`*.db`, generated embeddings), environment files (`.env`, `.venv`), and OS artifacts (`.DS_Store`) from version control.
```

### Run

```bash
# Full ingestion pipeline (scrape → clean → chunk 16 sources)
uv run python -m ingest.ingest

# Launch the chat app
make chat

# Launch the monitoring dashboard
make dashboard
```

### Docker

```bash
docker compose up --build
```

> **Note:** The first query in Docker will be slow (~30s) because it downloads the embedding model inside the container. Subsequent queries are fast (~1-2s). The embedding model also downloads on every container restart since it's not persisted in the volume.

---

## Makefile Targets

| Target                      | Command                                            | Description                                   |
| --------------------------- | -------------------------------------------------- | --------------------------------------------- |
| `make chat`               | `uv run streamlit run app/app.py`                | Launch the chat UI at localhost:8501          |
| `make dashboard`          | `uv run streamlit run monitoring/dashboard.py`   | Launch monitoring dashboard at localhost:8501 |
| `make ingest`             | `uv run python -m ingest.ingest`                 | Run the full ingestion pipeline               |
| `make evaluate-retrieval` | `uv run python -m evaluation.evaluate_retrieval` | Compare retriever performance (HR@K, MRR)     |
| `make evaluate-answers`   | `uv run python -m evaluation.evaluate_answers`   | Score answer quality via LLM-as-a-Judge       |

---

## Project Structure

```
├── app/app.py                    # Streamlit chat UI
├── monitoring/
│   ├── logger.py                 # SQLite query logger (tokens, cost, feedback)
│   └── dashboard.py              # Plotly dashboard (7 charts + KPI cards)
├── ingest/
│   ├── scraper.py                # Web scraping (16 sources)
│   ├── cleaner.py                # HTML-to-text cleaning
│   ├── chunker.py                # Sentence-boundary chunking
│   └── ingest.py                 # Orchestrator
├── rag/
│   ├── build_index.py            # MinSearch + vector indexes
│   ├── search.py                 # 4 retrievers (keyword/vector/hybrid/reranked)
│   ├── prompts.py                # System prompt templates + query rewrite
│   └── rag_pipeline.py           # ask() orchestrator
├── evaluation/
│   ├── judge_prompt.txt          # LLM-as-a-Judge scoring rubric
│   ├── generate_ground_truth.py  # 48 Q&A pairs from 16 articles
│   ├── evaluate_retrieval.py     # Retriever comparison (HR@K, MRR)
│   ├── evaluate_answers.py       # Generation quality via LLM judge
│   └── results/final_config.md   # Best config recommendation
├── data/
│   ├── raw/                      # Scraped HTML
│   └── processed/                # Cleaned chunks, indexes, ground truth
├── config.py                     # Centralised env-var configuration
└── docs/
    ├── DEVELOPMENT_LOG.md            # Detailed development history
    ├── IMPLEMENTATION_DECISIONS.md   # Why behind architectural choices
    ├── ONNX_EVALUATION_PLAN.md       # Plan for switching to ONNX runtime
    ├── SDD.md                       # Software Design Document
    └── TASK_BREAKDOWN.md             # Execution plan
```

---

## Configuration

All settings via `.env`:

| Variable            | Default                                  | Description                                                             |
| ------------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| `LLM_PROVIDER`    | `groq`                                 | Provider:`groq`, `openrouter`, `cerebras`, `gemini`, `openai` |
| `LLM_MODEL`       | `openai/gpt-oss-120b`                  | Model name for the provider                                             |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`                     | Sentence-transformer model for embeddings                               |
| `RERANK_MODEL`    | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-ranking                                            |
| `RETRIEVER_TYPE`  | `hybrid`                               | Retriever:`keyword`, `vector`, `hybrid`                           |
| `TOP_K`           | `5`                                    | Number of chunks to retrieve                                            |
| `CHUNK_SIZE`      | `512`                                  | Tokens per chunk                                                        |
| `CHUNK_OVERLAP`   | `50`                                   | Overlap between chunks                                                  |

---

## Example Questions

- "How much stamp duty for a RM500k property?"
- "What is the difference between MRTA and MLTA?"
- "What documents do I need to apply for a home loan?"
- "How does DSR affect my loan eligibility?"
- "What are the steps to buy my first house in Malaysia?"
- "What legal fees should I expect when buying property?"

---

## Retrieval Strategies

The pipeline implements and compares four retrieval approaches:

| Strategy           | Technique                                | When it works best                                                         |
| ------------------ | ---------------------------------------- | -------------------------------------------------------------------------- |
| **Keyword**  | MinSearch (TF-IDF)                       | Exact term matches — stamp duty tiers, specific law names                 |
| **Vector**   | `all-MiniLM-L6-v2` + cosine similarity | Semantic meaning — "what costs should I prepare?" without keyword overlap |
| **Hybrid**   | Weighted fusion (alpha=0.5)              | Combines both strengths — default choice                                  |
| **Reranked** | Cross-encoder on hybrid results          | Highest precision (2-5s extra latency) — used for evaluation comparison   |

**Winner:** Hybrid. Reranked adds latency with identical metrics. Query rewriting (expanding abbreviations like MRTA → Mortgage Reducing Term Assurance) provides a marginal improvement.

## Evaluation Results

### Retrieval Evaluation (48 ground-truth pairs)

| Retriever        | HR@1             | HR@3             | HR@5             | MRR              | Found           |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | --------------- |
| keyword          | 0.3125           | 0.4792           | 0.5625           | 0.7309           | 27/48           |
| vector           | 0.5625           | 0.7917           | 0.8125           | 0.8171           | 39/48           |
| **hybrid** | **0.5833** | **0.7500** | **0.8542** | **0.8069** | **41/48** |
| reranked         | 0.5833           | 0.7500           | 0.8542           | 0.8069           | 41/48           |
| hybrid+rewrite   | 0.6042           | 0.7500           | 0.8750           | 0.8056           | 42/48           |

**Decision:** Hybrid (alpha=0.5). Reranked adds latency with no gain. Query rewrite enabled for marginal HR improvement.

### Generation Evaluation (LLM-as-a-Judge)

| Config                      | Overall        | Faithfulness | Relevance | Completeness | Citation |
| --------------------------- | -------------- | ------------ | --------- | ------------ | -------- |
| **hybrid_k3_default** | **7.45** | 6.8          | 9.4       | 7.6          | 6.0      |
| hybrid_k5_default           | 6.30           | 5.8          | 8.4       | 7.2          | 3.8      |
| hybrid_k3_concise           | 6.35           | 5.0          | 8.8       | 6.4          | 5.2      |

**Winner:** Hybrid retriever, K=3, default prompt, query rewriting enabled.

Final config documented in [`evaluation/results/final_config.md`](evaluation/results/final_config.md).

---

## Monitoring

Every query is logged to `data/monitoring.db` with:

- Latency, tokens used (prompt/completion/total), and cost
- User feedback (👍/👎)
- Retrieved source citations

Run the dashboard: `make dashboard`

7 charts: query volume, latency distribution, feedback ratio, top sources, token usage, cost over time, cost distribution.

> **Note for Docker users:** The Docker container stores `monitoring.db` inside a named volume, not on your host filesystem. The local `make dashboard` reads from `./data/monitoring.db` on your machine and won't show queries answered inside Docker. To share the database, replace `data:/app/data` with `./data:/app/data` in `docker-compose.yml`.

---

## Deployment

### Docker (local)

```bash
docker compose up --build
```

### Google Cloud Run

```bash
export PROJECT_ID="your-gcp-project"
gcloud builds submit --tag gcr.io/$PROJECT_ID/real-estate-rag
gcloud run deploy real-estate-rag \
  --image gcr.io/$PROJECT_ID/real-estate-rag \
  --set-env-vars="GROQ_API_KEY=your_key" \
  --allow-unauthenticated
```

### Azure Container Apps

```bash
az containerapp up \
  --name real-estate-rag \
  --source . \
  --env-vars GROQ_API_KEY=your_key
```

---

## Troubleshooting

| Problem                                            | Cause                                          | Fix                                                          |
| -------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| Auth error / 401 from API                          | `.env` has inline comments after the value   | Remove any`# comment` from the same line as `API_KEY=`   |
| `ImportError: cannot import name 'CrossEncoder'` | Wrong Python environment                       | Run with`uv run` to use the project's `.venv`            |
| First Docker query is very slow                    | Embedding model downloaded at runtime          | Wait ~30s for model to download; subsequent queries are fast |
| `huggingface-hub` version error                  | Version mismatch with`sentence-transformers` | Ensure`huggingface-hub<0.24` is in pinned deps             |

---

## Development

- [`docs/SDD.md`](docs/SDD.md) — Software Design Document
- [`docs/TASK_BREAKDOWN.md`](docs/TASK_BREAKDOWN.md) — phased execution plan (26 tasks across 6 phases)
- [`docs/IMPLEMENTATION_DECISIONS.md`](docs/IMPLEMENTATION_DECISIONS.md) — why behind architectural choices
- [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) — chronological build log with blockers and decisions
- [`docs/ONNX_EVALUATION_PLAN.md`](docs/ONNX_EVALUATION_PLAN.md) — plan for switching to ONNX runtime
