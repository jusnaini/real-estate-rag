# ONNX Evaluation Plan

Replace `sentence-transformers` + `torch` + `transformers` (~2.2GB) with `onnxruntime` + `tokenizers` (~125MB) using the same `all-MiniLM-L6-v2` model exported to ONNX. Goal: identical retrieval quality with a Docker image that builds in ~2 min instead of 30+.

---

## Files Changed

| File | Change |
|---|---|
| `rag/embedder.py` | **New.** ONNX `Embedder` class for bi-encoder (embedding) + ONNX `CrossEncoder` class for re-ranking |
| `rag/build_index.py` | Replace `SentenceTransformer` import with `Embedder`. `build_vector_index` returns `(embeddings,)` only (embedder shared as singleton) |
| `rag/search.py` | Replace `SentenceTransformer` type hints with `Embedder`. `model.encode()` → `embedder.encode()`. Replace `CrossEncoder` (torch) with ONNX `CrossEncoder` |
| `rag/rag_pipeline.py` | Adjust `ask()` lazy loading to use singleton `Embedder` + singleton ONNX `CrossEncoder` instead of storing model alongside embeddings |
| `pyproject.toml` | Remove `sentence-transformers>=0.2.5.1`, `transformers>=4.38.0,<4.46.0`, and `huggingface-hub<0.24` pin. Add `tokenizers>=0.23.1` |
| `config.py` | Add `EMBEDDING_BACKEND=onnx` (default: `torch` for backward compat) |
| `Dockerfile` | Add `RUN uv run python -c "from rag.embedder import Embedder; Embedder()"` to download both ONNX models at build time |

**Total changes:** ~50 lines across 6 files. Everything else (evaluation scripts, monitoring, dashboard, prompts, app, Dockerfile CMD, docker-compose) stays untouched.

---

## ONNX Cross-Encoder (Re-ranking)

Re-ranking is kept as part of the project's best practices for additional course marks. The current torch-based `CrossEncoder` will be ported to ONNX using the same model (`cross-encoder/ms-marco-MiniLM-L-6-v2` exported to ONNX format from `Xenova/ms-marco-MiniLM-L-6-v2`).

The ONNX `CrossEncoder` follows the same pattern as the bi-encoder:
- Load `tokenizer.json` + `model.onnx` from `models/Xenova/ms-marco-MiniLM-L-6-v2/`
- Implement `predict(pairs)` → scores using ONNX inference
- Matches the existing `rerank()` API in `rag/search.py`

Note: current evaluation showed re-ranking adds latency with no retrieval gain (identical HR@1/MRR to hybrid). But it's kept active for code assessment purposes. The ONNX version will have the same performance characteristics without the torch dependency.

---

## What We Test

### 1. Embedding Quality (Unit)

Encode 5 fixed sentences with **both** backends and compare cosine similarity:

| Sentence | Expected |
|---|---|
| "How much stamp duty for a RM500k property?" | cosine > 0.999 |
| "What is MRTA?" | cosine > 0.999 |
| Random 3 from ground truth set | cosine > 0.999 |

If any sentence produces cosine < 0.99, investigate pooling or normalization mismatch.

### 1b. Cross-Encoder Quality (Unit)

Score 5 query-document pairs with **both** backends and compare scores:

| Pair | Expected |
|---|---|
| Same query and document (identical text) | score diff < 0.01 |
| Unrelated query and document | score diff < 0.01 |
| 3 realistic pairs from ground truth | score diff < 0.01 |

If any pair has score diff > 0.05, investigate ONNX cross-encoder implementation.

### 2. Retrieval Metrics (Evaluation)

Run `evaluate_retrieval.py` with `EMBEDDING_BACKEND=onnx` against the 48 ground-truth pairs. Compare all 5 retrievers:

| Retriever | Current HR@1 | ONNX HR@1 | Current MRR | ONNX MRR |
|-----------|-------------|-----------|-------------|----------|
| keyword | 0.3125 | No change | 0.7309 | No change |
| vector | 0.5625 | ≥ 0.560 | 0.8171 | ≥ 0.810 |
| hybrid | 0.5833 | ≥ 0.580 | 0.8069 | ≥ 0.800 |
| reranked | 0.5833 | ≥ 0.580 | 0.8069 | ≥ 0.800 |
| hybrid+rewrite | 0.6042 | ≥ 0.600 | 0.8056 | ≥ 0.800 |

All vector-based retrievers must be **within 0.01** (HR@1) and **within 0.01** (MRR) of current values. Keyword is unaffected (MinSearch only).

### 3. Generation Quality (LLM-as-a-Judge)

Run `evaluate_answers.py` on the 3 winning configs (`hybrid_k3_default`, `hybrid_k5_default`, `hybrid_k3_concise`) and compare:

| Config | Current Overall | ONNX Target |
|--------|----------------|-------------|
| hybrid_k3_default | 7.45 | ≥ 7.30 |
| hybrid_k5_default | 6.30 | ≥ 6.20 |
| hybrid_k3_concise | 6.35 | ≥ 6.20 |

### 4. Latency

Run the 8 test questions from `test/rag_test.ipynb`, measure average latency:

| Metric | Current | ONNX Target |
|--------|---------|-------------|
| Avg hybrid latency | ~2000ms | ≤ 2500ms |

ONNX may be slightly slower (no GPU) or slightly faster (less overhead). Neither is a concern as long as it stays under 3s.

### 5. Docker Build

| Metric | Current | Target |
|--------|---------|--------|
| Image size | ~2.5GB | ≤ 400MB |
| First build time | 30+ min | ≤ 3 min |

---

## Success Criteria

Green-lit to flip `EMBEDDING_BACKEND=onnx` as the new default if:

- [ ] Embedding cosine similarity ≥ 0.999 on all 5 test sentences
- [ ] HR@1 within 0.01 of current hybrid baseline
- [ ] MRR within 0.01 of current hybrid baseline
- [ ] Generation scores within 0.2 of current
- [ ] Image size ≤ 400MB

Otherwise, keep `EMBEDDING_BACKEND=torch` and investigate the ONNX pipeline.

---

## Revert Plan

A single `.env` change restores the current setup:

```
EMBEDDING_BACKEND=torch
```

No code changes, no git operations, no deployment rollback needed. Repeat evaluation to confirm metrics return to baseline.

---

## Order of Execution

1. Implement ONNX embedder + file changes (~15 min)
2. Run embedding quality test (step 1 above) (~1 min)
3. If pass → run retrieval evaluation (step 2) (~5 min)
4. If pass → run generation evaluation (step 3) (~10 min)
5. If pass → build Docker image, verify size + time (step 5) (~3 min)
6. Flip default to `onnx`, commit, deploy

If any step fails, flip `EMBEDDING_BACKEND=torch` and stop.
