# Final Configuration

Selected based on retrieval evaluation (48 ground-truth pairs, 489 chunks) and generation evaluation (LLM-as-a-Judge on 5 pairs × 3 configs).

---

## 1. Backend: ONNX Runtime

Replaced `sentence-transformers` (torch) with ONNX Runtime via `Xenova/all-MiniLM-L6-v2`. Embeddings are identical to the torch version (cosine similarity = 1.000 across all 489 chunks). Docker image reduced from ~2.5 GB to ~400 MB.

---

## 2. Retriever: Hybrid + Rerank

The retriever (keyword/vector/hybrid) fetches candidate chunks, then the cross-encoder reranker re-scores and reorders them. Reranking is a post-retrieval step, not a replacement for the retriever.

| Retriever | HR@1 | HR@3 | HR@5 | MRR | Found |
|-----------|------|------|------|-----|-------|
| keyword | 0.3125 | 0.3958 | 0.4583 | 0.7795 | 22/48 |
| vector | 0.4167 | 0.6458 | 0.8125 | 0.6842 | 39/48 |
| hybrid | 0.4583 | 0.6667 | 0.7708 | 0.7428 | 37/48 |
| **reranked** | **0.6042** | **0.6667** | **0.7708** | **0.8509** | **37/48** |
| hybrid+rewrite | 0.4583 | 0.6667 | 0.7708 | 0.7428 | 37/48 |

**Decision:** Hybrid retrieval with cross-encoder reranking. Reranking lifts HR@1 from 0.4583 → 0.6042 (+32%). Query rewrite adds no benefit with current LLM.

---

## 3. Top-K: 5

| K | Overall | Faithfulness | Relevance | Completeness | Citation |
|---|---------|-------------|-----------|-------------|----------|
| 3 | 7.0 | 5.4 | 9.4 | 7.6 | **5.6** |
| **5** | **7.15** | **6.6** | **9.4** | **8.4** | 4.2 |

K=5 selected for best overall score (7.15), faithfulness (6.6), and completeness (8.4).

---

## 4. Prompt Template: Default

Defined in `rag/prompts.py` (`SYSTEM_PROMPTS` dict).

- **default** — detailed instructions: answer using only context, cite source titles, say if insufficient, no legal advice
- **concise** — shortened version: answer concisely, cite source, say if insufficient

| Template | Overall | Faithfulness | Relevance | Completeness | Citation |
|----------|---------|-------------|-----------|-------------|----------|
| **default** | **7.0** | 5.4 | **9.4** | 7.6 | **5.6** |
| concise | **7.0** | **6.4** | 9.2 | 7.6 | 4.8 |

Both templates score the same overall. Default is preferred for higher relevance and citation quality.

---

## Final Settings (in .env)

```
RETRIEVER_TYPE=hybrid
TOP_K=5
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_BACKEND=onnx
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

Reranking is enabled in the pipeline (activated by default in `ask()` via the config).
