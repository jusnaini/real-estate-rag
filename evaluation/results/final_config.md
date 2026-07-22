# Final Configuration

Selected based on retrieval evaluation (48 ground-truth pairs) and generation evaluation (LLM-as-a-Judge on 5 pairs × 3 configs).

---

## 1. Retriever: Hybrid

| Retriever | HR@1 | HR@3 | HR@5 | MRR | Found |
|-----------|------|------|------|-----|-------|
| keyword | 0.3125 | 0.4792 | 0.5625 | 0.7309 | 27/48 |
| vector | 0.5625 | 0.7917 | 0.8125 | 0.8171 | 39/48 |
| **hybrid** | **0.5833** | **0.7500** | **0.8542** | **0.8069** | **41/48** |
| reranked | 0.5833 | 0.7500 | 0.8542 | 0.8069 | 41/48 |
| hybrid+rewrite | 0.6042 | 0.7500 | 0.8750 | 0.8056 | 42/48 |

**Decision:** Hybrid (alpha=0.5). Reranked adds latency with no gain. Query rewrite enabled for marginal HR improvement.

---

## 2. Top-K: 3

| K | Overall | Faithfulness | Relevance | Completeness | Citation |
|---|---------|-------------|-----------|-------------|----------|
| **3** | **7.45** | 6.8 | 9.4 | 7.6 | 6.0 |
| 5 | 6.30 | 5.8 | 8.4 | 7.2 | 3.8 |

K=5 introduces noisier chunks, reducing faithfulness and citation quality.

---

## 3. Prompt Template: Default

| Template | Overall | Faithfulness | Relevance | Completeness | Citation |
|----------|---------|-------------|-----------|-------------|----------|
| **default** | **7.45** | 6.8 | 9.4 | 7.6 | 6.0 |
| concise | 6.35 | 5.0 | 8.8 | 6.4 | 5.2 |

Detailed system prompt produces more faithful and complete answers.

---

## Final Settings (in .env)

```
RETRIEVER_TYPE=hybrid
TOP_K=3
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

Query rewriting is enabled in the pipeline (activated by default in `ask()` via the config).
