"""Evaluate and compare all retriever variants on Hit Rate and MRR.

For each ground-truth question, a chunk is considered relevant if it
belongs to the same article (source + title) that the question was
generated from.
"""

import json
import sys
import time
import warnings

import numpy as np

from openai import OpenAI

import config
from rag.build_index import load_documents, build_minsearch_index, build_vector_index
from rag.prompts import rewrite_query
from rag.search import search_keyword, search_vector, search_hybrid, rerank

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

RETRIEVERS = ["keyword", "vector", "hybrid", "reranked"]
K_VALUES = [1, 3, 5]


def load_ground_truth():
    with open(config.GROUND_TRUTH_PATH) as f:
        return json.load(f)["pairs"]


def build_article_map(docs: list[dict]) -> dict:
    """Map each article (source + title) to its set of chunk IDs."""
    mapping = {}
    for d in docs:
        key = (d["source"], d["title"])
        mapping.setdefault(key, set()).add(d["id"])
    return mapping


def is_relevant(chunk_id: str, gt_item: dict, article_map: dict) -> bool:
    key = (gt_item["source"], gt_item["title"])
    return chunk_id in article_map.get(key, set())


def hit_rate(ranks: list[int], k: int) -> float:
    ranks = [r for r in ranks if r is not None]
    if not ranks:
        return 0.0
    return 1.0 if min(ranks) <= k else 0.0


def mrr(ranks: list[int]) -> float:
    ranks = [r for r in ranks if r is not None]
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def evaluate_retriever(
    retriever_name: str,
    kw_index,
    embeddings,
    vec_model,
    docs,
    gt_pairs: list[dict],
    article_map: dict,
) -> dict:
    print(f"  Evaluating: {retriever_name}", flush=True)
    all_ranks = []

    for item in gt_pairs:
        query = item["question"]

        if retriever_name == "keyword":
            results = search_keyword(kw_index, query, k=max(K_VALUES))
        elif retriever_name == "vector":
            results = search_vector(embeddings, vec_model, docs, query, k=max(K_VALUES))
        elif retriever_name == "hybrid":
            results = search_hybrid(kw_index, embeddings, vec_model, docs, query, k=max(K_VALUES))
        elif retriever_name == "reranked":
            try:
                results = search_hybrid(kw_index, embeddings, vec_model, docs, query, k=max(K_VALUES))
                results = rerank(query, results)
            except Exception as e:
                print(f"    [WARN] Rerank model unavailable: {e}", flush=True)
                results = []
        else:
            results = []

        rank = None
        for pos, r in enumerate(results, 1):
            if is_relevant(r["id"], item, article_map):
                rank = pos
                break
        all_ranks.append(rank)

    valid_ranks = [r for r in all_ranks if r is not None]
    results = {"retriever": retriever_name, "total": len(gt_pairs), "found": len(valid_ranks)}
    for k in K_VALUES:
        hr = np.mean([hit_rate([r], k) for r in all_ranks])
        results[f"hit_rate@{k}"] = round(hr, 4)
    results["mrr"] = round(np.mean([mrr([r]) for r in all_ranks if r is not None]), 4)
    return results


def main():
    print("Loading documents and building indexes...", flush=True)
    docs = load_documents()
    kw_index = build_minsearch_index(docs)
    embeddings, vec_model = build_vector_index(docs)
    article_map = build_article_map(docs)
    print(f"  {len(docs)} chunks, {len(article_map)} articles", flush=True)

    gt_pairs = load_ground_truth()
    print(f"  {len(gt_pairs)} ground-truth pairs\n", flush=True)

    all_results = []
    for ret in RETRIEVERS:
        r = evaluate_retriever(ret, kw_index, embeddings, vec_model, docs, gt_pairs, article_map)
        all_results.append(r)

    # --- Query rewriting evaluation ---
    print("\n  Evaluating: hybrid + query rewrite", flush=True)
    client = OpenAI(
        base_url=config.PROVIDER_CONFIGS[config.LLM_PROVIDER]["base_url"],
        api_key=config.PROVIDER_CONFIGS[config.LLM_PROVIDER]["api_key"],
        timeout=30,
    )
    rewrite_results = []
    for item in gt_pairs:
        rewrite_prompt = rewrite_query(item["question"])
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.0,
            )
            rewritten = resp.choices[0].message.content.strip()
        except Exception:
            rewritten = item["question"]

        results = search_hybrid(kw_index, embeddings, vec_model, docs, rewritten, k=max(K_VALUES))
        rank = None
        for pos, r in enumerate(results, 1):
            if is_relevant(r["id"], item, article_map):
                rank = pos
                break
        rewrite_results.append(rank)

    rw = {"retriever": "hybrid+rewrite", "total": len(gt_pairs),
          "found": sum(1 for r in rewrite_results if r is not None)}
    for k in K_VALUES:
        rw[f"hit_rate@{k}"] = round(np.mean([hit_rate([r], k) for r in rewrite_results]), 4)
    rw["mrr"] = round(np.mean([mrr([r]) for r in rewrite_results if r is not None]), 4)
    all_results.append(rw)

    print(f"\n{'='*70}")
    print(f"{'Retriever':<12} {'HR@1':<8} {'HR@3':<8} {'HR@5':<8} {'MRR':<8} {'Found':<6}")
    print(f"{'='*70}")
    for r in all_results:
        print(
            f"{r['retriever']:<12} {r['hit_rate@1']:<8.4f} {r['hit_rate@3']:<8.4f} "
            f"{r['hit_rate@5']:<8.4f} {r['mrr']:<8.4f} {r['found']}/{r['total']:<4}"
        )
    print(f"{'='*70}")

    out_path = config.PROCESSED_DIR / "retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
