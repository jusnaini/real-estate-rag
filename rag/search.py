"""Unified search layer supporting keyword, vector, hybrid, and re-ranked retrieval.

Exports
-------
search_keyword(index, query, k) -> list[dict]
search_vector(embeddings, model, documents, query, k) -> list[dict]
search_hybrid(index, embeddings, model, documents, query, k, alpha) -> list[dict]
rerank(query, results, model_name) -> list[dict]
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

import config


def search_keyword(index, query: str, k: int = None) -> list[dict]:
    """Keyword search via MinSearch.

    Parameters
    ----------
    index : minsearch.Index
        The keyword index.
    query : str
        User query.
    k : int or None
        Number of results (default from config).

    Returns
    -------
    list[dict]
        Top-k results with ``id``, ``text``, ``source``, ``title``, plus ``score``.
    """
    if k is None:
        k = config.TOP_K
    results = index.search(query, num_results=k)
    for r in results:
        r["score"] = r.pop("_score", 0.0)
    return results


def search_vector(
    embeddings: np.ndarray,
    model: SentenceTransformer,
    documents: list[dict],
    query: str,
    k: int = None,
) -> list[dict]:
    """Vector (semantic) search via sentence-transformer cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        Pre-computed document embeddings, shape ``(N, D)``.
    model : SentenceTransformer
        The embedding model (used to embed the query).
    documents : list[dict]
        Original document chunks (aligned with embeddings rows).
    query : str
        User query.
    k : int or None
        Number of results (default from config).

    Returns
    -------
    list[dict]
        Top-k results with ``score``.
    """
    if k is None:
        k = config.TOP_K
    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    results = []
    for idx in top_indices:
        doc = dict(documents[idx])
        doc["score"] = float(sims[idx])
        results.append(doc)
    return results


def search_hybrid(
    index,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    documents: list[dict],
    query: str,
    k: int = None,
    alpha: float = 0.5,
) -> list[dict]:
    """Hybrid search fusing keyword and vector scores with weighted alpha.

    Scores are normalised to [0, 1] before fusion.

    Parameters
    ----------
    index : minsearch.Index
    embeddings : np.ndarray
    model : SentenceTransformer
    documents : list[dict]
    query : str
    k : int or None
    alpha : float
        Weight for keyword score (1 - alpha for vector). Default 0.5.

    Returns
    -------
    list[dict]
        Top-k results with combined ``score``.
    """
    if k is None:
        k = config.TOP_K
    kw_results = search_keyword(index, query, k=k * 2)
    vec_results = search_vector(embeddings, model, documents, query, k=k * 2)

    kw_scores = {r["id"]: r["score"] for r in kw_results}
    vec_scores = {r["id"]: r["score"] for r in vec_results}

    all_ids = set(kw_scores.keys()) | set(vec_scores.keys())

    def _normalise(scores_dict):
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 0.5 for k in scores_dict}
        return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}

    kw_norm = _normalise(kw_scores)
    vec_norm = _normalise(vec_scores)

    fused = []
    for doc_id in all_ids:
        combined = alpha * kw_norm.get(doc_id, 0.0) + (1 - alpha) * vec_norm.get(doc_id, 0.0)
        source_doc = None
        for r in kw_results + vec_results:
            if r["id"] == doc_id:
                source_doc = r
                break
        if source_doc:
            d = dict(source_doc)
            d["score"] = combined
            fused.append(d)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]


def rerank(query: str, results: list[dict], model_name: str = None) -> list[dict]:
    """Re-rank results using a cross-encoder model.

    Parameters
    ----------
    query : str
        Original user query.
    results : list[dict]
        Candidate documents (must have a ``text`` key).
    model_name : str or None
        Cross-encoder model name (default from config).

    Returns
    -------
    list[dict]
        Re-ranked results with updated ``score``.
    """
    if model_name is None:
        model_name = config.RERANK_MODEL
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, r["text"]) for r in results]
    scores = cross_encoder.predict(pairs)
    for r, s in zip(results, scores):
        r["score"] = float(s)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
