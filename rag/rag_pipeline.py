"""End-to-end RAG pipeline orchestrator.

Exports
-------
ask(question, retriever, k, rerank_results, prompt_template, model, rewrite) -> dict
"""

import time
from openai import OpenAI

import config
from monitoring.logger import calculate_cost
from rag.build_index import load_documents, build_minsearch_index, build_vector_index
from rag.search import (
    search_keyword,
    search_vector,
    search_hybrid,
    rerank,
)
from rag.prompts import build_prompt, rewrite_query


def get_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """Return an OpenAI-compatible client.

    Parameters
    ----------
    base_url : str or None
        Override the provider base URL (default from config).
    api_key : str or None
        Override the API key (default from config).
    """
    if base_url and api_key:
        return OpenAI(base_url=base_url, api_key=api_key)
    provider = config.PROVIDER_CONFIGS[config.LLM_PROVIDER]
    return OpenAI(base_url=provider["base_url"], api_key=provider["api_key"])


def _format_context(results: list[dict]) -> str:
    """Format a list of retrieved chunks into a single context string."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Source {i}] {r['title']} ({r['source']})\n{r['text']}")
    return "\n\n".join(parts)


def ask(
    question: str,
    retriever: str = None,
    k: int = None,
    rerank_results: bool = False,
    prompt_template: str = "default",
    model: str = None,
    rewrite: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Run the full RAG pipeline for a single question.

    Parameters
    ----------
    question : str
        The user's question.
    retriever : str or None
        One of ``"keyword"``, ``"vector"``, ``"hybrid"`` (default from config).
    k : int or None
        Number of chunks to retrieve (default from config).
    rerank_results : bool
        Whether to apply cross-encoder re-ranking.
    prompt_template : str
        Which system prompt template to use (default ``"default"``).
    model : str or None
        LLM model name (default from config).
    rewrite : bool
        Whether to rewrite the query before retrieval.
    api_key : str or None
        Override the API key (default from config/.env).
    base_url : str or None
        Override the provider base URL (default from config/.env).

    Returns
    -------
    dict
        ``{"answer": str, "sources": list[dict], "latency_ms": int}``.
    """
    t0 = time.time()

    if retriever is None:
        retriever = config.RETRIEVER_TYPE
    if k is None:
        k = config.TOP_K
    if model is None:
        model = config.LLM_MODEL

    # --- Load indexes (lazy, cached) ---
    if not hasattr(ask, "_docs"):
        ask._docs = load_documents()
        ask._kw_index = build_minsearch_index(ask._docs)
        ask._embeddings, ask._vec_model = build_vector_index(ask._docs)

    # --- Query rewriting ---
    final_query = question
    if rewrite:
        rewrite_prompt = rewrite_query(question)
        client = get_client(base_url=base_url, api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.0,
            )
            final_query = resp.choices[0].message.content.strip()
        except Exception:
            final_query = question

    # --- Retrieval ---
    if retriever == "keyword":
        results = search_keyword(ask._kw_index, final_query, k=k)
    elif retriever == "vector":
        results = search_vector(
            ask._embeddings, ask._vec_model, ask._docs, final_query, k=k
        )
    else:
        results = search_hybrid(
            ask._kw_index, ask._embeddings, ask._vec_model, ask._docs, final_query, k=k
        )

    if rerank_results and results:
        results = rerank(final_query, results)

    # --- Generation ---
    context = _format_context(results)
    messages = build_prompt(question, context, template=prompt_template)

    client = get_client(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
    except Exception as e:
        answer = f"Sorry, I encountered an error: {e}"
        prompt_tokens = completion_tokens = total_tokens = 0
        cost = 0.0

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "answer": answer,
        "sources": [
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "url": r["url"],
                "score": round(r.get("score", 0.0), 4),
            }
            for r in results
        ],
        "latency_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }
