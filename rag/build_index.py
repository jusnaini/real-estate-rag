"""Index builders for keyword (MinSearch) and vector retrieval.

Exports
-------
build_minsearch_index(documents) -> minsearch.Index
build_vector_index(documents) -> np.ndarray
get_embedder() -> object
get_cross_encoder(model_name) -> object
load_documents() -> list[dict]
"""

import json
import pickle

import minsearch
import numpy as np
from tqdm import tqdm

import config


def get_embedder():
    if config.EMBEDDING_BACKEND == "onnx":
        from rag.embedder import Embedder
        return Embedder()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)


def get_cross_encoder(model_name: str = None):
    if config.EMBEDDING_BACKEND == "onnx":
        from rag.embedder import CrossEncoder
        return CrossEncoder()
    from sentence_transformers import CrossEncoder
    if model_name is None:
        model_name = config.RERANK_MODEL
    return CrossEncoder(model_name)


def load_documents() -> list[dict]:
    """Load the chunked documents from ``data/processed/documents.json``.

    Returns
    -------
    list[dict]
        Document chunks with ``id``, ``text``, ``source``, etc.
    """
    with open(config.DOCUMENTS_PATH) as f:
        return json.load(f)


def build_minsearch_index(documents: list[dict]) -> minsearch.Index:
    """Build a MinSearch keyword index over the ``title`` and ``text`` fields.

    Parameters
    ----------
    documents : list[dict]
        Chunked documents.

    Returns
    -------
    minsearch.Index
        The populated index.
    """
    index = minsearch.Index(
        text_fields=["title", "text"],
        keyword_fields=["source", "category", "id"],
    )
    index.fit(documents)
    return index


def build_vector_index(documents: list[dict], force: bool = False) -> np.ndarray:
    """Compute embeddings for every document chunk using the configured backend.

    Parameters
    ----------
    documents : list[dict]
        Chunked documents.
    force : bool
        If True, recompute even if a cached file exists (default False).

    Returns
    -------
    np.ndarray
        Embeddings with shape ``(N, D)``.
    """
    cache_path = config.PROCESSED_DIR / "embeddings.npy"
    if not force and cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(documents):
            return np.array(cached)

    texts = [d["text"] for d in documents]
    if config.EMBEDDING_BACKEND == "onnx":
        from rag.embedder import Embedder
        embedder = Embedder()
        embeddings = []
        for i in tqdm(range(0, len(texts), 32), desc="Embedding"):
            batch = texts[i:i + 32]
            embeddings.append(embedder.encode_batch(batch))
        embeddings = np.vstack(embeddings)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings)
    np.save(cache_path, embeddings)
    return embeddings


def save_index(index: minsearch.Index, path: str = None):
    """Persist a MinSearch index to disk via pickle.

    Parameters
    ----------
    index : minsearch.Index
        The index to save.
    path : str or None
        File path (default ``data/processed/minsearch_index.pkl``).
    """
    if path is None:
        path = config.PROCESSED_DIR / "minsearch_index.pkl"
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path: str = None) -> minsearch.Index:
    """Load a pickled MinSearch index from disk.

    Parameters
    ----------
    path : str or None
        File path (default ``data/processed/minsearch_index.pkl``).
    """
    if path is None:
        path = config.PROCESSED_DIR / "minsearch_index.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    idx = build_minsearch_index(docs)
    save_index(idx)
    print(f"MinSearch index built and saved ({len(docs)} docs indexed)")

    emb = build_vector_index(docs)
    emb_path = config.PROCESSED_DIR / "embeddings.npy"
    np.save(emb_path, emb)
    print(f"Vector index built — shape {emb.shape}, backend {config.EMBEDDING_BACKEND}")
    print(f"Embeddings saved to {emb_path}")
