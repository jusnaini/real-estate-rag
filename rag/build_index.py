"""Index builders for keyword (MinSearch) and vector (sentence-transformers) retrieval.

Exports
-------
build_minsearch_index(documents) -> minsearch.Index
build_vector_index(documents) -> tuple[np.ndarray, SentenceTransformer]
load_documents() -> list[dict]
"""

import json
import pickle

import minsearch
import numpy as np
from sentence_transformers import SentenceTransformer

import config


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


def build_vector_index(documents: list[dict]) -> tuple[np.ndarray, SentenceTransformer]:
    """Compute sentence-transformer embeddings for every document chunk.

    Parameters
    ----------
    documents : list[dict]
        Chunked documents.

    Returns
    -------
    tuple[np.ndarray, SentenceTransformer]
        ``(embeddings, model)`` where embeddings has shape ``(N, D)``.
    """
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings), model


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

    emb, model = build_vector_index(docs)
    emb_path = config.PROCESSED_DIR / "embeddings.npy"
    np.save(emb_path, emb)
    print(f"Vector index built — shape {emb.shape}, model {config.EMBEDDING_MODEL}")
    print(f"Embeddings saved to {emb_path}")
