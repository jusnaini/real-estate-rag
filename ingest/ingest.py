"""End-to-end ingestion orchestrator.

Chains scraping → cleaning → chunking into a single ``ingest()`` call
that populates ``data/processed/documents.json`` with the final set of
retrieval-ready chunks.

Exports
-------
ingest : callable
    Runs the full ingestion pipeline and returns the chunk list.
"""

import json
import time

import config
from ingest.cleaner import clean_all
from ingest.chunker import chunk_all
from ingest.scraper import scrape_all


def ingest():
    """Run the full ingestion pipeline: scrape → clean → chunk.

    Returns
    -------
    list[dict]
        The final set of chunked documents saved to
        ``data/processed/documents.json``.
    """
    t0 = time.time()

    print("=" * 60)
    print("Phase 1: Scraping")
    print("=" * 60)
    raw_docs = scrape_all()

    print("\n" + "=" * 60)
    print("Phase 2: Cleaning")
    print("=" * 60)
    cleaned_docs = clean_all(raw_docs)
    total_before = sum(len(d.get("text", "")) for d in raw_docs)
    total_after = sum(len(d["text"]) for d in cleaned_docs)
    print(f"Chars: {total_before:,} → {total_after:,} ({(1 - total_after/total_before)*100:.1f}% reduction)")

    cleaned_out = config.PROCESSED_DIR / "cleaned_documents.json"
    with open(cleaned_out, "w") as f:
        json.dump(cleaned_docs, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 3: Chunking")
    print("=" * 60)
    print(f"Chunk size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}")
    chunks = chunk_all(cleaned_docs)

    out_path = config.PROCESSED_DIR / "documents.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Sources scraped:    {len(raw_docs)}")
    print(f"  Documents cleaned:  {len(cleaned_docs)}")
    print(f"  Chunks generated:   {len(chunks)}")
    print(f"  Time taken:         {elapsed:.1f}s")
    print(f"  Output:             {out_path}")

    by_category = {}
    for c in chunks:
        by_category.setdefault(c["category"], 0)
        by_category[c["category"]] += 1
    print(f"\n  Chunks by category:")
    for cat, count in sorted(by_category.items()):
        print(f"    {cat:20s}: {count}")

    return chunks


if __name__ == "__main__":
    ingest()
