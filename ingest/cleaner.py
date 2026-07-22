"""Document cleaner for scraped property articles.

Strips HTML boilerplate (scripts, styles, nav, ads), normalises
whitespace, and removes common promotional phrases. Produces clean
text-only documents ready for chunking.

Exports
-------
clean_all : callable
    Accepts a list of scraper-output dicts and returns cleaned dicts.
"""

import json
import re

from bs4 import BeautifulSoup

import config


def clean_html(raw_html: str) -> str | None:
    """Strip non-content HTML tags and return plain text.

    Removes ``<script>``, ``<style>``, ``<nav>``, ``<footer>``,
    ``<aside>``, and ``<noscript>`` elements before extracting text.

    Parameters
    ----------
    raw_html : str
        Raw HTML content.

    Returns
    -------
    str
        Plain text extracted from the cleaned HTML.
    """
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "li"]):
        tag.insert_before("\n")
    text = soup.get_text(separator=" ")
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and blank lines."""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate_phrases(text: str) -> str:
    """Remove common promotional phrases (share prompts, newsletter CTAs, etc.)."""

    patterns = [
        r"Related\s+(?:posts|articles|stories|guides)[:\s]*",
        r"Share\s+(?:this|article|post)\s",
        r"Subscribe\s+to\s+(?:our\s+)?newsletter",
        r"Follow\s+us\s+on\s+(?:Facebook|Twitter|LinkedIn|Instagram)",
        r"Click\s+here\s+to\s+subscribe",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def clean_document(doc: dict) -> dict | None:
    """Clean a single scraper-output document.

    Applies HTML stripping, whitespace normalisation, and boilerplate
    removal in sequence.

    Parameters
    ----------
    doc : dict
        Document with a ``raw_html`` key (from the scraper).

    Returns
    -------
    dict or None
        Cleaned document with ``text`` key, or None if no HTML content.
    """
    raw_html = doc.get("raw_html", "")
    if not raw_html:
        return None

    cleaned_text = clean_html(raw_html)
    cleaned_text = normalize_whitespace(cleaned_text)
    cleaned_text = remove_boilerplate_phrases(cleaned_text)

    return {
        "url": doc["url"],
        "source": doc["source"],
        "category": doc["category"],
        "title": doc["title"],
        "text": cleaned_text,
    }


def clean_all(docs: list[dict]) -> list[dict]:
    """Clean every document in the list.

    Parameters
    ----------
    docs : list[dict]
        Raw documents from the scraper.

    Returns
    -------
    list[dict]
        Cleaned documents (non-content keys preserved).
    """
    results = []
    for doc in docs:
        cleaned = clean_document(doc)
        if cleaned:
            results.append(cleaned)
    return results


if __name__ == "__main__":
    import glob

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(glob.glob(str(config.RAW_DIR / "*.json")))
    if not raw_files:
        print("No raw files found.")
        exit(0)

    all_docs = []
    for f in raw_files:
        with open(f) as fh:
            all_docs.append(json.load(fh))

    print(f"Loaded {len(all_docs)} raw documents")
    cleaned = clean_all(all_docs)
    print(f"Cleaned {len(cleaned)} documents")

    total_before = sum(len(d.get("text", "")) for d in all_docs)
    total_after = sum(len(d["text"]) for d in cleaned)
    print(f"Total chars: {total_before} → {total_after} ({(1 - total_after/total_before)*100:.1f}% reduction)")

    out_path = config.PROCESSED_DIR / "cleaned_documents.json"
    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Saved to {out_path}")
