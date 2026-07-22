"""Web scraper module for Malaysian property educational resources.

Fetches article content from a predefined list of source URLs, extracts
the main article body using per-domain HTML parsers, and saves the raw
output (HTML + extracted text) to ``data/raw/`` as JSON files.

Exports
-------
SOURCE_CONFIGS : list[dict]
    Metadata for every source URL (url, source name, category, title).
scrape_all : callable
    Orchestrates fetching and extraction across all sources.
"""

import json
import time

import requests
from bs4 import BeautifulSoup

import config

SOURCE_CONFIGS = [
    {
        "url": "https://www.iproperty.com.my/guides/documents-and-paperwork-buying-a-house-in-malaysia-71908",
        "source": "iproperty",
        "category": "buying_process",
        "title": "The Process of Buying a House in Malaysia 2026",
    },
    {
        "url": "https://iqiglobal.com/blog/step-by-step-guide-buy-house-malaysia/",
        "source": "iqi_global",
        "category": "buying_process",
        "title": "Step-by-Step Guide to Buying a House in Malaysia",
    },
    {
        "url": "https://iqiglobal.com/blog/complete-guide-to-purchasing-property-in-malaysia/",
        "source": "iqi_global",
        "category": "buying_process",
        "title": "Complete Guide to Purchasing Property in Malaysia",
    },
    {
        "url": "https://www.stashaway.my/r/complete-guide-first-time-home-buyer-buying-house-in-malaysia",
        "source": "stashaway",
        "category": "buying_process",
        "title": "Complete Guide For First Time Home Buyer",
    },
    {
        "url": "https://www.iproperty.com.my/guides/stamp-duty-spa-legal-fees-owning-a-house-malaysia-24760",
        "source": "iproperty",
        "category": "legal_cost",
        "title": "Property Stamp Duty in Malaysia: How to Calculate",
    },
    {
        "url": "https://propcashflow.my/tools/legal-fee-calculator/",
        "source": "propcashflow",
        "category": "legal_cost",
        "title": "Legal Fees Calculator Malaysia",
    },
    {
        "url": "https://propcashflow.my/blog/stamp-duty-malaysia-guide-2026/",
        "source": "propcashflow",
        "category": "legal_cost",
        "title": "Stamp Duty (MOT) Malaysia 2026",
    },
    {
        "url": "https://newprojek.com/calculators/stamp-duty-calculator",
        "source": "newprojek",
        "category": "legal_cost",
        "title": "Stamp Duty Calculator Malaysia 2026",
    },
    {
        "url": "https://rummah.my/tools/stamp-duty-calculator",
        "source": "rummah",
        "category": "legal_cost",
        "title": "Malaysian Stamp Duty Calculator",
    },
    {
        "url": "https://suppiahlaw.com/legal-fee-stamp-duty-calculator/",
        "source": "suppiah_law",
        "category": "legal_cost",
        "title": "Legal Fee and Stamp Duty Calculator",
    },
    {
        "url": "https://ihome.my/guides/home-loan-dsr-malaysia/",
        "source": "ihome",
        "category": "financing",
        "title": "How Much Home Loan Can You Afford? DSR Explained",
    },
    {
        "url": "https://propcashflow.my/blog/home-loan-eligibility-dsr-malaysia/",
        "source": "propcashflow",
        "category": "financing",
        "title": "DSR Calculation Malaysia: Home Loan Eligibility",
    },
    {
        "url": "https://propcashflow.my/blog/loan-margin-financing-property-malaysia/",
        "source": "propcashflow",
        "category": "financing",
        "title": "How Much Loan Can You Get? LTV Rules 2026",
    },
    {
        "url": "https://ringgitplus.com/en/home-loan/",
        "source": "ringgitplus",
        "category": "financing",
        "title": "Best Housing Loans in Malaysia 2026",
    },
    {
        "url": "https://ringgitplus.com/en/calculators/debt-service-ratio-dsr-calculator/",
        "source": "ringgitplus",
        "category": "financing",
        "title": "DSR Calculator",
    },
    {
        "url": "https://calculatormalaysia.com/loan/home-loan-eligibility-calculator-malaysia/",
        "source": "calculatormalaysia",
        "category": "financing",
        "title": "Home Loan Eligibility Calculator 2026",
    },
    {
        "url": "https://propcashflow.my/blog/property-insurance-fire-insurance-malaysia/",
        "source": "propcashflow",
        "category": "insurance",
        "title": "Property Insurance Malaysia: Fire, MRTA & Homeowner Coverage Compared",
    },
    {
        "url": "https://www.getfoundation.com.my/blog/bank-loan-fire-insurance-requirements-malaysia",
        "source": "foundation",
        "category": "insurance",
        "title": "Bank Loan Fire Insurance Malaysia: Requirements",
    },
    # PropertyGenie excluded — site blocks automated requests (429)
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RealEstateRAG/1.0; +https://github.com/DataTalksClub/llm-zoomcamp)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def fetch_page(url: str, retries: int = 3) -> str | None:
    """Fetch a URL with exponential-backoff retry on 429/errors.

    Parameters
    ----------
    url : str
        The URL to fetch.
    retries : int
        Max number of attempts (default 3).

    Returns
    -------
    str or None
        Response body as text, or None if all attempts fail.
    """
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 429 and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [RATE-LIMITED] Retrying in {wait}s (attempt {attempt + 2}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < retries - 1:
                print(f"  [RETRY] {e} — retrying ({attempt + 2}/{retries})")
                time.sleep(2)
                continue
            print(f"  [ERROR] Failed to fetch {url}: {e}")
            return None
    return None


def _find_and_clean_article(soup, container_selectors):
    """Find the main content container using a list of CSS/BS selectors
    and strip out non-content tags.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML.
    container_selectors : list
        List of tag names or dicts (e.g. ``["article", {"class": "content"}]``).
    """
    article = None
    for sel in container_selectors:
        if isinstance(sel, str):
            article = soup.find(sel)
        else:
            article = soup.find(**sel)
        if article:
            break
    if not article:
        return None
    for tag in article.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    return article.get_text(strip=True)


def extract_iproperty(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", "main", {"class": "content"}],
    )
    article = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
    if not article:
        return None
    for tag in article.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    return article.get_text(strip=True)


def extract_iqi_global(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", {"class": "entry-content"}],
    )


def extract_stashaway(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", "main"],
    )


def extract_propcashflow(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", {"class": "post-content"}],
    )


def extract_newprojek(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", "main"],
    )


def extract_rummah(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["main", "article"],
    )


def extract_suppiah_law(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", "main"],
    )


def extract_ihome(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", {"class": "content"}],
    )


def extract_ringgitplus(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["main", "article"],
    )


def extract_calculatormalaysia(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", "main"],
    )


def extract_foundation(html: str) -> str | None:
    return _find_and_clean_article(
        BeautifulSoup(html, "lxml"),
        ["article", "main"],
    )



EXTRACTORS = {
    "iproperty": extract_iproperty,
    "iqi_global": extract_iqi_global,
    "stashaway": extract_stashaway,
    "propcashflow": extract_propcashflow,
    "newprojek": extract_newprojek,
    "rummah": extract_rummah,
    "suppiah_law": extract_suppiah_law,
    "ihome": extract_ihome,
    "ringgitplus": extract_ringgitplus,
    "calculatormalaysia": extract_calculatormalaysia,
    "foundation": extract_foundation,
}


def scrape_source(cfg: dict) -> dict | None:
    """Scrape a single source defined by its config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``url``, ``source``, ``category``, and ``title`` keys.

    Returns
    -------
    dict or None
        Scraped document dict, or None on failure.
    """
    print(f"  Fetching: {cfg['url']}")
    html = fetch_page(cfg["url"])
    if html is None:
        return None

    extractor = EXTRACTORS.get(cfg["source"])
    if not extractor:
        print(f"  [WARN] No extractor for source '{cfg['source']}'")
        return None

    text = extractor(html)
    if not text or len(text.strip()) < 50:
        print(f"  [WARN] Extracted text too short (<50 chars) for {cfg['url']}")
        return None

    return {
        "url": cfg["url"],
        "source": cfg["source"],
        "category": cfg["category"],
        "title": cfg["title"],
        "raw_html": html,
        "text": text,
    }


def scrape_all() -> list[dict]:
    """Iterate over every source in SOURCE_CONFIGS and scrape each one.

    Results are saved individually to ``data/raw/`` as JSON files.

    Returns
    -------
    list[dict]
        All successfully scraped documents.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    errors = 0

    for i, cfg in enumerate(SOURCE_CONFIGS, 1):
        print(f"[{i}/{len(SOURCE_CONFIGS)}] {cfg['source']} — {cfg['title']}")
        doc = scrape_source(cfg)
        if doc:
            results.append(doc)
            out_path = config.RAW_DIR / f"{cfg['source']}_{i}.json"
            with open(out_path, "w") as f:
                json.dump(doc, f, indent=2)
        else:
            errors += 1
        time.sleep(1)

    print(f"\nDone. Scraped: {len(results)}, Errors: {errors}")
    return results


if __name__ == "__main__":
    scrape_all()
