"""Generate ground-truth Q&A pairs — one batch per article, not per chunk.

Groups the 81 chunks back into 16 articles, then generates 3-4 questions
per article via LLM. Saves to ``data/processed/ground_truth.json``.
"""

import json
import sys
import time

from openai import OpenAI

import config

PROMPT_TEMPLATE = """You are a Malaysian property expert. Given the document below, generate {num} question-answer pairs that a first-time home buyer might ask.

Rules:
- Questions must be answerable from the document alone.
- Answers must be accurate and grounded in the document.
- Questions should cover different aspects (e.g. process, costs, requirements).

Return ONLY a JSON array, no other text:
[{{"question": "...", "answer": "..."}}]

Source: {source}
Title: {title}
Document:
{document_text}
"""


def get_client():
    p = config.PROVIDER_CONFIGS[config.LLM_PROVIDER]
    return OpenAI(base_url=p["base_url"], api_key=p["api_key"], timeout=30)


def group_by_article(chunks: list[dict]) -> list[dict]:
    seen = {}
    for c in chunks:
        key = c["id"]
        # Use the article-level info from first chunk of each article
        art_key = (c["source"], c["title"])
        if art_key not in seen:
            seen[art_key] = {
                "source": c["source"],
                "title": c["title"],
                "category": c["category"],
                "url": c["url"],
                "texts": [],
            }
        seen[art_key]["texts"].append(c["text"])
    return list(seen.values())


def generate_pairs(article: dict, num: int = 4) -> list[dict]:
    full_text = "\n".join(article["texts"])
    # Truncate to stay within free-tier TPM limits (~8000 tokens)
    if len(full_text) > 3000:
        full_text = full_text[:3000] + "..."
    prompt = PROMPT_TEMPLATE.format(
        num=num,
        source=article["source"],
        title=article["title"],
        document_text=full_text,
    )
    client = get_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            pairs = json.loads(text)
            if isinstance(pairs, dict):
                pairs = pairs.get("pairs", pairs.get("questions", [pairs]))
            if isinstance(pairs, dict):
                pairs = [pairs]
            for p in pairs:
                p["source"] = article["source"]
                p["title"] = article["title"]
                p["category"] = article["category"]
                p["url"] = article["url"]
            return pairs
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    return []


def main():
    with open(config.DOCUMENTS_PATH) as f:
        chunks = json.load(f)

    articles = group_by_article(chunks)
    print(f"Loaded {len(chunks)} chunks → {len(articles)} articles", flush=True)

    all_pairs = []
    for i, art in enumerate(articles, 1):
        print(f"[{i}/{len(articles)}] {art['source']:20s} — {art['title'][:50]}", flush=True)
        pairs = generate_pairs(art, num=3)
        all_pairs.extend(pairs)
        print(f"  → {len(pairs)} pairs", flush=True)
        time.sleep(0.5)

    output = {"total_pairs": len(all_pairs), "pairs": all_pairs}
    with open(config.GROUND_TRUTH_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Generated {len(all_pairs)} ground-truth pairs", flush=True)
    by_cat = {}
    for p in all_pairs:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}", flush=True)


if __name__ == "__main__":
    main()
