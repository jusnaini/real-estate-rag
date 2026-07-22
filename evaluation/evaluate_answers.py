"""Evaluate generated answers using LLM-as-a-Judge.

Loads ground-truth pairs, runs the RAG pipeline for each question, then
scores the answer on faithfulness, relevance, completeness, and citation
correctness using the judge prompt.
"""

import json
import time
import warnings

import numpy as np
from openai import OpenAI

import config
from rag.rag_pipeline import ask

warnings.filterwarnings("ignore")


def load_judge_prompt() -> str:
    judge_path = config.PROJECT_ROOT / "evaluation" / "judge_prompt.txt"
    with open(judge_path) as f:
        return f.read()


def get_llm_client():
    p = config.PROVIDER_CONFIGS[config.LLM_PROVIDER]
    return OpenAI(base_url=p["base_url"], api_key=p["api_key"], timeout=30)


def judge_answer(question: str, context: str, answer: str) -> dict:
    judge_template = load_judge_prompt()
    prompt = judge_template.format(
        question=question,
        context=context[:2000],
        answer=answer,
    )
    client = get_llm_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            scores = json.loads(resp.choices[0].message.content)
            return {
                "faithfulness": int(scores.get("faithfulness", 0)),
                "relevance": int(scores.get("relevance", 0)),
                "completeness": int(scores.get("completeness", 0)),
                "citation": int(scores.get("citation", 0)),
            }
        except Exception as e:
            print(f"    Judge attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    return {"faithfulness": 0, "relevance": 0, "completeness": 0, "citation": 0}


def build_context_from_sources(sources: list[dict]) -> str:
    parts = []
    for s in sources:
        parts.append(f"[{s['title']}]({s['source']})\n{s.get('text', '')}")
    return "\n\n".join(parts)


def main():
    with open(config.GROUND_TRUTH_PATH) as f:
        gt = json.load(f)["pairs"]
    print(f"Loaded {len(gt)} ground-truth pairs\n", flush=True)

    # Configurations to evaluate
    configs = [
        {"name": "hybrid_k3_default", "retriever": "hybrid", "k": 3, "prompt": "default"},
        {"name": "hybrid_k5_default", "retriever": "hybrid", "k": 5, "prompt": "default"},
        {"name": "hybrid_k3_concise", "retriever": "hybrid", "k": 3, "prompt": "concise"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {cfg['name']}", flush=True)
        print(f"{'='*60}", flush=True)

        scores = {"faithfulness": [], "relevance": [], "completeness": [], "citation": []}

        for i, pair in enumerate(gt[:5], 1):  # Evaluate first 5 pairs per config
            print(f"  [{i}/10] {pair['question'][:60]}...", flush=True)

            result = ask(
                question=pair["question"],
                retriever=cfg["retriever"],
                k=cfg["k"],
                prompt_template=cfg["prompt"],
            )

            context = build_context_from_sources(result["sources"])
            scores_dict = judge_answer(pair["question"], context, result["answer"])

            for key in scores:
                scores[key].append(scores_dict[key])

            time.sleep(0.5)

        avg = {key: round(np.mean(vals), 2) for key, vals in scores.items()}
        avg["overall"] = round(np.mean(list(avg.values())), 2)
        print(f"\n  Scores for {cfg['name']}:")
        for key, val in avg.items():
            print(f"    {key}: {val}")

        all_results.append({"config": cfg["name"], **avg})

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['config']:<20s} overall={r['overall']} "
              f"faithfulness={r['faithfulness']} relevance={r['relevance']} "
              f"completeness={r['completeness']} citation={r['citation']}")

    out_path = config.PROCESSED_DIR / "answer_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
