"""LLM-as-a-Judge for monitoring answer quality.

Evaluates the relevance of a generated RAG answer to the user's question.
Classification is RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT.
"""

import json
import logging

import config
from rag.rag_pipeline import get_client

logger = logging.getLogger(__name__)

JUDGE_INSTRUCTIONS = """You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer fully addresses the question with accurate information
- PARTLY_RELEVANT: the answer partially addresses the question but may be incomplete or off-topic
- NON_RELEVANT: the answer does not address the question at all

Respond in JSON only: {"relevance": "...", "explanation": "..."}"""


def evaluate_relevance(question: str, answer: str) -> tuple[str, str]:
    """Run LLM-as-a-Judge to evaluate answer relevance.

    Returns
    -------
    tuple[str, str]
        ``(relevance_label, explanation)`` where label is one of
        ``"RELEVANT"``, ``"PARTLY_RELEVANT"``, ``"NON_RELEVANT"``.
        On failure returns ``("UNKNOWN", "")``.
    """
    try:
        judge_cfg = config.PROVIDER_CONFIGS[config.LLM_JUDGE_PROVIDER]
        client = get_client(
            base_url=judge_cfg["base_url"],
            api_key=judge_cfg["api_key"],
        )

        resp = client.chat.completions.create(
            model=config.LLM_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nGenerated Answer: {answer}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = json.loads(resp.choices[0].message.content)
        relevance = result.get("relevance", "UNKNOWN").upper()
        explanation = result.get("explanation", "")
        if relevance not in ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"):
            relevance = "UNKNOWN"
        return relevance, explanation
    except Exception:
        logger.warning("Judge evaluation failed", exc_info=True)
        return "UNKNOWN", ""
