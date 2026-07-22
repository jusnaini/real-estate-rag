"""Prompt templates for the RAG pipeline.

Exports
-------
build_prompt(question, context, template) -> list[dict]
rewrite_query(query) -> str
SYSTEM_PROMPTS : dict[str, str]
"""

SYSTEM_PROMPTS = {
    "default": (
        "You are a Malaysian property buying assistant. "
        "Answer the user's question using **only** the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Always cite the source article title at the end of your answer. "
        "Do not give personalised legal or financial advice."
    ),
    "concise": (
        "You are a Malaysian property buying assistant. "
        "Answer concisely using only the provided context. "
        "Cite the source. Say if information is insufficient."
    ),
}

USER_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}"""

REWRITE_PROMPT = (
    "Rewrite the following question to make it clearer for a search engine. "
    "Expand abbreviations (e.g. MRTA → Mortgage Reducing Term Assurance, "
    "DSR → Debt Service Ratio, SPA → Sale and Purchase Agreement). "
    "Output only the rewritten question, nothing else.\n\n"
    "Original: {query}\nRewritten:"
)


def build_prompt(question: str, context: str, template: str = "default") -> list[dict]:
    """Build a chat-completion message list for the LLM.

    Parameters
    ----------
    question : str
        The user's original question.
    context : str
        Concatenated retrieved document chunks.
    template : str
        Which system prompt to use (key in ``SYSTEM_PROMPTS``).

    Returns
    -------
    list[dict]
        Messages list with ``role`` and ``content`` keys.
    """
    system = SYSTEM_PROMPTS.get(template, SYSTEM_PROMPTS["default"])
    user = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rewrite_query(query: str) -> str:
    """Return the rewrite prompt string for a given query.

    The actual LLM call is made by the pipeline — this method
    just returns the formatted prompt template.

    Parameters
    ----------
    query : str
        Original user query.

    Returns
    -------
    str
        The formatted rewrite prompt.
    """
    return REWRITE_PROMPT.format(query=query)
