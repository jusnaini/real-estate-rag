"""Streamlit chat UI for the Real Estate RAG Assistant.

Features:
- Conversational chat interface
- Sidebar with config info and controls
- Upvote/downvote feedback on each answer
- Automatic logging to SQLite via monitoring/logger.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import config
from monitoring.logger import log_query, set_feedback, set_judge

st.set_page_config(
    page_title="Real Estate RAG Assistant",
    page_icon="🏠",
    layout="wide",
)

# ---------- Lazy-load RAG pipeline ----------
@st.cache_resource
def get_pipeline():
    from rag.rag_pipeline import ask

    return ask


ask = get_pipeline()

# ---------- Sidebar ----------
st.sidebar.title("🏠 Real Estate RAG Assistant")

# --- Custom API (for public deployment) ---
with st.sidebar.expander("API Settings", expanded=False):
    use_custom = st.checkbox("Use my own API key")
    if use_custom:
        provider_options = list(config.PROVIDER_CONFIGS.keys())
        custom_provider = st.selectbox("Provider", provider_options, index=provider_options.index(config.LLM_PROVIDER))
        custom_api_key = st.text_input("API Key", type="password")
        custom_model = st.text_input("Model", value=config.LLM_MODEL)
        st.session_state.custom_creds = {
            "provider": custom_provider,
            "api_key": custom_api_key,
            "model": custom_model,
            "base_url": config.PROVIDER_CONFIGS[custom_provider]["base_url"],
        }
    else:
        st.session_state.custom_creds = None

# --- Config display ---
creds = st.session_state.get("custom_creds")
display_provider = creds["provider"] if creds else config.LLM_PROVIDER
display_model = creds["model"] if creds else config.LLM_MODEL

st.sidebar.markdown(
    f"""
**Configuration**

| Setting | Value |
|---------|-------|
| Provider | `{display_provider}` |
| Model | `{display_model}` |
| Retriever | `{config.RETRIEVER_TYPE}` |
| Top-K | `{config.TOP_K}` |
| Rewrite | Enabled |
"""
)
st.sidebar.markdown("---")

# --- About & How-to-use ---
with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown(
        """
        This application helps users understand the property
        purchasing process in Malaysia through a conversational interface by
        answering common questions in buying process, legal cost, financing,
        and insurance.

        Answers are grounded in 16 curated Malaysian property guides.
        """
    )

with st.sidebar.expander("📖 How to Use", expanded=False):
    st.markdown(
        """
        1. Click **🔑 API Settings** in the sidebar to use your own API key and model.
        2. Type a question in the chat box below.
        3. The assistant retrieves relevant guides and generates an answer.
        4. Click **Sources** to view the original articles.
        5. Upvote 👍 or downvote 👎 each answer to help us improve our quality.

        **Example questions:**
        - How much money do I need upfront?
        - What documentations needed for the purchase?
        - Are there any government schemes to help me buy my first home?
        - Is there any special incentives for first time buyer?
        - What type of insurance available and which mandatory?
        - What legal fees should I prepare when buying a house?
        - How to calculate how much financing I can obtain?
        - What Islamic financing options available?
        """
    )

st.sidebar.caption("LLM Zoomcamp 2026 · Capstone Project")

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})  *(score: {s['score']})*")
        if msg.get("query_id"):
            col1, _ = st.columns([1, 10])
            with col1:
                if msg.get("feedback") is None:
                    if st.button("👍", key=f"up_{msg['query_id']}"):
                        set_feedback(msg["query_id"], 1)
                        msg["feedback"] = 1
                        st.rerun()
                    if st.button("👎", key=f"down_{msg['query_id']}"):
                        set_feedback(msg["query_id"], -1)
                        msg["feedback"] = -1
                        st.rerun()
                else:
                    icon = "👍" if msg["feedback"] == 1 else "👎"
                    st.caption(f"{icon} Feedback recorded")

# ---------- Chat input ----------
if prompt := st.chat_input("Ask about buying property in Malaysia..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            creds = st.session_state.get("custom_creds")
            result = ask(
                question=prompt,
                retriever=config.RETRIEVER_TYPE,
                k=config.TOP_K,
                rewrite=True,
                prompt_template="default",
                model=creds["model"] if creds else None,
                api_key=creds["api_key"] if creds else None,
                base_url=creds["base_url"] if creds else None,
            )

        answer = result["answer"]
        sources = result["sources"]
        latency = result["latency_ms"]
        prompt_tokens = result["prompt_tokens"]
        completion_tokens = result["completion_tokens"]
        total_tokens = result["total_tokens"]
        cost = result["cost"]
        used_model = creds["model"] if creds else config.LLM_MODEL

        # Log to SQLite
        query_id = log_query(
            question=prompt,
            answer=answer,
            sources=sources,
            latency_ms=latency,
            retriever=config.RETRIEVER_TYPE,
            k=config.TOP_K,
            model=used_model,
            rewrite=True,
            prompt_template="default",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.markdown(f"- [{s['title']}]({s['url']})  *(score: {s['score']})*")
        st.caption(f"_{latency}ms · {total_tokens} tokens · ${cost:.4f}_")

        col1, _ = st.columns([1, 10])
        with col1:
            if st.button("👍", key=f"up_{query_id}"):
                set_feedback(query_id, 1)
                st.rerun()
            if st.button("👎", key=f"down_{query_id}"):
                set_feedback(query_id, -1)
                st.rerun()

        # LLM-as-a-Judge (fire-and-forget, non-blocking)
        try:
            from monitoring.judge import evaluate_relevance

            relevance, explanation = evaluate_relevance(prompt, answer)
            if relevance != "UNKNOWN":
                set_judge(query_id, relevance, explanation)
        except Exception:
            pass

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "query_id": query_id,
            "feedback": None,
        }
    )
