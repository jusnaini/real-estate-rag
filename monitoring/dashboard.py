"""Plotly monitoring dashboard served via Streamlit.

Displays 7 visualisations extracted from the SQLite query log:
1. Query volume over time
2. Latency distribution
3. Feedback ratio
4. Top-cited sources
5. Token usage over time
6. Cost over time
7. Cost distribution
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from monitoring.logger import get_all_queries, get_stats

st.set_page_config(page_title="Real Estate RAG — Monitoring", layout="wide")
st.title("Monitoring Dashboard")

rows = get_all_queries()
if not rows:
    st.info("No queries logged yet. Ask a question through the chat app first.")
    st.stop()

records = [dict(r) for r in rows]
stats = get_stats()

# --- KPI cards ---
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Queries", stats["total_queries"])
kpi_cols[1].metric("Avg Latency", f'{stats["avg_latency_ms"] or 0:.0f} ms')
kpi_cols[2].metric("Total Cost", f'${stats["total_cost"] or 0:.4f}')
kpi_cols[3].metric("Avg Tokens", f'{stats["avg_tokens"] or 0:.0f}')
judged = [r for r in records if r.get("judge_relevance")]
relevant_count = sum(1 for r in judged if r["judge_relevance"] == "RELEVANT")
kpi_cols[4].metric(
    "Relevance Rate",
    f"{relevant_count / len(judged) * 100:.0f}%" if judged else "N/A",
)
st.markdown("---")

# --- Helper: parse timestamp ---
for r in records:
    r["_ts"] = datetime.fromisoformat(r["timestamp"])

# ---- 1. Query volume over time ----
st.subheader("Query Volume Over Time")
daily = Counter(r["_ts"].date().isoformat() for r in records)
dates = sorted(daily)
fig1 = go.Figure(data=go.Bar(x=dates, y=[daily[d] for d in dates]))
fig1.update_layout(xaxis_title="Date", yaxis_title="Queries", height=250)
st.plotly_chart(fig1, use_container_width=True)

# ---- 2. Latency distribution ----
st.subheader("Latency Distribution (ms)")
lats = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
if lats:
    fig2 = px.histogram(lats, nbins=20, labels={"value": "Latency (ms)"})
    fig2.update_layout(height=250)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.caption("No latency data yet.")

# ---- 3. Feedback ratio ----
st.subheader("User Feedback")
fb = {"👍 Upvote": 0, "👎 Downvote": 0, "No feedback": 0}
for r in records:
    if r["feedback"] == 1:
        fb["👍 Upvote"] += 1
    elif r["feedback"] == -1:
        fb["👎 Downvote"] += 1
    else:
        fb["No feedback"] += 1
fig3 = go.Figure(data=[go.Pie(labels=list(fb.keys()), values=list(fb.values()), hole=0.4)])
fig3.update_layout(height=250)
st.plotly_chart(fig3, use_container_width=True)

# ---- 4. Top cited sources ----
st.subheader("Most Frequently Cited Sources")
source_counter = Counter()
for r in records:
    if r["sources"]:
        try:
            srcs = json.loads(r["sources"])
            for s in srcs:
                url = s.get("source", "unknown")
                source_counter[url] += 1
        except (json.JSONDecodeError, TypeError):
            pass
if source_counter:
    top = source_counter.most_common(10)
    fig4 = go.Figure(
        data=go.Bar(x=[c for _, c in top], y=[n for n, _ in top], orientation="h")
    )
    fig4.update_layout(
        xaxis_title="Citations",
        yaxis_title="Source",
        height=300,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.caption("No source data logged yet.")

# ---- 5. Token usage over time ----
st.subheader("Token Usage Over Time")
token_rows = [(r["_ts"], r["total_tokens"]) for r in records if r.get("total_tokens")]
if token_rows:
    ts, tokens = zip(*token_rows)
    fig5 = go.Figure(data=go.Scatter(x=list(ts), y=list(tokens), mode="lines+markers"))
    fig5.update_layout(xaxis_title="Time", yaxis_title="Total Tokens", height=250)
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.caption("No token data yet.")

# ---- 6. Cost over time ----
st.subheader("Cost Over Time")
cost_rows = [(r["_ts"], r["cost"]) for r in records if r.get("cost")]
if cost_rows:
    ts, costs = zip(*cost_rows)
    fig6 = go.Figure(
        data=go.Scatter(x=list(ts), y=list(costs), mode="lines+markers", fill="tozeroy")
    )
    fig6.update_layout(xaxis_title="Time", yaxis_title="Cost ($)", height=250)
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.caption("No cost data yet.")

# ---- 7. Cost distribution ----
st.subheader("Cost Distribution (per query)")
all_costs = [r["cost"] for r in records if r.get("cost")]
if all_costs:
    fig7 = px.histogram(all_costs, nbins=15, labels={"value": "Cost ($)"})
    fig7.update_layout(height=250)
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.caption("No cost data yet.")

# ---- 8. Model usage ----
st.subheader("Model Usage")
model_counts = Counter(r["model"] for r in records if r.get("model"))
if model_counts:
    col8a, col8b = st.columns(2)
    with col8a:
        models_sorted = sorted(model_counts)
        fig8 = go.Figure(
            data=go.Bar(
                x=[model_counts[m] for m in models_sorted],
                y=models_sorted,
                orientation="h",
            )
        )
        fig8.update_layout(xaxis_title="Queries", yaxis_title="Model", height=250)
        st.plotly_chart(fig8, use_container_width=True)
    with col8b:
        fig8b = go.Figure(
            data=go.Pie(
                labels=models_sorted,
                values=[model_counts[m] for m in models_sorted],
                hole=0.4,
            )
        )
        fig8b.update_layout(height=250)
        st.plotly_chart(fig8b, use_container_width=True)
else:
    st.caption("No model data yet.")

# ---- 9. Judge relevance distribution ----
st.subheader("LLM-as-a-Judge Relevance")
judged = [r for r in records if r.get("judge_relevance")]
if judged:
    labels = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    counts = {l: sum(1 for r in judged if r["judge_relevance"] == l) for l in labels}
    fig8 = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.4,
                marker=dict(colors=["#2ecc71", "#f39c12", "#e74c3c"]),
            )
        ]
    )
    fig8.update_layout(height=300)
    st.plotly_chart(fig8, use_container_width=True)
    with st.expander("Judge explanations (latest 5)", expanded=False):
        for r in judged[:5]:
            st.markdown(f"**Q:** {r['question'][:100]}...")
            st.markdown(f"**Verdict:** `{r['judge_relevance']}`")
            st.caption(r.get("judge_explanation", ""))
            st.markdown("---")
else:
    st.caption("No judge evaluations yet.")
