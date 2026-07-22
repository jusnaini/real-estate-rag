"""Plotly monitoring dashboard served via Streamlit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from monitoring.logger import get_all_queries, get_stats

st.set_page_config(
    page_title="Real Estate RAG — Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {
    "primary": "#4F8BF9",
    "secondary": "#6C5CE7",
    "success": "#00B894",
    "warning": "#FDCB6E",
    "danger": "#E17055",
    "gray": "#636E72",
}

# --- Load data ---
rows = get_all_queries()
if not rows:
    with st.columns(3)[1]:
        st.info("No queries logged yet. Ask a question through the chat app first.")
    st.stop()

records = [dict(r) for r in rows]
stats = get_stats()

for r in records:
    r["_ts"] = datetime.fromisoformat(r["timestamp"])

df = pd.DataFrame(records)
min_date = df["_ts"].min().date()
max_date = df["_ts"].max().date()


def kpi_card(col, label, value, color=COLORS["primary"]):
    col.markdown(
        f"""
        <div style="background:white;border-radius:10px;padding:16px;border-left:4px solid {color};box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:8px">
            <div style="font-size:12px;color:#636E72;margin-bottom:4px">{label}</div>
            <div style="font-size:24px;font-weight:600;color:#2D3436">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


judged = [r for r in records if r.get("judge_relevance")]
relevant_count = sum(1 for r in judged if r["judge_relevance"] == "RELEVANT")
relevance_rate = f"{relevant_count / len(judged) * 100:.0f}%" if judged else "—"

kpi_data = [
    ("Total Queries", str(stats["total_queries"]), COLORS["primary"]),
    ("Avg Latency", f'{stats["avg_latency_ms"] or 0:.0f} ms', COLORS["secondary"]),
    ("Total Cost", f'${stats["total_cost"] or 0:.4f}', COLORS["success"]),
    ("Avg Tokens", f'{stats["avg_tokens"] or 0:.0f}', COLORS["warning"]),
    ("Relevance Rate", relevance_rate, COLORS["danger"]),
]

kpi_cols = st.columns(5)
for col, (label, value, color) in zip(kpi_cols, kpi_data):
    kpi_card(col, label, value, color=color)

with st.container():
    col_f1, col_f2, col_f3, _ = st.columns([2, 2, 2, 6])
    with col_f1:
        date_from = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
    with col_f2:
        date_to = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
    with col_f3:
        refresh = st.checkbox("Auto-refresh (30s)")

if refresh:
    st.rerun(30)

filtered = df[(df["_ts"].dt.date >= date_from) & (df["_ts"].dt.date <= date_to)]

days_diff = (date_to - date_from).days
rule = "h" if days_diff <= 2 else "D"

st.markdown("---")

tabs = st.tabs(["📈 Overview", "⚡ Performance", "🤖 Models", "⚖️ Judge"])


def _fig(fig, height=220):
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        hovermode="x unified",
        font=dict(family="system-ui, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#E0E0E0"),
        yaxis=dict(showgrid=False, zerolinecolor="#E0E0E0"),
        height=height,
    )
    return fig


# ---------- TAB 1: Overview ----------
with tabs[0]:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Query Volume**")
        volume = filtered.set_index("_ts").resample(rule).size().reset_index(name="count")
        fig = go.Figure(
            data=go.Bar(
                x=volume["_ts"].astype(str),
                y=volume["count"],
                marker_color=COLORS["primary"],
                marker_line_width=0,
            )
        )
        fig.update_yaxes(nticks=5)
        st.plotly_chart(_fig(fig), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("**Cost Over Time**")
        cost_data = filtered[filtered["cost"].notna()].copy()
        if not cost_data.empty:
            cost_resampled = cost_data.set_index("_ts").resample(rule)["cost"].sum().reset_index()
            fig = go.Figure(
                data=go.Scatter(
                    x=cost_resampled["_ts"].astype(str),
                    y=cost_resampled["cost"],
                    mode="lines+markers",
                    line=dict(color=COLORS["success"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,184,148,0.1)",
                )
            )
            st.plotly_chart(_fig(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No cost data yet.")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Token Usage**")
        token_data = filtered[filtered["total_tokens"].notna()]
        if not token_data.empty:
            tdf = token_data.set_index("_ts").resample(rule)["total_tokens"].sum().reset_index()
            fig = go.Figure(
                data=go.Scatter(
                    x=tdf["_ts"].astype(str),
                    y=tdf["total_tokens"],
                    mode="lines+markers",
                    line=dict(color=COLORS["secondary"], width=2),
                )
            )
            st.plotly_chart(_fig(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No token data yet.")

    with c4:
        st.markdown("**User Feedback**")
        fb = {"👍 Upvote": 0, "👎 Downvote": 0, "No feedback": 0}
        for _, r in filtered.iterrows():
            if r["feedback"] == 1:
                fb["👍 Upvote"] += 1
            elif r["feedback"] == -1:
                fb["👎 Downvote"] += 1
            else:
                fb["No feedback"] += 1
        fig = go.Figure(
            data=[go.Pie(
                labels=list(fb.keys()),
                values=list(fb.values()),
                hole=0.5,
                marker=dict(colors=[COLORS["success"], COLORS["danger"], COLORS["gray"]]),
                textinfo="percent",
                textposition="outside",
                hovertemplate="%{label}: %{value} (%{percent})",
            )]
        )
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=40, l=10, r=10))
        st.plotly_chart(_fig(fig, height=280), use_container_width=True, config={"displayModeBar": False})

# ---------- TAB 2: Performance ----------
with tabs[1]:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Latency Distribution**")
        lat_df = filtered[filtered["latency_ms"].notna()].copy()
        if not lat_df.empty:
            lat_df["latency_s"] = lat_df["latency_ms"] / 1000
            fig = px.histogram(
                lat_df,
                x="latency_s",
                nbins=25,
                labels={"latency_s": "Latency (s)"},
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig.update_traces(hovertemplate="Latency Range: %{x}s<br>Count: %{y}")
            fig.update_layout(
                bargap=0.08,
                xaxis=dict(title="Latency (seconds)", gridcolor="#F0F0F0"),
                yaxis=dict(title="Count", showgrid=True, gridcolor="#F0F0F0"),
            )
            st.plotly_chart(_fig(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No latency data yet.")

    with c2:
        st.markdown("**Latency Trend**")
        lat_trend = filtered[filtered["latency_ms"].notna()].copy()
        if not lat_trend.empty:
            lat_trend = lat_trend.set_index("_ts").resample(rule)["latency_ms"].mean().reset_index()
            lat_trend["latency_s"] = lat_trend["latency_ms"] / 1000
            fig = go.Figure(
                data=go.Scatter(
                    x=lat_trend["_ts"].astype(str),
                    y=lat_trend["latency_s"],
                    mode="lines+markers",
                    line=dict(color=COLORS["secondary"], width=2.5, shape="spline"),
                    fill="tozeroy",
                    fillcolor="rgba(108, 92, 231, 0.08)",
                    hovertemplate="%{x}<br>Avg Latency: %{y:.2f} s",
                )
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis=dict(title="Avg Latency (s)", showgrid=True, gridcolor="#F0F0F0"),
            )
            st.plotly_chart(_fig(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No latency data yet.")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Cost Distribution**")
        costs = filtered[filtered["cost"].notna()]["cost"]
        if not costs.empty:
            fig = px.histogram(
                costs,
                nbins=15,
                labels={"value": "Cost ($)"},
                color_discrete_sequence=[COLORS["success"]],
            )
            fig.update_traces(hovertemplate="Cost Range: $%{x}<br>Count: %{y}")
            fig.update_layout(
                bargap=0.08,
                xaxis=dict(title="Cost ($)", tickprefix="$", gridcolor="#F0F0F0"),
                yaxis=dict(title="Count", showgrid=True, gridcolor="#F0F0F0"),
            )
            st.plotly_chart(_fig(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No cost data yet.")

    with c4:
        st.markdown("**Top Cited Sources**")
        source_counter = Counter()
        for _, r in filtered.iterrows():
            if r["sources"]:
                try:
                    for s in json.loads(r["sources"]):
                        source_counter[s.get("source", "unknown")] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        if source_counter:
            top = dict(source_counter.most_common(10))
            chart_height = max(220, len(top) * 28)
            fig = go.Figure(
                data=go.Bar(
                    x=list(top.values()),
                    y=list(top.keys()),
                    orientation="h",
                    text=list(top.values()),
                    textposition="outside",
                    marker_color=COLORS["primary"],
                    marker=dict(cornerradius=4),
                )
            )
            fig.update_layout(
                xaxis=dict(title="Citations", showgrid=True, gridcolor="#F0F0F0"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(_fig(fig, height=chart_height), use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No source data yet.")

# ---------- TAB 3: Models ----------
with tabs[2]:
    if not filtered.empty and filtered["model"].notna().any():
        model_stats = (
            filtered.groupby("model")
            .agg(
                query_count=("model", "count"),
                total_cost=("cost", "sum"),
                avg_latency=("latency_ms", "mean"),
                avg_tokens=("total_tokens", "mean"),
            )
            .reset_index()
            .sort_values(by="query_count", ascending=True)
        )

        dynamic_height = max(240, len(model_stats) * 32)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Query Count by Model**")
            fig = go.Figure(
                data=go.Bar(
                    x=model_stats["query_count"],
                    y=model_stats["model"],
                    orientation="h",
                    text=model_stats["query_count"],
                    textposition="outside",
                    marker_color=COLORS["primary"],
                    marker=dict(cornerradius=4),
                )
            )
            fig.update_layout(
                xaxis=dict(title="Queries", showgrid=True, gridcolor="#F0F0F0"),
                yaxis=dict(title=None),
            )
            st.plotly_chart(_fig(fig, height=dynamic_height), use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown("**Total Cost by Model ($)**")
            cost_stats = model_stats.sort_values(by="total_cost", ascending=True)
            fig = go.Figure(
                data=go.Bar(
                    x=cost_stats["total_cost"],
                    y=cost_stats["model"],
                    orientation="h",
                    text=[f"${c:.4f}" for c in cost_stats["total_cost"]],
                    textposition="outside",
                    marker_color=COLORS["danger"],
                    marker=dict(cornerradius=4),
                )
            )
            fig.update_layout(
                xaxis=dict(title="Total Cost ($)", tickprefix="$", showgrid=True, gridcolor="#F0F0F0"),
                yaxis=dict(title=None),
            )
            st.plotly_chart(_fig(fig, height=dynamic_height), use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.markdown("**Model Efficiency Summary**")

        summary_df = model_stats.sort_values(by="query_count", ascending=False).copy()
        summary_df["avg_cost_per_query"] = summary_df["total_cost"] / summary_df["query_count"]
        summary_df["avg_latency_s"] = summary_df["avg_latency"] / 1000

        st.dataframe(
            summary_df[[
                "model", "query_count", "total_cost", "avg_cost_per_query", "avg_latency_s", "avg_tokens"
            ]].rename(columns={
                "model": "Model Name",
                "query_count": "Queries",
                "total_cost": "Total Cost ($)",
                "avg_cost_per_query": "Avg Cost / Query ($)",
                "avg_latency_s": "Avg Latency (s)",
                "avg_tokens": "Avg Tokens",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total Cost ($)": st.column_config.NumberColumn(format="$%.4f"),
                "Avg Cost / Query ($)": st.column_config.NumberColumn(format="$%.6f"),
                "Avg Latency (s)": st.column_config.NumberColumn(format="%.2f s"),
                "Avg Tokens": st.column_config.NumberColumn(format="%.0f"),
            }
        )
    else:
        st.caption("No model data yet.")

# ---------- TAB 4: Judge ----------
with tabs[3]:
    judged = filtered[filtered["judge_relevance"].notna()]
    if not judged.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Relevance Distribution**")
            labels = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
            counts = {l: int((judged["judge_relevance"] == l).sum()) for l in labels}
            fig = go.Figure(
                data=[go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.5,
                    marker=dict(colors=[COLORS["success"], COLORS["warning"], COLORS["danger"]]),
                    textinfo="label+percent",
                )]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(_fig(fig, height=280), use_container_width=True)
        with c2:
            st.markdown("**Relevance Trend**")
            rel_trend = judged.set_index("_ts").resample("D")["judge_relevance"].apply(
                lambda x: (x == "RELEVANT").mean() * 100
            ).reset_index(name="rate")
            fig = go.Figure(
                data=go.Scatter(
                    x=rel_trend["_ts"].astype(str),
                    y=rel_trend["rate"],
                    mode="lines+markers",
                    line=dict(color=COLORS["success"], width=2),
                )
            )
            fig.update_layout(xaxis_title=None, yaxis_title="Relevant (%)", yaxis_range=[0, 100])
            st.plotly_chart(_fig(fig), use_container_width=True)

        st.markdown("---")
        st.markdown("**Latest Judge Verdicts**")
        for _, r in judged.sort_values("_ts", ascending=False).head(5).iterrows():
            color_map = {"RELEVANT": COLORS["success"], "PARTLY_RELEVANT": COLORS["warning"], "NON_RELEVANT": COLORS["danger"]}
            color = color_map.get(r["judge_relevance"], COLORS["gray"])
            st.markdown(
                f"""
                <div style="background:white;border-radius:8px;padding:12px;margin-bottom:8px;border-left:3px solid {color};box-shadow:0 1px 2px rgba(0,0,0,0.05)">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <strong>{r['question'][:120]}{'…' if len(r['question']) > 120 else ''}</strong>
                        <span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:11px;white-space:nowrap">{r['judge_relevance']}</span>
                    </div>
                    <div style="font-size:12px;color:#636E72;margin-top:4px">{r.get('judge_explanation', '')[:200]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No judge evaluations yet. Ask questions through the chat app first.")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {len(records)} total queries in DB")
