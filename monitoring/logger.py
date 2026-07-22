"""SQLite-based query logger for monitoring and analysis.

Logs every query to a local SQLite database for offline analysis,
dashboard visualisation, and user feedback collection. Tracks token
usage and cost for each LLM call.
"""

import json
import sqlite3
from datetime import datetime

import config


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    sources         TEXT,
    latency_ms      INTEGER,
    retriever       TEXT,
    k               INTEGER,
    model           TEXT,
    rewrite         INTEGER,
    prompt_template TEXT,
    feedback        INTEGER,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    total_tokens    INTEGER,
    cost            REAL
)
"""

MIGRATIONS = [
    ("prompt_tokens",      "ALTER TABLE queries ADD COLUMN prompt_tokens INTEGER"),
    ("completion_tokens",  "ALTER TABLE queries ADD COLUMN completion_tokens INTEGER"),
    ("total_tokens",       "ALTER TABLE queries ADD COLUMN total_tokens INTEGER"),
    ("cost",               "ALTER TABLE queries ADD COLUMN cost REAL"),
    ("judge_relevance",    "ALTER TABLE queries ADD COLUMN judge_relevance TEXT"),
    ("judge_explanation",  "ALTER TABLE queries ADD COLUMN judge_explanation TEXT"),
]


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.MONITORING_DB_PATH))
    conn.execute(CREATE_TABLE_SQL)
    _migrate(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(queries)").fetchall()
    }
    for col, sql in MIGRATIONS:
        if col not in existing:
            conn.execute(sql)
    conn.commit()


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if "gpt-5.4-mini" in model:
        return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
    elif "openai/gpt-oss-120b" in model:
        return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
    else:
        return 0.0


def log_query(
    question: str,
    answer: str,
    sources: list[dict] | None = None,
    latency_ms: int | None = None,
    retriever: str | None = None,
    k: int | None = None,
    model: str | None = None,
    rewrite: bool | None = None,
    prompt_template: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost: float | None = None,
) -> int:
    """Insert a query log entry and return its row id."""
    conn = _get_db()
    cur = conn.execute(
        """INSERT INTO queries
           (timestamp, question, answer, sources, latency_ms,
            retriever, k, model, rewrite, prompt_template,
            prompt_tokens, completion_tokens, total_tokens, cost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            question,
            answer,
            json.dumps(sources) if sources else None,
            latency_ms,
            retriever,
            k,
            model,
            int(rewrite) if rewrite is not None else None,
            prompt_template,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def set_feedback(query_id: int, feedback: int) -> None:
    """Record user feedback (-1, 0, or 1) for a given query."""
    if feedback not in (-1, 0, 1):
        raise ValueError("feedback must be -1, 0, or 1")
    conn = _get_db()
    conn.execute("UPDATE queries SET feedback = ? WHERE id = ?", (feedback, query_id))
    conn.commit()
    conn.close()


def set_judge(query_id: int, relevance: str, explanation: str) -> None:
    """Record LLM-as-a-Judge relevance verdict for a given query."""
    conn = _get_db()
    conn.execute(
        "UPDATE queries SET judge_relevance = ?, judge_explanation = ? WHERE id = ?",
        (relevance, explanation, query_id),
    )
    conn.commit()
    conn.close()


def get_all_queries() -> list[sqlite3.Row]:
    """Return all logged queries ordered by timestamp."""
    conn = _get_db()
    rows = conn.execute("SELECT * FROM queries ORDER BY timestamp DESC").fetchall()
    conn.close()
    return rows


def get_stats() -> dict:
    """Return aggregate statistics over all logged queries."""
    conn = _get_db()
    row = conn.execute(
        """SELECT
               COUNT(*)                                    AS total_queries,
               ROUND(AVG(latency_ms), 1)                   AS avg_latency_ms,
               ROUND(SUM(cost), 4)                         AS total_cost,
               ROUND(AVG(total_tokens), 0)                 AS avg_tokens,
               SUM(CASE WHEN feedback = 1   THEN 1 ELSE 0 END)  AS upvotes,
               SUM(CASE WHEN feedback = -1  THEN 1 ELSE 0 END)  AS downvotes,
               COUNT(feedback)                             AS total_feedback
           FROM queries"""
    ).fetchone()
    conn.close()
    return dict(row)
