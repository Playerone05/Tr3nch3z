"""Score news headlines with FinBERT, a free model trained on financial text.

    python -m signalbot.sentiment            # score every headline not scored yet
    python -m signalbot.sentiment --limit 500

Scores are cached by story, so each headline is only scored once. The first run
downloads the model (~440 MB) and can take 10-20 minutes on a laptop CPU for
~30,000 headlines. Later runs only score new headlines and take seconds.

Each headline gets positive / negative / neutral probabilities and
score = positive - negative, from -1 (very negative) to +1 (very positive).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Iterable

from . import db
from .config import load_settings

MODEL_NAME = "ProsusAI/finbert"

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_sentiment (
    story_key TEXT PRIMARY KEY,
    pos REAL, neg REAL, neu REAL,
    score REAL,                 -- pos - neg, from -1 to +1
    model TEXT,
    scored_at TEXT
);
"""

# A scorer takes a list of headlines and returns one {"positive","negative","neutral"} dict each.
Scorer = Callable[[list[str]], list[dict]]


def ensure_table(conn) -> None:
    conn.executescript(SCHEMA)


def finbert_scorer(batch_size: int = 32) -> Scorer:
    try:
        from transformers import pipeline
    except ImportError:
        raise SystemExit(
            "FinBERT needs two extra packages. Install them with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install -r requirements-ml.txt"
        )
    clf = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True, max_length=64)

    def score(texts: list[str]) -> list[dict]:
        out = clf(texts, batch_size=batch_size)
        return [{d["label"].lower(): float(d["score"]) for d in row} for row in out]

    return score


def unscored(conn, limit: int | None = None) -> list[tuple[str, str]]:
    """(story_key, headline) for stories with no score yet, one headline per story."""
    sql = """SELECT n.story_key, MIN(n.headline)
             FROM news n LEFT JOIN news_sentiment s ON s.story_key = n.story_key
             WHERE s.story_key IS NULL
             GROUP BY n.story_key"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def score_stories(conn, scorer: Scorer, model: str = MODEL_NAME, limit: int | None = None,
                  chunk: int = 256, progress: Callable[[str], None] | None = print) -> int:
    ensure_table(conn)
    todo = unscored(conn, limit)
    if not todo:
        if progress:
            progress("All headlines are already scored.")
        return 0
    started = time.time()
    done = 0
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        results = scorer([h or "" for _, h in part])
        stamp = db.now_utc()
        rows = []
        for (key, _), r in zip(part, results):
            pos, neg, neu = r.get("positive", 0.0), r.get("negative", 0.0), r.get("neutral", 0.0)
            rows.append((key, pos, neg, neu, pos - neg, model, stamp))
        conn.executemany(
            "INSERT OR REPLACE INTO news_sentiment (story_key, pos, neg, neu, score, model, scored_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()  # saved every chunk, so stopping partway keeps progress
        done += len(part)
        if progress:
            rate = done / max(time.time() - started, 1e-6)
            left = (len(todo) - done) / rate if rate else 0
            progress(f"  scored {done:,}/{len(todo):,} headlines  (~{left / 60:.0f} min left)")
    return done


def coverage(conn) -> tuple[int, int]:
    ensure_table(conn)
    total = conn.execute("SELECT COUNT(DISTINCT story_key) FROM news").fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM news_sentiment WHERE story_key IN (SELECT story_key FROM news)"
    ).fetchone()[0]
    return scored, total


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score news headlines with FinBERT.")
    p.add_argument("--limit", type=int, help="only score this many headlines (for a quick test)")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(list(argv) if argv is not None else None)

    s = load_settings()
    conn = db.connect(s.database_path)
    ensure_table(conn)
    left = len(unscored(conn, args.limit))
    if not left:
        print("All headlines are already scored.")
        return 0
    print(f"Loading FinBERT (first run downloads ~440 MB)...", flush=True)
    scorer = finbert_scorer(args.batch_size)
    print(f"Scoring {left:,} headlines...", flush=True)
    score_stories(conn, scorer, limit=args.limit, progress=lambda m: print(m, flush=True))
    scored, total = coverage(conn)
    print(f"\nDone. {scored:,} of {total:,} stories have a sentiment score.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
