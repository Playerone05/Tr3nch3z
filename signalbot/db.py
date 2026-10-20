"""SQLite storage: one local file holds prices, news, earnings, and a refresh log."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker     TEXT NOT NULL,
    date       TEXT NOT NULL,          -- trading day, YYYY-MM-DD
    open       REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
    source     TEXT,
    updated_at TEXT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS news (
    ticker       TEXT NOT NULL,
    story_key    TEXT NOT NULL,        -- hash of the normalized headline (dedupes reposts)
    published_at TEXT NOT NULL,        -- UTC, ISO 8601
    trading_day  TEXT NOT NULL,        -- first session whose close comes after publication
    headline     TEXT,
    summary      TEXT,
    source       TEXT,                 -- outlet, e.g. Benzinga
    url          TEXT,
    provider     TEXT,                 -- alpaca | finnhub
    n_tickers    INTEGER,              -- how many tickers the story is tagged with
    PRIMARY KEY (ticker, story_key)
);
CREATE INDEX IF NOT EXISTS idx_news_day ON news (ticker, trading_day);

CREATE TABLE IF NOT EXISTS earnings (
    ticker           TEXT NOT NULL,
    date             TEXT NOT NULL,
    hour             TEXT,             -- bmo (before open) | amc (after close) | dmh | ''
    eps_estimate     REAL, eps_actual REAL,
    revenue_estimate REAL, revenue_actual REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS refresh_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT,
    finished_at TEXT,
    step        TEXT,
    rows        INTEGER,
    status      TEXT,                  -- ok | skipped | error
    detail      TEXT
);
"""

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
NEWS_COLUMNS = [
    "ticker", "story_key", "published_at", "trading_day", "headline",
    "summary", "source", "url", "provider", "n_tickers",
]
EARNINGS_COLUMNS = [
    "ticker", "date", "hour", "eps_estimate", "eps_actual", "revenue_estimate", "revenue_actual",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _none_if_nan(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    stamp = now_utc()
    rows = [
        tuple(_none_if_nan(r[c]) for c in PRICE_COLUMNS) + (stamp,)
        for r in df[PRICE_COLUMNS].to_dict("records")
    ]
    before = conn.total_changes
    conn.executemany(
        """INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume, source, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (ticker, date) DO UPDATE SET
             open = excluded.open, high = excluded.high, low = excluded.low,
             close = excluded.close, adj_close = excluded.adj_close, volume = excluded.volume,
             source = excluded.source, updated_at = excluded.updated_at""",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def insert_news(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Insert stories. If the same story (same headline) is already stored for a
    ticker, keep whichever copy was published EARLIEST, since that is when the
    market first saw it. Returns the number of rows inserted or moved earlier."""
    data = [tuple(r.get(c) for c in NEWS_COLUMNS) for r in rows]
    if not data:
        return 0
    before = conn.total_changes
    conn.executemany(
        f"""INSERT INTO news ({", ".join(NEWS_COLUMNS)})
            VALUES ({", ".join("?" * len(NEWS_COLUMNS))})
            ON CONFLICT (ticker, story_key) DO UPDATE SET
              published_at = excluded.published_at, trading_day = excluded.trading_day,
              source = excluded.source, url = excluded.url, provider = excluded.provider
            WHERE excluded.published_at < news.published_at""",
        data,
    )
    conn.commit()
    return conn.total_changes - before


def upsert_earnings(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    data = [tuple(r.get(c) for c in EARNINGS_COLUMNS) for r in rows]
    if not data:
        return 0
    before = conn.total_changes
    conn.executemany(
        f"""INSERT INTO earnings ({", ".join(EARNINGS_COLUMNS)})
            VALUES ({", ".join("?" * len(EARNINGS_COLUMNS))})
            ON CONFLICT (ticker, date) DO UPDATE SET
              hour = excluded.hour, eps_estimate = excluded.eps_estimate,
              eps_actual = excluded.eps_actual, revenue_estimate = excluded.revenue_estimate,
              revenue_actual = excluded.revenue_actual""",
        data,
    )
    conn.commit()
    return conn.total_changes - before


def trading_days(conn: sqlite3.Connection, reference: str = "SPY") -> list[str]:
    """Sorted list of known trading days, taken from the reference ticker's prices."""
    days = [r[0] for r in conn.execute(
        "SELECT date FROM prices WHERE ticker = ? ORDER BY date", (reference,)
    )]
    if not days:
        days = [r[0] for r in conn.execute("SELECT DISTINCT date FROM prices ORDER BY date")]
    return days


def last_price_date(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)).fetchone()
    return row[0] if row else None


def last_news_time(conn: sqlite3.Connection, provider: str) -> str | None:
    row = conn.execute("SELECT MAX(published_at) FROM news WHERE provider = ?", (provider,)).fetchone()
    return row[0] if row else None


def log_step(conn: sqlite3.Connection, step: str, started_at: str, rows: int, status: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO refresh_log (started_at, finished_at, step, rows, status, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (started_at, now_utc(), step, rows, status, detail[:500]),
    )
    conn.commit()


def summary(conn: sqlite3.Connection) -> dict:
    """Counts and date ranges, used by the CLI and the Telegram /status command."""
    prices = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(date), MAX(date) FROM prices"
    ).fetchone()
    news = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(trading_day), MAX(trading_day) FROM news"
    ).fetchone()
    earnings = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM earnings").fetchone()
    last = conn.execute(
        "SELECT finished_at, step, status, detail FROM refresh_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {
        "price_rows": prices[0], "price_tickers": prices[1], "price_from": prices[2], "price_to": prices[3],
        "news_rows": news[0], "news_tickers": news[1], "news_from": news[2], "news_to": news[3],
        "earnings_rows": earnings[0], "earnings_from": earnings[1], "earnings_to": earnings[2],
        "last_refresh": last[0] if last else None,
    }
