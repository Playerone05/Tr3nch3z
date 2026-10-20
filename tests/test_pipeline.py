"""End-to-end refresh with fake data sources (no network, no keys)."""

import pandas as pd
import pytest

from signalbot import db
from signalbot.config import Settings
from signalbot.refresh import prepare_news, refresh_earnings, refresh_news, refresh_prices

DATES = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19"]


def fake_prices(tickers, start):
    rows = []
    for t in tickers:
        for i, d in enumerate(DATES):
            px = 100 + i
            rows.append({"ticker": t, "date": d, "open": px, "high": px + 1, "low": px - 1,
                         "close": px, "adj_close": px, "volume": 1000, "source": "fake"})
    return pd.DataFrame(rows)


def fake_alpaca(symbols, start, end, key, secret):
    return [
        # Same story twice; the earlier copy should win.
        {"headline": "NVDA beats estimates!", "created_at": "2026-10-15T21:05:00Z", "symbols": ["NVDA"]},
        {"headline": "nvda beats estimates", "created_at": "2026-10-15T20:30:00Z", "symbols": ["NVDA"]},
        {"headline": "Weekend chip feature", "created_at": "2026-10-17T15:00:00Z", "symbols": ["NVDA", "AMD"]},
    ]


def fake_finnhub(symbol, start, end, token):
    if symbol != "NVDA":
        return []
    # Same headline as Alpaca's, published later -> not a new story
    later = int(pd.Timestamp("2026-10-15T22:00:00Z").timestamp())
    return [{"datetime": later, "headline": "NVDA Beats Estimates", "related": "NVDA"}]


def fake_earnings(symbol, start, end, token):
    return [{"symbol": symbol, "date": "2026-11-19", "hour": "amc", "epsEstimate": 1.0}]


@pytest.fixture
def setup(tmp_path):
    s = Settings(
        watchlist=["NVDA", "AMD"], context=["SPY"], database_path=tmp_path / "t.db",
        alpaca_api_key="k", alpaca_secret_key="s", finnhub_api_key="f",
    )
    conn = db.connect(s.database_path)
    yield s, conn
    conn.close()


def test_full_refresh(setup):
    s, conn = setup

    r = refresh_prices(conn, s, fetch=fake_prices)
    assert r.status == "ok" and r.rows == 12          # 3 tickers x 4 days
    assert db.trading_days(conn) == DATES

    results = refresh_news(conn, s, fetch_alpaca=fake_alpaca, fetch_finnhub=fake_finnhub)
    assert [x.status for x in results] == ["ok", "ok"]

    stories = conn.execute(
        "SELECT ticker, published_at, trading_day, provider FROM news ORDER BY ticker, published_at"
    ).fetchall()
    assert stories == [
        ("AMD", "2026-10-17T15:00:00Z", "2026-10-19", "alpaca"),       # Saturday -> Monday
        ("NVDA", "2026-10-15T20:30:00Z", "2026-10-16", "alpaca"),      # 4:30 PM ET -> next day, earliest copy kept
        ("NVDA", "2026-10-17T15:00:00Z", "2026-10-19", "alpaca"),
    ]

    e = refresh_earnings(conn, s, fetch=fake_earnings)
    assert e.rows == 2


def test_refresh_is_repeatable(setup):
    s, conn = setup
    refresh_prices(conn, s, fetch=fake_prices)
    refresh_news(conn, s, fetch_alpaca=fake_alpaca, fetch_finnhub=fake_finnhub)
    second = refresh_news(conn, s, fetch_alpaca=fake_alpaca, fetch_finnhub=fake_finnhub)
    assert sum(x.rows for x in second) == 0
    assert conn.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 3


def test_earlier_copy_replaces_later_one(setup):
    s, conn = setup
    late = prepare_news(
        [{"ticker": "NVDA", "published_at": "2026-10-15T22:00:00Z", "headline": "Big news"}], DATES,
        "America/New_York", "16:00")
    early = prepare_news(
        [{"ticker": "NVDA", "published_at": "2026-10-15T13:00:00Z", "headline": "BIG NEWS"}], DATES,
        "America/New_York", "16:00")
    db.insert_news(conn, late)
    db.insert_news(conn, early)
    row = conn.execute("SELECT published_at, trading_day FROM news").fetchone()
    assert row == ("2026-10-15T13:00:00Z", "2026-10-15")


def test_missing_keys_skip_cleanly(tmp_path):
    s = Settings(watchlist=["NVDA"], database_path=tmp_path / "t.db")
    conn = db.connect(s.database_path)
    assert [r.status for r in refresh_news(conn, s)] == ["skipped", "skipped"]
    assert refresh_earnings(conn, s).status == "skipped"
    conn.close()


def test_summary_on_empty_db(tmp_path):
    conn = db.connect(tmp_path / "e.db")
    assert db.summary(conn)["price_rows"] == 0
    conn.close()
