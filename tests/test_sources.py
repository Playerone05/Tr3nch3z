import numpy as np
import pandas as pd

from signalbot.sources.earnings import parse_finnhub_earnings
from signalbot.sources.news import parse_alpaca_news, parse_finnhub_news
from signalbot.sources.prices import reshape_yf

ALPACA_SAMPLE = [
    {
        "id": 1, "headline": "NVIDIA Unveils New Chip", "created_at": "2026-10-13T14:30:00Z",
        "summary": "x", "url": "https://example.com/1", "source": "benzinga",
        "symbols": ["NVDA", "AMD", "INTC"],
    },
    {"id": 2, "headline": "", "created_at": "2026-10-13T14:31:00Z", "symbols": ["NVDA"]},     # dropped: no headline
    {"id": 3, "headline": "Bank earnings preview", "created_at": "2026-10-13T15:00:00Z", "symbols": ["WFC"]},  # not watched
]

FINNHUB_SAMPLE = [
    {"category": "company", "datetime": 1760365800, "headline": "Apple supplier news",
     "id": 9, "related": "AAPL", "source": "Yahoo", "summary": "s", "url": "https://example.com/a"},
    {"datetime": None, "headline": "broken"},
]


def test_parse_alpaca_keeps_only_watchlist_symbols():
    rows = parse_alpaca_news(ALPACA_SAMPLE, ["NVDA", "AMD", "AAPL"])
    assert sorted(r["ticker"] for r in rows) == ["AMD", "NVDA"]
    nv = next(r for r in rows if r["ticker"] == "NVDA")
    assert nv["published_at"] == "2026-10-13T14:30:00Z"
    assert nv["n_tickers"] == 3
    assert nv["provider"] == "alpaca"


def test_parse_finnhub_converts_unix_time():
    rows = parse_finnhub_news(FINNHUB_SAMPLE, "aapl")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["published_at"] == "2025-10-13T14:30:00Z"
    assert rows[0]["n_tickers"] == 1


def test_parse_finnhub_earnings():
    rows = parse_finnhub_earnings([
        {"date": "2026-10-29", "epsActual": None, "epsEstimate": 1.62, "hour": "amc",
         "quarter": 4, "revenueActual": None, "revenueEstimate": 1.2e11, "symbol": "AAPL", "year": 2026},
        {"date": "", "symbol": "MSFT"},
    ])
    assert rows == [{
        "ticker": "AAPL", "date": "2026-10-29", "hour": "amc", "eps_estimate": 1.62,
        "eps_actual": None, "revenue_estimate": 1.2e11, "revenue_actual": None,
    }]


def _yf_frame(tickers, field_first=False):
    idx = pd.to_datetime(["2026-10-15", "2026-10-16"])
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = []
    data = []
    for t in tickers:
        for f in fields:
            cols.append((f, t) if field_first else (t, f))
            data.append([100.0, 101.0] if f != "Volume" else [1e6, 2e6])
    frame = pd.DataFrame(np.array(data).T, index=idx, columns=pd.MultiIndex.from_tuples(cols))
    return frame


def test_reshape_yf_ticker_first():
    out = reshape_yf(_yf_frame(["SPY", "AAPL"]), ["SPY", "AAPL", "MISSING"])
    assert len(out) == 4
    assert set(out["ticker"]) == {"SPY", "AAPL"}
    assert list(out["date"].unique()) == ["2026-10-15", "2026-10-16"]
    assert out["adj_close"].notna().all()


def test_reshape_yf_field_first():
    out = reshape_yf(_yf_frame(["^VIX"], field_first=True), ["^VIX"])
    assert len(out) == 2 and set(out["ticker"]) == {"^VIX"}


def test_reshape_yf_drops_rows_without_close():
    frame = _yf_frame(["SPY"])
    frame.loc[frame.index[1], ("SPY", "Close")] = np.nan
    out = reshape_yf(frame, ["SPY"])
    assert list(out["date"]) == ["2026-10-15"]
