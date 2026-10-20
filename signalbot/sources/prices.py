"""Daily price history from Yahoo Finance (no API key needed)."""

from __future__ import annotations

import pandas as pd

from ..db import PRICE_COLUMNS

_FIELD_MAP = {
    "open": "open", "high": "high", "low": "low", "close": "close",
    "adj close": "adj_close", "adj_close": "adj_close", "volume": "volume",
}


def fetch_prices_yf(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf  # imported here so tests don't need network access

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    return reshape_yf(raw, tickers)


def _tidy_one(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = frame.rename(columns=lambda c: _FIELD_MAP.get(str(c).strip().lower(), str(c).lower()))
    frame = frame.dropna(how="all")
    if "close" not in frame.columns:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = frame.dropna(subset=["close"])
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    out = pd.DataFrame({
        "ticker": ticker,
        "date": pd.to_datetime(frame.index).strftime("%Y-%m-%d"),
        "open": frame.get("open"),
        "high": frame.get("high"),
        "low": frame.get("low"),
        "close": frame["close"],
        "adj_close": frame["adj_close"],
        "volume": frame.get("volume"),
        "source": "yfinance",
    })
    return out.reset_index(drop=True)[PRICE_COLUMNS]


def reshape_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Turn yfinance's wide download into one long table: one row per ticker per day."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    parts = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        for t in tickers:
            if t in level0:
                parts.append(_tidy_one(raw[t].copy(), t))
            elif t in level1:  # some yfinance versions put the field first
                parts.append(_tidy_one(raw.xs(t, axis=1, level=1).copy(), t))
    elif len(tickers) == 1:
        parts.append(_tidy_one(raw.copy(), tickers[0]))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(parts, ignore_index=True)
