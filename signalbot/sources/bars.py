"""Minute bars from Alpaca's market data API (free with a paper-trading account).

The free plan includes the IEX feed in real time. IEX is one exchange, so its
volume is a small slice of the market, but for big stocks its prices track the
full market closely. We use IEX for both history and live calls, so the bot
learns from the same kind of data it will see when you ask for a call.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .news import TIMEOUT, SourceError, _check

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
BAR_COLUMNS = ["ticker", "ts", "open", "high", "low", "close", "volume", "vwap", "trades"]


def fetch_minute_bars(
    symbols: list[str],
    start: str,
    end: str,
    key: str,
    secret: str,
    feed: str = "iex",
    session: requests.Session | None = None,
    pause: float = 0.3,
    max_pages: int = 2000,
    progress=None,
) -> list[tuple[str, dict]]:
    """Returns (symbol, raw bar) pairs between start and end (ISO UTC).
    Crypto symbols contain a slash (BTC/USD) and use Alpaca's crypto endpoint."""
    crypto = all("/" in x for x in symbols)
    s = session or requests.Session()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ",".join(symbols), "timeframe": "1Min", "start": start, "end": end,
              "limit": 10000, "sort": "asc"}
    if not crypto:
        params.update(feed=feed, adjustment="raw")
    url = ALPACA_CRYPTO_BARS_URL if crypto else ALPACA_BARS_URL
    out: list[tuple[str, dict]] = []
    for page in range(1, max_pages + 1):
        resp = s.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if resp.status_code == 403 and "subscription" in resp.text.lower():
            raise SourceError(f"Alpaca says this data feed isn't on your plan ({feed}). "
                              "Set trench.feed to 'iex' in config.yaml.")
        _check(resp, "Alpaca bars")
        body = resp.json()
        for sym, bars in (body.get("bars") or {}).items():
            out.extend((sym, b) for b in bars)
        if progress and page % 10 == 0:
            progress(f"  minute bars: page {page}, {len(out):,} bars so far")
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        time.sleep(pause)
    return out


def parse_minute_bars(raw: list[tuple[str, dict]]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = pd.DataFrame([{
        "ticker": sym.upper(), "ts": b.get("t"), "open": b.get("o"), "high": b.get("h"),
        "low": b.get("l"), "close": b.get("c"), "volume": b.get("v"),
        "vwap": b.get("vw"), "trades": b.get("n"),
    } for sym, b in raw])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("min")
    df = df.dropna(subset=["close"]).drop_duplicates(["ticker", "ts"], keep="last")
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)[BAR_COLUMNS]
