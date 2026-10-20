"""Company news from Alpaca and Finnhub.

Fetch functions call the APIs; parse functions turn raw JSON into rows and are
covered by tests with sample payloads.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import requests

from ..align import iso_utc

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

TIMEOUT = 30


class SourceError(RuntimeError):
    pass


def _check(resp: requests.Response, name: str) -> None:
    if resp.status_code in (401, 403):
        raise SourceError(f"{name} rejected the API key (HTTP {resp.status_code}). Check your .env values.")
    if resp.status_code == 429:
        raise SourceError(f"{name} rate limit hit. Wait a minute and run the refresh again.")
    if resp.status_code >= 400:
        raise SourceError(f"{name} returned HTTP {resp.status_code}: {resp.text[:200]}")


# ---------------------------------------------------------------- Alpaca

def fetch_alpaca_news(
    symbols: list[str],
    start: str,
    end: str,
    key: str,
    secret: str,
    session: requests.Session | None = None,
    max_pages: int = 400,
    pause: float = 0.35,
    on_batch=None,
    progress=print,
) -> list[dict]:
    """All stories tagged with any of `symbols` between start and end (ISO UTC).

    on_batch(items) is called after each page so callers can save as they go;
    stopping partway keeps everything fetched so far.
    """
    s = session or requests.Session()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "symbols": ",".join(symbols),
        "start": start,
        "end": end,
        "limit": 50,
        "sort": "asc",
        "include_content": "false",
    }
    items: list[dict] = []
    for page in range(1, max_pages + 1):
        resp = s.get(ALPACA_NEWS_URL, headers=headers, params=params, timeout=TIMEOUT)
        _check(resp, "Alpaca News")
        body = resp.json()
        batch = body.get("news", [])
        items.extend(batch)
        if on_batch and batch:
            on_batch(batch)
        if progress and (page % 10 == 0 or not body.get("next_page_token")):
            latest = (batch[-1].get("created_at") or "")[:10] if batch else ""
            progress(f"  alpaca: page {page}, {len(items)} stories so far (up to {latest})")
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        time.sleep(pause)
    else:
        if progress:
            progress(f"  alpaca: stopped at {max_pages} pages; the next refresh continues from here")
    return items


def parse_alpaca_news(items: list[dict], watchlist: list[str]) -> list[dict]:
    wanted = set(watchlist)
    rows = []
    for item in items:
        headline = (item.get("headline") or "").strip()
        published = item.get("created_at") or item.get("updated_at")
        if not headline or not published:
            continue
        symbols = [s.upper() for s in item.get("symbols") or []]
        for ticker in symbols:
            if ticker not in wanted:
                continue
            rows.append({
                "ticker": ticker,
                "published_at": iso_utc(published),
                "headline": headline,
                "summary": (item.get("summary") or "").strip(),
                "source": item.get("source") or "",
                "url": item.get("url") or "",
                "provider": "alpaca",
                "n_tickers": len(symbols),
            })
    return rows


# ---------------------------------------------------------------- Finnhub

def fetch_finnhub_news(
    symbol: str,
    start: date,
    end: date,
    token: str,
    session: requests.Session | None = None,
    window_days: int = 30,
    pause: float = 1.1,
) -> list[dict]:
    """Finnhub caps results per call, so ask in 30-day windows."""
    s = session or requests.Session()
    items: list[dict] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=window_days - 1), end)
        resp = s.get(
            FINNHUB_NEWS_URL,
            params={"symbol": symbol, "from": cursor.isoformat(), "to": window_end.isoformat(), "token": token},
            timeout=TIMEOUT,
        )
        _check(resp, "Finnhub")
        body = resp.json()
        if isinstance(body, list):
            items.extend(body)
        cursor = window_end + timedelta(days=1)
        time.sleep(pause)  # free tier allows about 60 calls a minute
    return items


def parse_finnhub_news(items: list[dict], symbol: str) -> list[dict]:
    rows = []
    for item in items:
        headline = (item.get("headline") or "").strip()
        ts = item.get("datetime")
        if not headline or not ts:
            continue
        related = [r.strip() for r in (item.get("related") or symbol).split(",") if r.strip()]
        rows.append({
            "ticker": symbol.upper(),
            "published_at": iso_utc(int(ts)),
            "headline": headline,
            "summary": (item.get("summary") or "").strip(),
            "source": item.get("source") or "",
            "url": item.get("url") or "",
            "provider": "finnhub",
            "n_tickers": max(len(related), 1),
        })
    return rows
