"""Earnings dates from Finnhub's earnings calendar."""

from __future__ import annotations

import time
from datetime import date

import requests

from .news import TIMEOUT, _check

FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"


def fetch_finnhub_earnings(
    symbol: str,
    start: date,
    end: date,
    token: str,
    session: requests.Session | None = None,
    pause: float = 1.1,
) -> list[dict]:
    s = session or requests.Session()
    resp = s.get(
        FINNHUB_EARNINGS_URL,
        params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat(), "token": token},
        timeout=TIMEOUT,
    )
    _check(resp, "Finnhub earnings")
    time.sleep(pause)
    return (resp.json() or {}).get("earningsCalendar", []) or []


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def parse_finnhub_earnings(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        if not item.get("symbol") or not item.get("date"):
            continue
        rows.append({
            "ticker": item["symbol"].upper(),
            "date": item["date"],
            "hour": item.get("hour") or "",
            "eps_estimate": _num(item.get("epsEstimate")),
            "eps_actual": _num(item.get("epsActual")),
            "revenue_estimate": _num(item.get("revenueEstimate")),
            "revenue_actual": _num(item.get("revenueActual")),
        })
    return rows
