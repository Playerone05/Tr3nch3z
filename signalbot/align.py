"""Timestamp rules that keep future information out of the model.

The bot predicts from each day's close. A headline can only count toward a
trading day if it was public before that day's 4:00 PM ET close. Anything later
(after hours, overnight, weekends, holidays) rolls forward to the next session.
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left
from datetime import date, datetime, time, timedelta

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_headline(headline: str) -> str:
    text = (headline or "").lower()
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def story_key(headline: str) -> str:
    """Same headline from different outlets -> same key, so a story counts once."""
    return hashlib.sha1(normalize_headline(headline).encode("utf-8")).hexdigest()[:16]


def to_utc(value) -> pd.Timestamp:
    """Accepts unix seconds, ISO strings, datetimes. Returns a tz-aware UTC Timestamp."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.Timestamp(int(value), unit="s", tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_cutoff(cutoff: str) -> time:
    hours, minutes = cutoff.split(":")
    return time(int(hours), int(minutes))


def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d += timedelta(days=1)
    return d


def assign_trading_day(
    published,
    trading_days: list[str],
    tz: str = "America/New_York",
    cutoff: str = "16:00",
) -> str:
    """Return the trading day (YYYY-MM-DD) this news item belongs to.

    trading_days must be sorted ascending. Outside the range it covers (very old
    news, or today's news before today's prices exist) we fall back to the next
    weekday, which is right except on market holidays.
    """
    local = to_utc(published).tz_convert(tz)
    candidate = local.date()
    if local.time() >= _parse_cutoff(cutoff):
        candidate += timedelta(days=1)

    key = candidate.isoformat()
    if trading_days and trading_days[0] <= key:
        i = bisect_left(trading_days, key)
        if i < len(trading_days):
            return trading_days[i]
    return _next_weekday(candidate).isoformat()


def iso_utc(value) -> str:
    return to_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> datetime:
    return pd.Timestamp.now(tz="UTC").to_pydatetime()
