"""Phase 1 data pipeline. One command refreshes everything:

    python -m signalbot.refresh              # prices, news, earnings
    python -m signalbot.refresh --only prices
    python -m signalbot.refresh --notify     # also send a summary to Telegram

Safe to run as often as you like: it only fetches what's new and never duplicates rows.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from . import db
from .align import assign_trading_day, iso_utc, story_key, to_utc
from .config import Settings, load_settings
from .notify import escape, send_message
from .sources import earnings as earnings_src
from .sources import news as news_src
from .sources import prices as prices_src

PRICE_OVERLAP_DAYS = 7   # re-pull the last week so late corrections land
NEWS_OVERLAP_DAYS = 1


@dataclass
class StepResult:
    step: str
    rows: int
    status: str          # ok | skipped | error
    detail: str = ""


# ---------------------------------------------------------------- helpers

def prepare_news(rows: list[dict], trading_days: list[str], tz: str, cutoff: str) -> list[dict]:
    """Add story keys and trading days, and keep only the earliest copy of each story per ticker."""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = story_key(r["headline"])
        row = {
            **r,
            "story_key": key,
            "trading_day": assign_trading_day(r["published_at"], trading_days, tz, cutoff),
        }
        k = (row["ticker"], key)
        if k not in best or row["published_at"] < best[k]["published_at"]:
            best[k] = row
    return list(best.values())


def _news_window(conn, provider: str, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.now(tz="UTC")
    last = db.last_news_time(conn, provider)
    if last:
        start = to_utc(last) - pd.Timedelta(days=NEWS_OVERLAP_DAYS)
    else:
        start = end - pd.Timedelta(days=lookback_days)
    return start, end


# ---------------------------------------------------------------- steps

def refresh_prices(conn, s: Settings, fetch=prices_src.fetch_prices_yf) -> StepResult:
    starts = []
    for t in s.price_tickers:
        last = db.last_price_date(conn, t)
        if last:
            starts.append((date.fromisoformat(last) - timedelta(days=PRICE_OVERLAP_DAYS)).isoformat())
        else:
            starts.append(s.history_start)
    start = min(starts)
    df = fetch(s.price_tickers, start)
    n = db.upsert_prices(conn, df)
    got = set(df["ticker"]) if not df.empty else set()
    missing = [t for t in s.price_tickers if t not in got]
    detail = f"from {start}"
    if missing:
        detail += f"; no data for {', '.join(missing)}"
    return StepResult("prices", n, "ok", detail)


def refresh_news(
    conn,
    s: Settings,
    fetch_alpaca=news_src.fetch_alpaca_news,
    fetch_finnhub=news_src.fetch_finnhub_news,
) -> list[StepResult]:
    days = db.trading_days(conn)
    results: list[StepResult] = []

    if s.has_alpaca:
        start, end = _news_window(conn, "alpaca", s.news_lookback_days)
        raw = fetch_alpaca(s.watchlist, iso_utc(start), iso_utc(end), s.alpaca_api_key, s.alpaca_secret_key)
        rows = prepare_news(news_src.parse_alpaca_news(raw, s.watchlist), days, s.market_timezone, s.news_cutoff)
        n = db.insert_news(conn, rows)
        results.append(StepResult("news:alpaca", n, "ok", f"{len(raw)} stories since {start.date()}"))
    else:
        results.append(StepResult("news:alpaca", 0, "skipped", "ALPACA_API_KEY / ALPACA_SECRET_KEY not set"))

    if s.has_finnhub:
        start, end = _news_window(conn, "finnhub", s.news_lookback_days)
        total_raw, total_new = 0, 0
        for ticker in s.watchlist:
            raw = fetch_finnhub(ticker, start.date(), end.date(), s.finnhub_api_key)
            total_raw += len(raw)
            rows = prepare_news(news_src.parse_finnhub_news(raw, ticker), days, s.market_timezone, s.news_cutoff)
            total_new += db.insert_news(conn, rows)
        results.append(StepResult("news:finnhub", total_new, "ok", f"{total_raw} stories since {start.date()}"))
    else:
        results.append(StepResult("news:finnhub", 0, "skipped", "FINNHUB_API_KEY not set"))

    return results


def refresh_earnings(conn, s: Settings, fetch=earnings_src.fetch_finnhub_earnings) -> StepResult:
    if not s.has_finnhub:
        return StepResult("earnings", 0, "skipped", "FINNHUB_API_KEY not set")
    today = date.today()
    start, end = today - timedelta(days=400), today + timedelta(days=90)
    n = 0
    for ticker in s.watchlist:
        n += db.upsert_earnings(conn, earnings_src.parse_finnhub_earnings(fetch(ticker, start, end, s.finnhub_api_key)))
    return StepResult("earnings", n, "ok", f"{start} to {end}")


# ---------------------------------------------------------------- CLI

def _run(conn, name: str, fn) -> list[StepResult]:
    started = db.now_utc()
    try:
        out = fn()
        results = out if isinstance(out, list) else [out]
    except Exception as exc:  # keep going so one bad source doesn't block the others
        msg = str(exc) or exc.__class__.__name__
        if not isinstance(exc, news_src.SourceError):
            traceback.print_exc()
        results = [StepResult(name, 0, "error", msg)]
    for r in results:
        db.log_step(conn, r.step, started, r.rows, r.status, r.detail)
    return results


def format_report(results: list[StepResult], summary: dict) -> str:
    lines = ["Refresh results:"]
    for r in results:
        mark = {"ok": "OK  ", "skipped": "SKIP", "error": "FAIL"}[r.status]
        lines.append(f"  [{mark}] {r.step:<13} {r.rows:>7} rows  {r.detail}")
    lines += [
        "",
        "Database now holds:",
        f"  prices:   {summary['price_rows']:>7} rows, {summary['price_tickers']} tickers, "
        f"{summary['price_from']} to {summary['price_to']}",
        f"  news:     {summary['news_rows']:>7} stories, {summary['news_tickers']} tickers, "
        f"{summary['news_from']} to {summary['news_to']}",
        f"  earnings: {summary['earnings_rows']:>7} dates",
    ]
    return "\n".join(lines)


def format_telegram(results: list[StepResult], summary: dict) -> str:
    icon = {"ok": "✅", "skipped": "⏭", "error": "❌"}
    parts = ["<b>Data refresh</b>"]
    for r in results:
        parts.append(f"{icon[r.status]} {escape(r.step)}: {r.rows} rows" + (f" · {escape(r.detail)}" if r.status != "ok" else ""))
    parts.append(
        f"\nPrices through <code>{escape(summary['price_to'])}</code> · "
        f"{summary['news_rows']} news stories stored"
    )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh prices, news, and earnings into the local database.")
    parser.add_argument("--only", choices=["prices", "news", "earnings"], help="run a single step")
    parser.add_argument("--notify", action="store_true", help="send a summary to your Telegram chat")
    args = parser.parse_args(argv)

    s = load_settings()
    conn = db.connect(s.database_path)
    steps = {
        "prices": lambda: refresh_prices(conn, s),
        "news": lambda: refresh_news(conn, s),
        "earnings": lambda: refresh_earnings(conn, s),
    }
    # Prices run first: news alignment needs the list of trading days.
    order = [args.only] if args.only else ["prices", "news", "earnings"]

    results: list[StepResult] = []
    for name in order:
        print(f"Refreshing {name}...", flush=True)
        results += _run(conn, name, steps[name])

    summary = db.summary(conn)
    print()
    print(format_report(results, summary))

    if args.notify:
        if s.has_telegram:
            try:
                send_message(s.telegram_bot_token, s.telegram_chat_id, format_telegram(results, summary))
                print("\nSent summary to Telegram.")
            except Exception as exc:
                print(f"\nCouldn't send to Telegram: {exc}")
        else:
            print("\n--notify skipped: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

    conn.close()
    return 1 if any(r.status == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
