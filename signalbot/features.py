"""Phase 2: turn raw prices, news, and earnings into one feature table.

    python -m signalbot.features              # build the table
    python -m signalbot.features --show NVDA  # print NVDA's latest rows

One row per stock per trading day. Every column in a row uses only information
that was public by that day's 4:00 PM ET close. The label is whether the stock
closed higher the NEXT trading day; the newest row has no label yet, and that's
the row the model will predict from.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import db
from .config import Settings, load_settings
from .sentiment import coverage as sentiment_coverage
from .sentiment import ensure_table as ensure_sentiment_table

MARKET = "SPY"

# Columns the model may learn from. Everything else in the table is an ID, label, or helper.
TECH = [
    "ret_1d", "ret_5d", "ret_20d", "gap", "rsi_14", "macd_hist",
    "dist_sma20", "dist_sma50", "vol_ratio_20", "atr_pct_14", "volat_20", "range_pos_20",
]
CONTEXT = [
    "spy_ret_1d", "spy_ret_5d", "qqq_ret_1d", "vix", "vix_chg_1d", "vix_vs_20d",
    "sector_ret_1d", "rel_sector_5d", "rel_spy_5d", "tnx", "tnx_chg_5d", "curve",
]
NEWS = [
    "news_n", "news_n_ratio", "news_sent", "news_sent_min", "news_sent_max",
    "news_pos_share", "news_neg_share", "news_sent_3d", "news_focus_n",
]
EVENTS = ["earn_next", "earn_today", "days_to_earn", "dow"]
FEATURES = TECH + CONTEXT + NEWS + EVENTS
LABEL = "label_up"


# ---------------------------------------------------------------- loading

def load_prices(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT ticker, date, open, high, low, close, adj_close, volume FROM prices", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_news(conn) -> pd.DataFrame:
    ensure_sentiment_table(conn)
    df = pd.read_sql(
        """SELECT n.ticker, n.trading_day, n.n_tickers, s.score, s.pos, s.neg
           FROM news n LEFT JOIN news_sentiment s ON s.story_key = n.story_key""",
        conn,
    )
    df["date"] = pd.to_datetime(df["trading_day"])
    return df


def load_earnings(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT ticker, date, hour FROM earnings", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------- indicators

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def technicals(p: pd.DataFrame) -> pd.DataFrame:
    """p: one ticker's prices sorted by date. Uses split/dividend-adjusted prices."""
    p = p.copy()
    adj = p["adj_close"].fillna(p["close"])
    factor = (adj / p["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    o, h, l, c = p["open"] * factor, p["high"] * factor, p["low"] * factor, adj

    out = pd.DataFrame({"date": p["date"]})
    out["close"] = c
    out["ret_1d"] = c.pct_change(1, fill_method=None)
    out["ret_5d"] = c.pct_change(5, fill_method=None)
    out["ret_20d"] = c.pct_change(20, fill_method=None)
    out["gap"] = o / c.shift(1) - 1
    out["rsi_14"] = rsi(c, 14)

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    out.loc[out.index[:34], "macd_hist"] = np.nan          # not enough history yet

    out["dist_sma20"] = c / c.rolling(20).mean() - 1
    out["dist_sma50"] = c / c.rolling(50).mean() - 1
    vol = p["volume"].astype(float)
    out["vol_ratio_20"] = vol / vol.rolling(20).mean()

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr_pct_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    out["volat_20"] = out["ret_1d"].rolling(20).std()
    lo, hi = l.rolling(20).min(), h.rolling(20).max()
    out["range_pos_20"] = (c - lo) / (hi - lo).replace(0, np.nan)

    nxt = c.shift(-1)
    out["next_ret"] = nxt / c - 1
    out[LABEL] = np.where(nxt.isna(), np.nan, (nxt > c).astype(float))
    return out


def market_context(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily market-wide columns indexed by date, plus the wide table of adjusted closes."""
    wide = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    wide = wide.sort_index()

    def col(t):
        return wide[t] if t in wide.columns else pd.Series(np.nan, index=wide.index)

    spy, qqq, vix, tnx, irx = col(MARKET), col("QQQ"), col("^VIX"), col("^TNX"), col("^IRX")
    ctx = pd.DataFrame(index=wide.index)
    ctx["spy_ret_1d"] = spy.pct_change(1, fill_method=None)
    ctx["spy_ret_5d"] = spy.pct_change(5, fill_method=None)
    ctx["qqq_ret_1d"] = qqq.pct_change(1, fill_method=None)
    ctx["vix"] = vix
    ctx["vix_chg_1d"] = vix.diff()
    ctx["vix_vs_20d"] = vix / vix.rolling(20).mean() - 1
    ctx["tnx"] = tnx
    ctx["tnx_chg_5d"] = tnx.diff(5)
    ctx["curve"] = tnx - irx
    return ctx, wide


# ---------------------------------------------------------------- news + events

def news_daily(news: pd.DataFrame) -> pd.DataFrame:
    """Per ticker per trading day: how much news, and how positive."""
    if news.empty:
        cols = ["news_n", "news_focus_n", "news_scored", "news_sent_sum", "news_sent",
                "news_sent_min", "news_sent_max", "news_pos_share", "news_neg_share"]
        out = pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
        out.insert(0, "date", pd.Series(dtype="datetime64[ns]"))
        out.insert(0, "ticker", pd.Series(dtype=object))
        return out
    n = news.copy()
    n["is_pos"] = (n["score"] > 0.3).astype(float).where(n["score"].notna())
    n["is_neg"] = (n["score"] < -0.3).astype(float).where(n["score"].notna())
    n["focused"] = (n["n_tickers"].fillna(1) <= 2).astype(int)
    g = n.groupby(["ticker", "date"])
    out = pd.DataFrame({
        "news_n": g.size(),
        "news_focus_n": g["focused"].sum(),
        "news_scored": g["score"].count(),
        "news_sent_sum": g["score"].sum(min_count=1),
        "news_sent": g["score"].mean(),
        "news_sent_min": g["score"].min(),
        "news_sent_max": g["score"].max(),
        "news_pos_share": g["is_pos"].mean(),
        "news_neg_share": g["is_neg"].mean(),
    }).reset_index()
    return out


def add_news(feat: pd.DataFrame, daily: pd.DataFrame, coverage_start: pd.Timestamp | None) -> pd.DataFrame:
    f = feat.merge(daily, on=["ticker", "date"], how="left")
    for c in ["news_n", "news_focus_n", "news_scored"]:
        f[c] = f[c].fillna(0)
    f["news_sent_sum"] = f["news_sent_sum"].fillna(0)
    # No news that day = neutral, not unknown.
    for c in ["news_sent", "news_sent_min", "news_sent_max", "news_pos_share", "news_neg_share"]:
        f[c] = f[c].where(f["news_scored"] > 0, 0.0)

    g = f.groupby("ticker", group_keys=False)
    avg20 = g["news_n"].transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
    f["news_n_ratio"] = f["news_n"] / avg20.replace(0, np.nan)
    # Not enough history for an average yet: call any news "busy", none "normal".
    fallback = pd.Series(np.where(f["news_n"] > 0, 3.0, 1.0), index=f.index)
    f["news_n_ratio"] = f["news_n_ratio"].fillna(fallback)
    s3 = g["news_sent_sum"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    n3 = g["news_scored"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    f["news_sent_3d"] = (s3 / n3.replace(0, np.nan)).fillna(0.0)

    # Before news history starts, these are unknown, not "zero news".
    if coverage_start is not None:
        before = f["date"] < coverage_start
        f.loc[before, NEWS] = np.nan
        f["news_covered"] = (~before).astype(int)
    else:
        f[NEWS] = np.nan
        f["news_covered"] = 0
    return f.drop(columns=["news_sent_sum", "news_scored"])


def add_earnings(feat: pd.DataFrame, earnings: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """earn_next = 1 when tomorrow's move includes an earnings reaction
    (report after today's close, or before tomorrow's open)."""
    f = feat.copy()
    f["earn_next"], f["earn_today"], f["days_to_earn"] = 0.0, 0.0, np.nan
    if earnings.empty:
        f[["earn_next", "earn_today", "days_to_earn"]] = np.nan
        return f

    pos = {d: i for i, d in enumerate(days)}

    def session_index(d: pd.Timestamp) -> int | None:
        i = days.searchsorted(d)
        return int(i) if i < len(days) else None

    covered_from = earnings["date"].min() - pd.Timedelta(days=100)
    for ticker, e in earnings.groupby("ticker"):
        mask = f["ticker"] == ticker
        if not mask.any():
            continue
        # Trading-day index of the session whose return reacts to each report
        reacts = []
        for d, hour in zip(e["date"], e["hour"].fillna("")):
            i = session_index(d)
            if i is None:
                reacts.append(len(days) + (d - days[-1]).days)   # future, beyond known calendar
                continue
            reacts.append(i + 1 if hour == "amc" else i)
        reacts = np.array(sorted(set(reacts)))
        idx = f.loc[mask, "date"].map(pos).to_numpy()
        nxt = idx + 1
        f.loc[mask, "earn_next"] = np.isin(nxt, reacts).astype(float)
        f.loc[mask, "earn_today"] = np.isin(idx, reacts).astype(float)
        j = np.searchsorted(reacts, nxt)
        dist = np.where(j < len(reacts), reacts[np.minimum(j, len(reacts) - 1)] - idx, np.nan)
        f.loc[mask, "days_to_earn"] = np.minimum(dist, 60)
    unknown = f["date"] < covered_from
    f.loc[unknown, ["earn_next", "earn_today", "days_to_earn"]] = np.nan
    return f


# ---------------------------------------------------------------- build

def build_features(prices: pd.DataFrame, news: pd.DataFrame, earnings: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "date", *FEATURES, "next_ret", LABEL])
    ctx, wide = market_context(prices)
    days = pd.DatetimeIndex(sorted(prices.loc[prices["ticker"] == MARKET, "date"].unique()))
    if days.empty:
        days = pd.DatetimeIndex(sorted(prices["date"].unique()))

    parts = []
    for ticker in s.watchlist:
        p = prices[prices["ticker"] == ticker]
        if p.empty:
            continue
        t = technicals(p)
        t.insert(0, "ticker", ticker)
        sector = s.sectors.get(ticker, MARKET)
        sec = wide[sector] if sector in wide.columns else wide.get(MARKET)
        t = t.join(ctx, on="date")
        if sec is not None:
            sec_ret1 = sec.pct_change(1, fill_method=None)
            sec_ret5 = sec.pct_change(5, fill_method=None)
            t["sector_ret_1d"] = t["date"].map(sec_ret1)
            t["rel_sector_5d"] = t["ret_5d"] - t["date"].map(sec_ret5)
        else:
            t["sector_ret_1d"] = t["rel_sector_5d"] = np.nan
        t["rel_spy_5d"] = t["ret_5d"] - t["spy_ret_5d"]
        parts.append(t)

    feat = pd.concat(parts, ignore_index=True)
    feat = feat[feat["date"].isin(days)].copy()   # market sessions only
    feat["dow"] = feat["date"].dt.dayofweek.astype(float)

    daily = news_daily(news)
    start = news["date"].min() if not news.empty else None
    feat = add_news(feat, daily, start)
    feat = add_earnings(feat, earnings, days)

    cols = ["ticker", "date", "close", *FEATURES, "news_covered", "next_ret", LABEL]
    return feat[cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def save(conn, feat: pd.DataFrame) -> None:
    out = feat.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_sql("features", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features ON features (ticker, date)")
    conn.commit()


def report(feat: pd.DataFrame, scored: int, total: int) -> str:
    labeled = feat[LABEL].notna()
    usable = labeled & feat[TECH + CONTEXT].notna().all(axis=1)
    with_news = usable & (feat["news_covered"] == 1)
    up = feat.loc[labeled, LABEL].mean() if labeled.any() else float("nan")
    latest = feat["date"].max()
    lines = [
        f"Feature table: {len(feat):,} rows, {feat['ticker'].nunique()} stocks, "
        f"{feat['date'].min():%Y-%m-%d} to {latest:%Y-%m-%d}",
        f"  {len(FEATURES)} features: {len(TECH)} technical, {len(CONTEXT)} market, "
        f"{len(NEWS)} news, {len(EVENTS)} calendar",
        f"  rows ready for training:  {usable.sum():,}  ({with_news.sum():,} with news coverage)",
        f"  rows to predict (no label yet): {int((~labeled).sum())}  (latest day {latest:%Y-%m-%d})",
        f"  base rate: stocks closed up the next day {up:.1%} of the time  <- the number to beat",
        f"  headlines with sentiment: {scored:,} of {total:,}",
    ]
    if total and scored < total:
        lines.append("  -> run `python -m signalbot.sentiment` to score the rest, then rebuild")
    return "\n".join(lines)


def show(feat: pd.DataFrame, ticker: str, n: int = 5) -> str:
    cols = ["date", "close", "ret_1d", "rsi_14", "dist_sma20", "vol_ratio_20", "vix",
            "news_n", "news_sent", "earn_next", LABEL]
    sub = feat[feat["ticker"] == ticker.upper()].tail(n)[cols].copy()
    if sub.empty:
        return f"No rows for {ticker}. Is it in the watchlist?"
    sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        return sub.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the feature table from stored data.")
    p.add_argument("--show", metavar="TICKER", help="print this stock's latest rows after building")
    args = p.parse_args(argv)

    s = load_settings()
    conn = db.connect(s.database_path)
    prices = load_prices(conn)
    if prices.empty:
        print("No prices yet. Run `python -m signalbot.refresh` first.")
        return 1
    missing = [t for t in ("SPY", "^VIX", "^TNX") if t not in set(prices["ticker"])]
    if missing:
        print(f"Note: no prices for {', '.join(missing)} yet; run `python -m signalbot.refresh` to fill them in.")

    print("Building features...", flush=True)
    feat = build_features(prices, load_news(conn), load_earnings(conn), s)
    save(conn, feat)
    scored, total = sentiment_coverage(conn)
    print(report(feat, scored, total))
    if args.show:
        print()
        print(show(feat, args.show))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
