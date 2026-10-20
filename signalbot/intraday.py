"""Trench mode: 1-5 minute calls on stocks, crypto, and commodities.

    python -m signalbot.intraday refresh      # pull minute bars + news for every Trench market
    python -m signalbot.intraday backtest     # walk-forward test, then train and save the live models
    python -m signalbot.intraday call NVDA    # one live call
    python -m signalbot.intraday scan         # every market, right now

How a call is made
- Features come from the last hour of minute bars: recent moves, volume spikes,
  distance from VWAP, minute RSI, what SPY/QQQ (or BTC for crypto) are doing,
  and how many headlines just hit and how positive they were.
- A model per hold time (1, 2, 3, 5 min) gives the chance price is higher after
  that hold, measured from the price ONE MINUTE after the alert (your reaction time).
- The backtest decides, per market type and hold time, how confident the model
  must be before it's worth a trade, and whether that hold has any tested edge.
- The bot picks the hold with the best tested result. If none qualifies: No trade.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import db
from .align import assign_trading_day, iso_utc, story_key
from .config import ROOT, Settings, load_settings
from .sources.bars import BAR_COLUMNS, fetch_minute_bars, parse_minute_bars
from .sources import news as news_src

ET = "America/New_York"
CRYPTO_CONTEXT = "BTC/USD"
EPS = 1e-4
DELTAS = [0.02, 0.03, 0.05, 0.08]     # how far from 50% the model must be to make a call
TRAIN_WEEKS = 6
MIN_CALLS = 40
TRAIN_CAP = 150_000                    # max rows per model fit (keeps a laptop run to minutes)
MODEL_PATH = ROOT / "data" / "intraday_models.joblib"

FEATURES = [
    "r1", "r3", "r5", "r15", "r30", "r60", "rv30", "vol5_ratio", "range_pos30", "vwap60_dist",
    "hl5", "rsi14", "minute_et", "dow", "mins_open", "is_crypto", "is_commodity",
    "mkt_r1", "mkt_r5", "mkt_r15", "rel_r5", "qqq_r5",
    "news_n30", "news_sent30", "news_n240", "mins_since_news",
]

BARS_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1m (
    ticker TEXT NOT NULL,
    ts     TEXT NOT NULL,             -- minute start, UTC ISO
    open REAL, high REAL, low REAL, close REAL, volume REAL, vwap REAL, trades REAL,
    PRIMARY KEY (ticker, ts)
);
CREATE TABLE IF NOT EXISTS intraday_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, summary TEXT
);
"""


# ---------------------------------------------------------------- settings

@dataclass
class Trench:
    multiplier: float = 25
    fee_bps: float | None = None
    entry_delay: int = 1
    holds: list = field(default_factory=lambda: [1, 2, 3, 5])
    history_days: int = 120
    feed: str = "iex"
    kinds: dict = field(default_factory=dict)        # ticker -> stock | crypto | commodity
    labels: dict = field(default_factory=dict)       # ticker -> friendly name (GLD -> gold)
    context: list = field(default_factory=lambda: ["SPY", "QQQ"])

    @property
    def tickers(self) -> list[str]:
        return list(self.kinds)

    @property
    def stock_like(self) -> list[str]:
        t = [k for k, v in self.kinds.items() if v != "crypto"]
        return t + [c for c in self.context if c not in t]

    @property
    def crypto(self) -> list[str]:
        t = [k for k, v in self.kinds.items() if v == "crypto"]
        return t + ([CRYPTO_CONTEXT] if CRYPTO_CONTEXT not in t else [])


def trench_settings(s: Settings) -> Trench:
    raw = s.trench or {}
    m = raw.get("markets") or {}
    kinds, labels = {}, {}
    for t in m.get("stocks") or []:
        kinds[str(t).upper()] = "stock"
    for t in m.get("crypto") or []:
        kinds[str(t).upper()] = "crypto"
    com = m.get("commodities") or {}
    for t, name in (com.items() if isinstance(com, dict) else ((c, c) for c in com)):
        kinds[str(t).upper()] = "commodity"
        labels[str(t).upper()] = str(name)
    fee = raw.get("fee_bps")
    return Trench(
        multiplier=float(raw.get("multiplier", 25)), fee_bps=None if fee in (None, "") else float(fee),
        entry_delay=int(raw.get("entry_delay_min", 1)), holds=[int(h) for h in raw.get("holds", [1, 2, 3, 5])],
        history_days=int(raw.get("history_days", 120)), feed=str(raw.get("feed", "iex")),
        kinds=kinds, labels=labels, context=[str(c).upper() for c in raw.get("context", ["SPY", "QQQ"])],
    )


ALIASES = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "BITCOIN": "BTC/USD",
           "ETHEREUM": "ETH/USD", "GOLD": "GLD", "SILVER": "SLV", "OIL": "USO"}


def resolve_ticker(text: str, t: Trench) -> str | None:
    x = text.upper().strip().replace("-", "/")
    x = ALIASES.get(x, x)
    if x in t.kinds:
        return x
    if f"{x}/USD" in t.kinds:
        return f"{x}/USD"
    for k, name in t.labels.items():
        if name.upper() == x:
            return k
    return None


# ---------------------------------------------------------------- storage

def upsert_bars(conn, df: pd.DataFrame) -> int:
    conn.executescript(BARS_SCHEMA)
    if df is None or df.empty:
        return 0
    d = df.copy()
    d["ts"] = d["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    before = conn.total_changes
    conn.executemany(
        f"INSERT OR REPLACE INTO bars_1m ({', '.join(BAR_COLUMNS)}) VALUES ({', '.join('?' * len(BAR_COLUMNS))})",
        d[BAR_COLUMNS].itertuples(index=False, name=None))
    conn.commit()
    return conn.total_changes - before


def load_bars(conn, tickers: list[str], since: pd.Timestamp | None = None) -> pd.DataFrame:
    conn.executescript(BARS_SCHEMA)
    q = f"SELECT * FROM bars_1m WHERE ticker IN ({', '.join('?' * len(tickers))})"
    args = list(tickers)
    if since is not None:
        q += " AND ts >= ?"
        args.append(iso_utc(since))
    df = pd.read_sql(q, conn, params=args)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


def load_intraday_news(conn, tickers: list[str], since: pd.Timestamp | None = None) -> pd.DataFrame:
    from .sentiment import ensure_table
    ensure_table(conn)
    q = f"""SELECT n.ticker, n.published_at, s.score FROM news n
            LEFT JOIN news_sentiment s ON s.story_key = n.story_key
            WHERE n.ticker IN ({', '.join('?' * len(tickers))})"""
    args = list(tickers)
    if since is not None:
        q += " AND n.published_at >= ?"
        args.append(iso_utc(since))
    df = pd.read_sql(q, conn, params=args)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


# ---------------------------------------------------------------- refresh

def news_symbol(t: str) -> str:
    return t.replace("/", "")           # Alpaca news tags crypto as BTCUSD


def refresh(conn, s: Settings, t: Trench, fetch=fetch_minute_bars, fetch_news=news_src.fetch_alpaca_news,
            progress=print) -> dict:
    if not s.has_alpaca:
        raise SystemExit("Trench mode needs ALPACA_API_KEY and ALPACA_SECRET_KEY in .env.")
    conn.executescript(BARS_SCHEMA)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    floor = now - pd.Timedelta(days=t.history_days)
    out = {}
    for group in (t.stock_like, t.crypto):
        lasts = [conn.execute("SELECT MAX(ts) FROM bars_1m WHERE ticker = ?", (x,)).fetchone()[0] for x in group]
        start = min((pd.Timestamp(x) - pd.Timedelta(minutes=10)) if x else floor for x in lasts)
        start = max(start, floor)
        if progress:
            progress(f"  minute bars for {', '.join(group)} since {start:%Y-%m-%d %H:%M} UTC")
        raw = fetch(group, iso_utc(start), iso_utc(now), s.alpaca_api_key, s.alpaca_secret_key,
                    feed=t.feed, progress=progress)
        out[group[0]] = upsert_bars(conn, parse_minute_bars(raw))
    # News for every Trench market (the daily pipeline only covers its own watchlist)
    last = conn.execute("SELECT MAX(published_at) FROM news WHERE provider = 'alpaca-intraday'").fetchone()[0]
    start = max(pd.Timestamp(last) - pd.Timedelta(hours=1) if last else floor, floor)
    days = db.trading_days(conn)
    sym_map = {news_symbol(x): x for x in t.tickers}
    saved = 0

    def save(batch):
        nonlocal saved
        rows = []
        for r in news_src.parse_alpaca_news(batch, list(sym_map)):
            r = {**r, "ticker": sym_map[r["ticker"]], "provider": "alpaca-intraday",
                 "story_key": story_key(r["headline"])}
            r["trading_day"] = assign_trading_day(r["published_at"], days)
            rows.append(r)
        saved += db.insert_news(conn, rows)

    if progress:
        progress(f"  news for {len(sym_map)} markets since {start:%Y-%m-%d}")
    fetch_news(list(sym_map), iso_utc(start), iso_utc(now), s.alpaca_api_key, s.alpaca_secret_key,
               on_batch=save, progress=progress)
    out["news"] = saved
    return out


# ---------------------------------------------------------------- features

def grid(b: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One ticker's bars on a complete minute grid. Stocks/ETFs: regular hours only."""
    b = b.set_index("ts").sort_index()
    if b.empty:
        return b
    if kind == "crypto":
        idx = pd.date_range(b.index.min(), b.index.max(), freq="min", tz="UTC")
        g = b.reindex(idx)
        g["session"] = 0
    else:
        local = b.index.tz_convert(ET)
        days = sorted(set(local.date))
        parts = [pd.date_range(pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=9, minutes=30),
                               periods=390, freq="min").tz_convert("UTC") for d in days]
        idx = parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]
        g = b.reindex(idx)
        g["session"] = g.index.tz_convert(ET).strftime("%Y%m%d").astype(int)
    g["real"] = g["close"].notna()
    g["close"] = g.groupby("session")["close"].ffill(limit=30)
    for c in ("open", "high", "low"):
        g[c] = g[c].fillna(g["close"])
    g["volume"] = g["volume"].fillna(0.0)
    g["vwap"] = g["vwap"].fillna(g["close"])
    g.index.name = "ts"
    return g


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def price_features(g: pd.DataFrame, kind: str) -> pd.DataFrame:
    c, v = g["close"], g["volume"]
    f = pd.DataFrame(index=g.index)
    for k in (1, 3, 5, 15, 30, 60):
        f[f"r{k}"] = c.pct_change(k, fill_method=None)
    f["rv30"] = f["r1"].rolling(30, min_periods=10).std()
    f["vol5_ratio"] = v.rolling(5).sum() / (v.rolling(60, min_periods=20).mean() * 5).replace(0, np.nan)
    lo30, hi30 = g["low"].rolling(30, min_periods=10).min(), g["high"].rolling(30, min_periods=10).max()
    f["range_pos30"] = (c - lo30) / (hi30 - lo30).replace(0, np.nan)
    pv = (g["vwap"] * v).rolling(60, min_periods=10).sum()
    f["vwap60_dist"] = c / (pv / v.rolling(60, min_periods=10).sum().replace(0, np.nan)) - 1
    f["hl5"] = (g["high"].rolling(5).max() - g["low"].rolling(5).min()) / c
    f["rsi14"] = rsi(c)
    local = g.index.tz_convert(ET)
    f["minute_et"] = local.hour * 60 + local.minute
    f["dow"] = local.dayofweek
    f["mins_open"] = np.nan if kind == "crypto" else (f["minute_et"] - 570)
    f["is_crypto"] = float(kind == "crypto")
    f["is_commodity"] = float(kind == "commodity")
    return f


def labels(g: pd.DataFrame, holds: list[int], delay: int) -> pd.DataFrame:
    """Forward return over each hold, entering `delay` minutes after the alert,
    plus the worst move against a long and a short during the hold (for bust checks)."""
    c, sess = g["close"], g["session"]
    entry = c.shift(-delay)
    out = pd.DataFrame(index=g.index)
    for h in holds:
        same = sess.shift(-(delay + h)) == sess
        fwd = c.shift(-(delay + h)) / entry - 1
        out[f"fwd{h}"] = fwd.where(same)
        out[f"mae_long{h}"] = (g["low"].rolling(h).min().shift(-(delay + h)) / entry - 1).where(same)
        out[f"mae_short{h}"] = (g["high"].rolling(h).max().shift(-(delay + h)) / entry - 1).where(same)
    return out


def context_returns(bars: pd.DataFrame, ticker: str, kind: str) -> pd.DataFrame:
    b = bars[bars["ticker"] == ticker]
    if b.empty:
        return pd.DataFrame()
    g = grid(b, kind)
    c = g["close"]
    return pd.DataFrame({k: c.pct_change(n, fill_method=None) for k, n in (("r1", 1), ("r5", 5), ("r15", 15))},
                        index=g.index)


def news_features(index: pd.DatetimeIndex, news: pd.DataFrame) -> pd.DataFrame:
    """Counts and tone of headlines published up to each minute's close."""
    out = pd.DataFrame(index=index)
    if news.empty:
        out["news_n30"] = 0.0
        out["news_sent30"] = 0.0
        out["news_n240"] = 0.0
        out["mins_since_news"] = 240.0
        return out
    MIN = 60_000_000_000                                       # one minute in nanoseconds
    n = news.sort_values("published_at")
    known = n["published_at"].dt.floor("min").dt.as_unit("ns").astype("int64").to_numpy()  # known by that minute's close
    score = n["score"].fillna(0.0).to_numpy(dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(score)])
    t = index.as_unit("ns").asi8
    hi = np.searchsorted(known, t, side="right")
    lo30 = np.searchsorted(known, t - 30 * MIN, side="right")
    lo240 = np.searchsorted(known, t - 240 * MIN, side="right")
    n30 = hi - lo30
    out["news_n30"] = n30.astype(float)
    out["news_sent30"] = np.where(n30 > 0, (cs[hi] - cs[lo30]) / np.maximum(n30, 1), 0.0)
    out["news_n240"] = (hi - lo240).astype(float)
    mins = np.where(hi > 0, (t - known[np.maximum(hi - 1, 0)]) / MIN, 240.0)
    out["mins_since_news"] = np.clip(mins, 0, 240)
    return out


def build_frame(bars: pd.DataFrame, news: pd.DataFrame, t: Trench, tickers: list[str] | None = None,
                with_labels: bool = True) -> pd.DataFrame:
    spy = context_returns(bars, "SPY", "stock")
    qqq = context_returns(bars, "QQQ", "stock")
    btc = context_returns(bars, CRYPTO_CONTEXT, "crypto")
    parts = []
    for tk in tickers or t.tickers:
        kind = t.kinds.get(tk, "stock")
        b = bars[bars["ticker"] == tk]
        if b.empty:
            continue
        g = grid(b, kind)
        f = price_features(g, kind)
        mkt = btc if kind == "crypto" else spy
        if not mkt.empty:
            m = mkt.reindex(g.index)
            f["mkt_r1"], f["mkt_r5"], f["mkt_r15"] = m["r1"], m["r5"], m["r15"]
        else:
            f["mkt_r1"] = f["mkt_r5"] = f["mkt_r15"] = np.nan
        f["rel_r5"] = f["r5"] - f["mkt_r5"]
        f["qqq_r5"] = qqq["r5"].reindex(g.index) if (not qqq.empty and kind != "crypto") else np.nan
        nf = news_features(g.index, news[news["ticker"] == tk] if not news.empty else news)
        f = f.join(nf)
        if with_labels:
            f = f.join(labels(g, t.holds, t.entry_delay))
        f["ticker"], f["kind"], f["real"], f["close"] = tk, kind, g["real"], g["close"]
        f["day"] = g.index.tz_convert(ET).strftime("%Y-%m-%d")
        parts.append(f.reset_index())
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df


# ---------------------------------------------------------------- models

def make_model(seed: int = 0):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=400, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


@dataclass
class HoldModel:
    hold: int
    model: object
    a: float = 1.0
    b: float = 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.model.predict_proba(X[FEATURES])[:, 1], EPS, 1 - EPS)
        z = self.a * np.log(p / (1 - p)) + self.b
        return 1 / (1 + np.exp(-z))


def fit_hold(train: pd.DataFrame, cal: pd.DataFrame, hold: int, cap: int | None = None, seed: int = 0) -> HoldModel:
    cap = cap or TRAIN_CAP
    y = f"fwd{hold}"
    tr = train[train[y].notna() & train["real"]]
    if len(tr) > cap:
        tr = tr.sample(cap, random_state=seed)
    m = make_model(seed).fit(tr[FEATURES], (tr[y] > 0).astype(int))
    hm = HoldModel(hold, m)
    ca = cal[cal[y].notna() & cal["real"]]
    if len(ca) >= 500:
        p = np.clip(m.predict_proba(ca[FEATURES])[:, 1], EPS, 1 - EPS)
        pl = LogisticRegression(C=1.0).fit(np.log(p / (1 - p)).reshape(-1, 1), (ca[y] > 0).astype(int))
        hm.a, hm.b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
    return hm


# ---------------------------------------------------------------- trade math

def trade_stats(p, fwd, mae_long, mae_short, days, delta: float, mult: float, fee_bps: float | None,
                seed: int = 0) -> dict:
    """What would have happened trading every call at this confidence level."""
    p, fwd = np.asarray(p, float), np.asarray(fwd, float)
    call = np.abs(p - 0.5) >= delta
    n = int(call.sum())
    if n == 0:
        return {"n": 0}
    d = np.where(p[call] > 0.5, 1.0, -1.0)
    gross = d * fwd[call] * 1e4                                   # bps of position in your favor
    adverse = np.where(d > 0, -np.asarray(mae_long, float)[call], np.asarray(mae_short, float)[call])
    bust = adverse >= 1.0 / mult
    fee = fee_bps or 0.0
    stake = np.where(bust, -1.0, np.maximum(mult * (gross - fee) / 1e4, -1.0))
    dd = np.asarray(days)[call]
    g = pd.Series(gross).groupby(pd.Series(dd)).agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(g), size=(1000, len(g)))
    means = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    nb = ~bust
    max_fee = ((gross[nb].sum() / 1e4 * mult - bust.sum()) / (nb.sum() * mult) * 1e4) if nb.any() else float("-inf")
    return {
        "n": n, "per_day": round(n / max(len(set(days)), 1), 2), "delta": delta,
        "hit": float((gross > 0).mean()), "bps": float(gross.mean()),
        "bps_lo": float(np.percentile(means, 5)), "bps_hi": float(np.percentile(means, 95)),
        "bust_rate": float(bust.mean()), "stake_ret": float(stake.mean()), "max_fee_bps": float(max_fee),
    }


# ---------------------------------------------------------------- backtest

def weeks(df: pd.DataFrame) -> list[pd.Timestamp]:
    w = pd.to_datetime(df["day"]).dt.to_period("W-SUN").dt.start_time
    return sorted(w.unique())


def walk_forward(df: pd.DataFrame, t: Trench, progress=print, seed: int = 0) -> pd.DataFrame:
    week = pd.to_datetime(df["day"]).dt.to_period("W-SUN").dt.start_time
    wk = sorted(week.unique())
    if len(wk) <= TRAIN_WEEKS:
        raise ValueError(f"Need more than {TRAIN_WEEKS} weeks of minute data; increase trench.history_days.")
    parts, t0 = [], time.time()
    tests = wk[TRAIN_WEEKS:]
    for k, w in enumerate(tests, 1):
        test = df[week == w]
        hist = df[(week >= w - pd.Timedelta(weeks=TRAIN_WEEKS)) & (week < w)]
        hist_days = sorted(hist["day"].unique())
        if len(hist_days) < 6:
            continue
        cal = hist[hist["day"].isin(hist_days[-2:])]
        train = hist[hist["day"].isin(hist_days[:-3])]        # one-day gap before calibration
        for h in t.holds:
            hm = fit_hold(train, cal, h, seed=seed)
            te = test[test[f"fwd{h}"].notna() & test["real"]]
            if te.empty:
                continue
            parts.append(pd.DataFrame({
                "ts": te["ts"].to_numpy(), "day": te["day"].to_numpy(), "ticker": te["ticker"].to_numpy(),
                "kind": te["kind"].to_numpy(), "hold": h, "p": hm.predict(te),
                "fwd": te[f"fwd{h}"].to_numpy(), "mae_long": te[f"mae_long{h}"].to_numpy(),
                "mae_short": te[f"mae_short{h}"].to_numpy(),
            }))
        if progress:
            left = (time.time() - t0) / k * (len(tests) - k)
            progress(f"  week {k}/{len(tests)} ({w:%Y-%m-%d}) done, ~{left / 60:.1f} min left")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plan_from_oos(oos: pd.DataFrame, t: Trench, seed: int = 0) -> dict:
    """Per market type and hold: the confidence bar to use and whether it's tradeable.

    Picking the best of several bars flatters the result, so the bar is CHOSEN on the
    first half of the test days and JUDGED on the second half, which played no part
    in choosing it. The verdict and the numbers shown come from that second half.
    """
    plan = {}
    bar = t.fee_bps or 0.0
    all_days = sorted(oos["day"].unique())
    cut = all_days[len(all_days) // 2]
    for (kind, h), g in oos.groupby(["kind", "hold"]):
        pick, judge = g[g["day"] < cut], g[g["day"] >= cut]
        y = (judge["fwd"] > 0).astype(int)
        auc = float(roc_auc_score(y, judge["p"])) if y.nunique() == 2 else float("nan")

        def stats(part, dl):
            return trade_stats(part["p"], part["fwd"], part["mae_long"], part["mae_short"], part["day"],
                               dl, t.multiplier, t.fee_bps, seed)

        options = [st for dl in DELTAS if (st := stats(pick, dl))["n"] >= MIN_CALLS // 2]
        chosen = max(options, key=lambda r: r["bps_lo"])["delta"] if options else None
        best = stats(judge, chosen) if chosen is not None else None
        if best is not None and best["n"] < MIN_CALLS // 2:
            best = None
        tradeable = bool(best and auc > 0.505 and best["bps_lo"] > bar and best["stake_ret"] > 0)
        plan[f"{kind}:{h}"] = {"kind": kind, "hold": int(h), "auc": auc, "rows": int(len(judge)),
                               "best": best, "chosen_on": f"before {cut}", "judged_on": f"{cut} onward",
                               "tradeable": tradeable}
    return plan


# ---------------------------------------------------------------- report

def format_plan(plan: dict, t: Trench, start: str, end: str) -> str:
    fee = f"{t.fee_bps:.1f} bps" if t.fee_bps is not None else "not set (fill trench.fee_bps)"
    cut = next(iter(plan.values()))["judged_on"].split()[0] if plan else start
    L = [f"Trench backtest {start} to {end} · {t.multiplier:g}x · fee {fee} · "
         f"entry {t.entry_delay} min after the alert",
         f"Every prediction is out-of-sample. Confidence bars were picked on the days before {cut};",
         f"the numbers below are from {cut} onward, which played no part in picking them.", ""]
    for kind in ("stock", "crypto", "commodity"):
        rows = [v for v in plan.values() if v["kind"] == kind]
        if not rows:
            continue
        L.append(f"[{kind}]")
        L.append(f"  {'hold':<6}{'AUC':>6}{'bar':>6}{'calls/day':>10}{'right':>7}{'avg move':>10}"
                 f"{'90% range':>18}{'busts':>7}{'per trade':>11}{'max fee':>9}  verdict")
        for v in sorted(rows, key=lambda r: r["hold"]):
            b = v["best"]
            if not b:
                L.append(f"  {v['hold']:<3}min{v['auc']:>6.3f}   too few confident calls")
                continue
            verdict = "TRADE" if v["tradeable"] else "no trade"
            L.append(f"  {v['hold']:<3}min{v['auc']:>6.3f}{b['delta'] * 100:>5.0f}%{b['per_day']:>10.1f}"
                     f"{b['hit']:>7.0%}{b['bps']:>+8.2f}bp{b['bps_lo']:>+8.2f} to {b['bps_hi']:>+5.2f}bp"
                     f"{b['bust_rate']:>7.1%}{b['stake_ret']:>+10.1%}{b['max_fee_bps']:>+7.2f}bp  {verdict}")
        L.append("")
    L += [
        "How to read it:",
        "  bar        how far from 50% the model must be to call (e.g. 5% = 55%+ up or 45%- down)",
        "  avg move   average price move in your favor per call, in bps (1 bp = 0.01%) before fees",
        "  90% range  where that average lands in 90% of day-level resamples; below 0 = could be luck",
        f"  busts      share of calls where price moved {100 / t.multiplier:.1f}% against you mid-hold "
        f"(a bust at {t.multiplier:g}x)",
        f"  per trade  average result on your stake at {t.multiplier:g}x, after fees and busts",
        "  max fee    the highest Trench fee (bps of position) these calls could pay and still break even",
        "  TRADE      AUC above 0.505, the low end of the range beats the fee, and per-trade result > 0",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------- train + save

def train_live(df: pd.DataFrame, t: Trench, seed: int = 0) -> dict:
    days = sorted(df["day"].unique())
    recent = df[df["day"].isin(days[-TRAIN_WEEKS * 5:])]
    rd = sorted(recent["day"].unique())
    cal = recent[recent["day"].isin(rd[-2:])]
    train = recent[recent["day"].isin(rd[:-3])]
    return {h: fit_hold(train, cal, h, seed=seed) for h in t.holds}


def feature_scales(df: pd.DataFrame) -> dict:
    sub = df[df["real"]]
    return {c: (float(sub[c].mean()), float(sub[c].std() or 1.0)) for c in FEATURES}


def save_models(models: dict, plan: dict, scales: dict, t: Trench) -> None:
    import joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "plan": plan, "scales": scales, "created_at": db.now_utc(),
                 "multiplier": t.multiplier, "fee_bps": t.fee_bps}, MODEL_PATH)


def load_models():
    import joblib
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------- live calls

@dataclass
class Call:
    ticker: str
    kind: str
    direction: str            # UP | DOWN | NONE
    hold: int | None
    p: float | None
    tested: dict | None
    reasons: list
    note: str = ""
    price: float | None = None
    asof: pd.Timestamp | None = None


def market_open(kind: str, now: pd.Timestamp | None = None) -> bool:
    if kind == "crypto":
        return True
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert(ET)
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and 570 <= m < 960


def reasons_for(row: pd.Series, scales: dict, news_row: dict | None) -> list[str]:
    out = []
    picks = [("r5", "last 5 min", "{:+.2%}"), ("r15", "last 15 min", "{:+.2%}"),
             ("vol5_ratio", "volume vs normal", "{:.1f}x"), ("vwap60_dist", "vs 1-hr VWAP", "{:+.2%}"),
             ("rel_r5", "vs market (5 min)", "{:+.2%}"), ("rsi14", "minute RSI", "{:.0f}")]
    scored = []
    for col, label, fmt in picks:
        v = row.get(col)
        if v is None or pd.isna(v):
            continue
        mu, sd = scales.get(col, (0.0, 1.0))
        scored.append((abs((v - mu) / (sd or 1.0)), f"{label} {fmt.format(v)}"))
    out = [txt for _, txt in sorted(scored, reverse=True)[:3]]
    if news_row:
        out.append(f"news: \"{news_row['headline'][:90]}\" ({news_row['mins']:.0f} min ago, "
                   f"tone {news_row['score']:+.2f})" if news_row.get("score") is not None
                   else f"news: \"{news_row['headline'][:90]}\" ({news_row['mins']:.0f} min ago)")
    return out


def latest_headline(news: pd.DataFrame, now: pd.Timestamp) -> dict | None:
    if news.empty or "headline" not in news:
        return None
    n = news[news["published_at"] <= now].sort_values("published_at")
    if n.empty:
        return None
    r = n.iloc[-1]
    mins = (now - r["published_at"]).total_seconds() / 60
    if mins > 240:
        return None
    return {"headline": r["headline"], "mins": mins, "score": None if pd.isna(r.get("score")) else r["score"]}


def decide(row: pd.Series, art: dict, kind: str) -> tuple[str, int | None, float | None, dict | None]:
    best = None
    for h, hm in art["models"].items():
        pl = art["plan"].get(f"{kind}:{h}")
        if not pl or not pl["tradeable"] or not pl["best"]:
            continue
        p = float(hm.predict(row.to_frame().T)[0])
        if abs(p - 0.5) >= pl["best"]["delta"]:
            score = pl["best"]["stake_ret"]
            if best is None or score > best[3]["stake_ret"]:
                best = ("UP" if p > 0.5 else "DOWN", int(h), p, pl["best"])
    return best if best else ("NONE", None, None, None)


def live_call(ticker: str, s: Settings, t: Trench, art: dict, fetch=fetch_minute_bars,
              fetch_news=news_src.fetch_alpaca_news, scorer=None, now: pd.Timestamp | None = None) -> Call:
    kind = t.kinds[ticker]
    now = (now or pd.Timestamp.now(tz="UTC")).floor("min")
    if not market_open(kind, now):
        return Call(ticker, kind, "NONE", None, None, None, [], note="US market is closed for this one.")
    ctx = [CRYPTO_CONTEXT] if kind == "crypto" else ["SPY", "QQQ"]
    syms = [ticker] + [c for c in ctx if c != ticker]
    start = now - pd.Timedelta(hours=8 if kind != "crypto" else 3)
    raw = fetch(syms, iso_utc(start), iso_utc(now), s.alpaca_api_key, s.alpaca_secret_key, feed=t.feed)
    bars = parse_minute_bars(raw)
    items = fetch_news([news_symbol(ticker)], iso_utc(now - pd.Timedelta(hours=4)), iso_utc(now),
                       s.alpaca_api_key, s.alpaca_secret_key, progress=None)
    rows = news_src.parse_alpaca_news(items, [news_symbol(ticker)])
    news = pd.DataFrame(rows)
    if not news.empty:
        news["ticker"] = ticker
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
        news["score"] = np.nan
        if scorer is not None:
            res = scorer(news["headline"].tolist())
            news["score"] = [r.get("positive", 0) - r.get("negative", 0) for r in res]
    frame = build_frame(bars, news if not news.empty else pd.DataFrame(columns=["ticker", "published_at", "score"]),
                        t, [ticker], with_labels=False)
    if frame.empty:
        return Call(ticker, kind, "NONE", None, None, None, [], note="No recent price data.")
    frame = frame[frame["real"]]
    row = frame.iloc[-1]
    age = (now - row["ts"]).total_seconds() / 60
    if age > 3:
        return Call(ticker, kind, "NONE", None, None, None, [], note=f"Latest price is {age:.0f} min old.")
    direction, hold, p, tested = decide(row[FEATURES].astype(float), art, kind)
    return Call(ticker, kind, direction, hold, p, tested,
                reasons_for(row, art["scales"], latest_headline(news, now)),
                price=float(row["close"]), asof=row["ts"])


def format_call(c: Call, t: Trench, html: bool = False) -> str:
    b = (lambda x: f"<b>{x}</b>") if html else (lambda x: x)
    name = f"{c.ticker} ({t.labels[c.ticker]})" if c.ticker in t.labels else c.ticker
    if c.direction == "NONE":
        head = f"{b(name)} ⚪ No trade"
        lines = [head, f"   {c.note}" if c.note else "   No tested setup right now."]
    else:
        arrow = "⬆️ UP" if c.direction == "UP" else "⬇️ DOWN"
        conf = c.p if c.direction == "UP" else 1 - c.p
        tb = c.tested
        lines = [
            f"{b(name)} {arrow} · hold {c.hold} min",
            f"   {conf:.0%} confident · enter now, close after {c.hold} min",
            f"   tested: right {tb['hit']:.0%}, avg {tb['bps']:+.1f} bps, "
            f"{tb['bust_rate']:.1%} busts at {t.multiplier:g}x, {tb['per_day']:.1f} calls/day",
        ]
    for r in c.reasons:
        lines.append(f"   • {r}")
    if c.price is not None and c.asof is not None:
        lines.append(f"   price {c.price:,.2f} at {c.asof.tz_convert('America/Chicago'):%-I:%M %p} CT")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def _finbert_or_none():
    try:
        from .sentiment import finbert_scorer
        return finbert_scorer()
    except BaseException:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Trench mode: 1-5 minute calls.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refresh", help="pull minute bars and news")
    sub.add_parser("backtest", help="walk-forward test, then train and save live models")
    c = sub.add_parser("call", help="live call for one market")
    c.add_argument("ticker")
    sub.add_parser("scan", help="live calls for every market")
    args = ap.parse_args(argv)

    s = load_settings()
    t = trench_settings(s)
    conn = db.connect(s.database_path)

    if args.cmd == "refresh":
        print("Refreshing Trench markets...", flush=True)
        out = refresh(conn, s, t, progress=lambda m: print(m, flush=True))
        n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM bars_1m").fetchone()
        print(f"\nStored {n[0]:,} minute bars for {n[1]} symbols; {out['news']} new headlines.")
        print("Next: python -m signalbot.sentiment   then   python -m signalbot.intraday backtest")
        return 0

    if args.cmd == "backtest":
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=t.history_days)
        bars = load_bars(conn, list(dict.fromkeys(t.stock_like + t.crypto)), since)
        if bars.empty:
            print("No minute bars yet. Run `python -m signalbot.intraday refresh` first.")
            return 1
        print("Building minute features...", flush=True)
        df = build_frame(bars, load_intraday_news(conn, t.tickers, since), t)
        print(f"  {len(df):,} minute rows across {df['ticker'].nunique()} markets", flush=True)
        print(f"Walk-forward over {len(weeks(df)) - TRAIN_WEEKS} weeks x {len(t.holds)} hold times...", flush=True)
        oos = walk_forward(df, t, progress=lambda m: print(m, flush=True))
        plan = plan_from_oos(oos, t)
        print()
        print(format_plan(plan, t, oos["day"].min(), oos["day"].max()))
        print("\nTraining live models on the latest weeks...", flush=True)
        save_models(train_live(df, t), plan, feature_scales(df), t)
        conn.executescript(BARS_SCHEMA)
        conn.execute("INSERT INTO intraday_runs (created_at, summary) VALUES (?, ?)",
                     (db.now_utc(), json.dumps({"plan": plan, "start": oos["day"].min(), "end": oos["day"].max(),
                                                "multiplier": t.multiplier, "fee_bps": t.fee_bps}, default=float)))
        conn.commit()
        print("Saved. Use /call NVDA or /scan in Telegram.")
        return 0

    art = load_models()
    if art is None:
        print("No trained models yet. Run `python -m signalbot.intraday backtest` first.")
        return 1
    scorer = _finbert_or_none()
    targets = t.tickers if args.cmd == "scan" else [resolve_ticker(args.ticker, t)]
    if targets == [None]:
        print(f"{args.ticker} isn't one of your Trench markets: {', '.join(t.tickers)}")
        return 1
    for tk in targets:
        try:
            print(format_call(live_call(tk, s, t, art, scorer=scorer), t))
        except Exception as exc:
            print(f"{tk}: couldn't make a call ({exc})")
        print()
    print("Not financial advice. Calls are only as good as the backtest above them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
