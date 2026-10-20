"""Trench mode tests with synthetic minute data (no network)."""

import numpy as np
import pandas as pd
import pytest

import signalbot.intraday as ix
from signalbot.config import Settings
from signalbot.intraday import (
    FEATURES, Trench, build_frame, decide, format_call, format_plan, grid, labels, live_call, news_features,
    plan_from_oos, resolve_ticker, trade_stats, walk_forward,
)

T = Trench(multiplier=25, fee_bps=None, entry_delay=1, holds=[1, 5],
           kinds={"AAPL": "stock", "GLD": "commodity", "BTC/USD": "crypto"},
           labels={"GLD": "gold"}, context=["SPY", "QQQ"])


def synth_bars(n_days=55, momentum=0.0, seed=0, start="2026-06-01"):
    """Minute bars. momentum > 0 plants a real edge: each minute partly repeats the last 5."""
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.bdate_range(start, periods=n_days)
    for tk in ["AAPL", "GLD", "SPY", "QQQ"]:
        for d in days:
            ts = pd.date_range(pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30),
                               periods=390, freq="min").tz_convert("UTC")
            rows.append((tk, ts))
    crypto_ts = pd.date_range(pd.Timestamp(days[0]).tz_localize("UTC"), pd.Timestamp(days[-1]).tz_localize("UTC")
                              + pd.Timedelta(hours=23, minutes=59), freq="min")
    rows.append(("BTC/USD", crypto_ts))
    out = []
    for tk, ts in rows:
        n = len(ts)
        eps = rng.normal(0, 0.0008, n)
        r = np.zeros(n)
        for i in range(n):
            r[i] = eps[i] + momentum * r[max(i - 5, 0):i].mean() if i else eps[i]
        px = 100 * np.exp(np.cumsum(r))
        out.append(pd.DataFrame({"ticker": tk, "ts": ts, "open": px * (1 - 1e-4), "high": px * (1 + 3e-4),
                                 "low": px * (1 - 3e-4), "close": px, "volume": rng.integers(100, 1000, n).astype(float),
                                 "vwap": px, "trades": 10.0}))
    return pd.concat(out, ignore_index=True)


def no_news():
    return pd.DataFrame(columns=["ticker", "published_at", "score"])


@pytest.fixture(autouse=True)
def small_fits(monkeypatch):
    monkeypatch.setattr(ix, "TRAIN_CAP", 40_000)


def test_grid_keeps_regular_hours_and_fills_gaps():
    b = synth_bars(n_days=2)
    b = b[b["ticker"] == "AAPL"].drop(index=b[b["ticker"] == "AAPL"].index[10:15])   # 5 missing minutes
    g = grid(b, "stock")
    assert len(g) == 2 * 390
    assert g["close"].notna().all() and (~g["real"]).sum() == 5
    local = g.index.tz_convert("America/New_York")
    assert local.min().strftime("%H:%M") == "09:30" and local.max().strftime("%H:%M") == "15:59"


def test_labels_use_entry_delay_and_stay_in_session():
    b = synth_bars(n_days=2)
    g = grid(b[b["ticker"] == "AAPL"], "stock")
    lab = labels(g, [5], delay=1)
    c = g["close"].to_numpy()
    assert lab["fwd5"].iloc[100] == pytest.approx(c[106] / c[101] - 1)
    # The last 6 minutes of each day can't complete a 1+5 minute trade before the close
    assert lab["fwd5"].iloc[384:390].isna().all() and lab["fwd5"].iloc[383] == lab["fwd5"].iloc[383]
    assert lab["mae_long5"].iloc[100] == pytest.approx(g["low"].iloc[102:107].min() / c[101] - 1)


def test_news_counts_only_what_was_already_published():
    idx = pd.date_range("2026-06-01 14:00", periods=60, freq="min", tz="UTC")
    news = pd.DataFrame({"ticker": "AAPL", "score": [0.8, -0.4],
                         "published_at": pd.to_datetime(["2026-06-01 14:10:30", "2026-06-01 14:20:05"], utc=True)})
    nf = news_features(idx, news)
    assert nf.loc[idx[9], "news_n30"] == 0              # 14:09 bar closes before the 14:10:30 headline
    assert nf.loc[idx[10], "news_n30"] == 1             # 14:10 bar close (14:11) is after it
    assert nf.loc[idx[20], "news_sent30"] == pytest.approx(0.2)
    assert nf.loc[idx[25], "mins_since_news"] == 5
    assert nf.loc[idx[45], "news_n30"] == 1             # 14:45: the 14:10 headline has aged out, 14:20 remains
    assert nf.loc[idx[59], "news_n30"] == 0


def test_trade_stats_bust_and_fee_math():
    p = np.array([0.6, 0.6, 0.4, 0.5])
    fwd = np.array([0.001, -0.001, -0.002, 0.0])
    mae_l = np.array([-0.0005, -0.05, 0.0, 0.0])          # 2nd long dips 5%: busts at 25x (4%)
    mae_s = np.array([0.0, 0.0, 0.0005, 0.0])
    st = trade_stats(p, fwd, mae_l, mae_s, ["d1", "d1", "d2", "d2"], 0.05, 25, fee_bps=2.0)
    assert st["n"] == 3
    assert st["bust_rate"] == pytest.approx(1 / 3)
    assert st["bps"] == pytest.approx((10 - 10 + 20) / 3)
    # per-stake: +25*(10-2)/1e4 = +2%, bust = -100%, +25*(20-2)/1e4 = +4.5%
    assert st["stake_ret"] == pytest.approx((0.02 - 1 + 0.045) / 3)


def test_resolve_ticker_aliases():
    assert resolve_ticker("btc", T) == "BTC/USD"
    assert resolve_ticker("gold", T) == "GLD"
    assert resolve_ticker("aapl", T) == "AAPL"
    assert resolve_ticker("ZZZ", T) is None


def _backtest(momentum, seed=0):
    df = build_frame(synth_bars(momentum=momentum, seed=seed), no_news(), T)
    assert set(FEATURES).issubset(df.columns)
    oos = walk_forward(df, T, progress=None)
    return df, oos, plan_from_oos(oos, T)


def test_planted_momentum_is_found_and_noise_is_not():
    _, _, plan = _backtest(momentum=0.6)
    assert plan["stock:1"]["tradeable"] and plan["crypto:1"]["tradeable"]
    assert plan["stock:1"]["auc"] > 0.55
    _, _, noise = _backtest(momentum=0.0, seed=1)
    assert not any(v["tradeable"] for v in noise.values())
    assert "no trade" in format_plan(noise, T, "a", "b")


def test_live_call_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ix, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(ix, "OOS_PATH", tmp_path / "oos.joblib")
    df, oos, plan = _backtest(momentum=0.6)
    ix.save_models(ix.train_live(df, T), plan, ix.feature_scales(df), T)
    art = ix.load_models()

    # The chart draws from the saved out-of-sample results
    from signalbot.report import trench_chart
    ix.save_oos(oos)
    png = trench_chart(ix.load_oos(), plan, T, tmp_path / "c.png")
    assert png.exists() and png.stat().st_size > 20_000

    bars = synth_bars(n_days=1, momentum=0.6, seed=5, start="2026-09-01")
    now = pd.Timestamp("2026-09-01 15:00", tz="America/New_York").tz_convert("UTC")

    def fake_fetch(syms, start, end, key, secret, feed="iex", **kw):
        b = bars[bars["ticker"].isin(syms) & (bars["ts"] < pd.Timestamp(end))]
        return [(r.ticker, {"t": r.ts.isoformat(), "o": r.open, "h": r.high, "l": r.low, "c": r.close,
                            "v": r.volume, "vw": r.vwap, "n": r.trades}) for r in b.itertuples()]

    def fake_news(symbols, start, end, key, secret, **kw):
        return [{"headline": "Apple supplier raises guidance", "created_at": "2026-09-01T18:55:00Z",
                 "symbols": ["AAPL"]}]

    s = Settings(watchlist=["AAPL"], alpaca_api_key="k", alpaca_secret_key="s")
    c = live_call("AAPL", s, T, art, fetch=fake_fetch, fetch_news=fake_news,
                  scorer=lambda hs: [{"positive": 0.9, "negative": 0.05}] * len(hs), now=now)
    assert c.direction in ("UP", "DOWN", "NONE")
    text = format_call(c, T)
    assert "AAPL" in text and "supplier" in text
    # Market closed -> no trade, crypto still allowed
    sat = pd.Timestamp("2026-09-05 12:00", tz="UTC")
    assert live_call("AAPL", s, T, art, fetch=fake_fetch, fetch_news=fake_news, now=sat).note.startswith("US market")


def test_tested_text_reads_last_run(tmp_path):
    import json
    from signalbot import db
    from signalbot.bot import tested_text
    from signalbot.intraday import BARS_SCHEMA
    s = Settings(watchlist=["AAPL"], database_path=tmp_path / "t.db",
                 trench={"multiplier": 25, "markets": {"stocks": ["AAPL"]}})
    conn = db.connect(s.database_path)
    conn.executescript(BARS_SCHEMA)
    best = {"hit": 0.56, "bps": 1.4, "per_day": 12.0, "bust_rate": 0.0, "max_fee_bps": 1.3}
    plan = {"stock:3": {"kind": "stock", "hold": 3, "tradeable": True, "best": best},
            "stock:5": {"kind": "stock", "hold": 5, "tradeable": False, "best": None}}
    conn.execute("INSERT INTO intraday_runs (created_at, summary) VALUES (?, ?)",
                 ("x", json.dumps({"plan": plan, "start": "2026-07-01", "end": "2026-10-19",
                                   "multiplier": 25, "fee_bps": None})))
    conn.commit()
    conn.close()
    text = tested_text(s)
    assert "🟢 3 min" in text and "too few" in text and "fee_bps" in text
