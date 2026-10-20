"""Phase 2 tests. The most important one: changing the future must not change the past."""

import numpy as np
import pandas as pd
import pytest

from signalbot import db
from signalbot.config import Settings
from signalbot.features import (
    CONTEXT, FEATURES, LABEL, NEWS, TECH, add_earnings, build_features, load_news, news_daily, rsi, technicals,
)
from signalbot.sentiment import score_stories

TICKERS = ["AAA", "BBB"]
CONTEXT_TICKERS = ["SPY", "QQQ", "^VIX", "XLK", "^TNX", "^IRX"]


def synthetic_prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-03-02", periods=n)
    rows = []
    for t in TICKERS + CONTEXT_TICKERS:
        level = {"^VIX": 18.0, "^TNX": 4.2, "^IRX": 3.9}.get(t, 100.0)
        px = level * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
        for d, c in zip(days, px):
            rows.append({"ticker": t, "date": d, "open": c * 0.998, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "adj_close": c, "volume": float(rng.integers(1e6, 5e6))})
    return pd.DataFrame(rows)


def settings():
    return Settings(watchlist=TICKERS, sectors={"AAA": "XLK"})


def empty_news():
    return pd.DataFrame(columns=["ticker", "trading_day", "n_tickers", "score", "pos", "neg", "date"])


def empty_earnings():
    return pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                         "hour": pd.Series(dtype=str)})


# ------------------------------------------------------------ indicators

def test_rsi_extremes_and_range():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert rsi(down).iloc[-1] == pytest.approx(0.0)
    wiggly = pd.Series(100 + np.sin(np.arange(200)) * 3)
    r = rsi(wiggly).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert rsi(up).iloc[:13].isna().all()


def test_label_is_next_day_direction():
    p = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=4), "open": 1.0, "high": 1.0, "low": 1.0,
                      "close": [10, 11, 10.5, 12.0], "adj_close": [10, 11, 10.5, 12.0], "volume": 1.0})
    t = technicals(p)
    assert t[LABEL].tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(t[LABEL].iloc[-1])                      # tomorrow unknown
    assert t["next_ret"].iloc[0] == pytest.approx(0.1)


# ------------------------------------------------------------ no lookahead

def test_changing_future_prices_never_changes_past_features():
    prices = synthetic_prices()
    base = build_features(prices, empty_news(), empty_earnings(), settings())

    cut = prices["date"].sort_values().unique()[80]
    shocked = prices.copy()
    future = shocked["date"] > cut
    for c in ["open", "high", "low", "close", "adj_close"]:
        shocked.loc[future, c] *= 3.0                       # wild future move
    shocked.loc[future, "volume"] *= 10
    after = build_features(shocked, empty_news(), empty_earnings(), settings())

    cols = TECH + CONTEXT
    past_a = base[base["date"] <= cut].set_index(["ticker", "date"])[cols]
    past_b = after[after["date"] <= cut].set_index(["ticker", "date"])[cols]
    pd.testing.assert_frame_equal(past_a, past_b)

    # Only the label of the cut day itself is allowed to change (it IS about tomorrow).
    lab_a = base[base["date"] < cut].set_index(["ticker", "date"])[LABEL]
    lab_b = after[after["date"] < cut].set_index(["ticker", "date"])[LABEL]
    pd.testing.assert_series_equal(lab_a, lab_b)


def test_feature_table_shape_and_context():
    feat = build_features(synthetic_prices(), empty_news(), empty_earnings(), settings())
    assert set(feat["ticker"]) == set(TICKERS)
    assert set(FEATURES).issubset(feat.columns)
    last = feat[feat["date"] == feat["date"].max()]
    assert last[LABEL].isna().all()                         # newest day is the one to predict
    ready = feat.dropna(subset=TECH + CONTEXT + [LABEL])
    assert len(ready) > 100
    # BBB has no sector entry, so it's compared to SPY
    row = feat[(feat["ticker"] == "BBB")].dropna(subset=["sector_ret_1d"]).iloc[-1]
    assert row["sector_ret_1d"] == pytest.approx(row["spy_ret_1d"])
    # No news history at all -> news columns unknown, not zero
    assert feat[NEWS].isna().all().all()


# ------------------------------------------------------------ news

def test_news_features_use_trading_day_and_sentiment(tmp_path):
    prices = synthetic_prices()
    days = sorted(prices["date"].unique())
    d1, d2 = days[60].strftime("%Y-%m-%d"), days[61].strftime("%Y-%m-%d")

    conn = db.connect(tmp_path / "n.db")
    db.insert_news(conn, [
        {"ticker": "AAA", "story_key": "k1", "published_at": "x", "trading_day": d1, "headline": "AAA soars",
         "n_tickers": 1, "provider": "t"},
        {"ticker": "AAA", "story_key": "k2", "published_at": "x", "trading_day": d1, "headline": "AAA sued",
         "n_tickers": 5, "provider": "t"},
        {"ticker": "AAA", "story_key": "k3", "published_at": "x", "trading_day": d2, "headline": "AAA flat",
         "n_tickers": 1, "provider": "t"},
    ])
    fake = {"AAA soars": 0.9, "AAA sued": -0.7, "AAA flat": 0.0}
    score_stories(conn, lambda hs: [{"positive": max(fake[h], 0), "negative": max(-fake[h], 0),
                                     "neutral": 1 - abs(fake[h])} for h in hs], progress=None)
    news = load_news(conn)
    conn.close()

    feat = build_features(prices, news, empty_earnings(), settings()).set_index(["ticker", "date"])
    r1 = feat.loc[("AAA", pd.Timestamp(d1))]
    assert r1["news_n"] == 2 and r1["news_focus_n"] == 1
    assert r1["news_sent"] == pytest.approx(0.1)
    assert r1["news_sent_min"] == pytest.approx(-0.7)
    assert r1["news_pos_share"] == pytest.approx(0.5)
    r2 = feat.loc[("AAA", pd.Timestamp(d2))]
    assert r2["news_sent_3d"] == pytest.approx((0.9 - 0.7 + 0.0) / 3)
    # Quiet day after coverage starts = zero news, neutral sentiment
    r3 = feat.loc[("BBB", pd.Timestamp(d2))]
    assert r3["news_n"] == 0 and r3["news_sent"] == 0
    # Before coverage starts = unknown
    assert np.isnan(feat.loc[("AAA", pd.Timestamp(days[10]))]["news_n"])


def test_sentiment_scores_each_story_once(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    db.insert_news(conn, [
        {"ticker": t, "story_key": "same", "published_at": "x", "trading_day": "2026-01-05",
         "headline": "Chip stocks rally", "n_tickers": 2, "provider": "t"} for t in ("AAA", "BBB")
    ])
    calls = []

    def scorer(hs):
        calls.append(len(hs))
        return [{"positive": 0.8, "negative": 0.1, "neutral": 0.1} for _ in hs]

    assert score_stories(conn, scorer, progress=None) == 1
    assert score_stories(conn, scorer, progress=None) == 0   # cached
    assert calls == [1]
    assert conn.execute("SELECT score FROM news_sentiment").fetchone()[0] == pytest.approx(0.7)
    conn.close()


# ------------------------------------------------------------ earnings

def test_earnings_flags_point_at_the_reaction_day():
    days = pd.DatetimeIndex(pd.bdate_range("2026-10-19", periods=10))   # Mon Oct 19 ...
    feat = pd.DataFrame({"ticker": "AAA", "date": days})
    earnings = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": pd.to_datetime(["2026-10-21", "2026-10-29"]),   # Wed after close, Thu before open
        "hour": ["amc", "bmo"],
    })
    f = add_earnings(feat, earnings, days).set_index("date")
    # Wed report after close -> Thu's move reacts -> flag on Wed's row
    assert f.loc["2026-10-21", "earn_next"] == 1
    assert f.loc["2026-10-22", "earn_today"] == 1
    assert f.loc["2026-10-20", "earn_next"] == 0
    assert f.loc["2026-10-19", "days_to_earn"] == 3
    # Thu before-open report -> Thu reacts -> flag on Wed Oct 28's row
    assert f.loc["2026-10-28", "earn_next"] == 1
    assert f.loc["2026-10-29", "earn_today"] == 1
