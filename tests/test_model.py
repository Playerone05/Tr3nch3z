"""Phase 3 tests: the model finds a planted signal, finds nothing in noise, and the bot can read its output."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signalbot import db
from signalbot.bot import odds_text, today_text
from signalbot.config import Settings
from signalbot.features import FEATURES, LABEL, NEWS, save
from signalbot.model import (
    BASE_FEATURES, buckets, evaluate, predict_latest, save_predictions, score, train_final,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def synthetic_features(n_days=700, signal=1.0, news_days=250, news_signal=0.0, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for d_i, d in enumerate(days):
        for t in TICKERS:
            r = {c: rng.normal() for c in FEATURES}
            r.update(ticker=t, date=d, close=100.0)
            covered = d_i >= n_days - news_days
            r["news_covered"] = int(covered)
            if not covered:
                for c in NEWS:
                    r[c] = np.nan
            z = signal * r["ret_1d"] + (news_signal * r["news_sent"] if covered else 0) + rng.normal()
            r[LABEL] = float(z > 0)
            r["next_ret"] = 0.0
            rows.append(r)
    feat = pd.DataFrame(rows)
    feat.loc[feat["date"] == days[-1], LABEL] = np.nan          # today: unknown
    return feat


def test_finds_planted_signal_and_calibrates():
    ev = evaluate(synthetic_features(signal=1.0))
    best = ev.results[ev.winner]
    assert best["accuracy"] > 0.65
    assert best["edge_pts"] > 10
    assert best["log_loss"] < best["log_loss_base"]
    assert ev.shuffle_accuracy <= ev.shuffle_bar + 0.03
    # Calibration: high bucket goes up more often than low bucket
    by = {name: actual for name, _, _, actual in ev.buckets}
    assert by["58%+"] > 0.7 and by["<42%"] < 0.3


def test_pure_noise_shows_no_edge():
    ev = evaluate(synthetic_features(signal=0.0))
    for name in ("logistic", "trees"):
        assert ev.results[name]["edge_pts"] < 3      # noise must never look like an edge
    assert not ev.use_news


def test_news_addon_used_only_when_news_helps():
    ev = evaluate(synthetic_features(signal=0.3, news_signal=1.5))
    assert ev.use_news
    best = ev.winner
    assert ev.results[f"{best} + news"]["accuracy"] > ev.results[f"{best} (news days)"]["accuracy"] + 0.05


def test_blank_and_constant_columns_are_skipped():
    feat = synthetic_features(signal=1.0)
    feat["earn_next"] = np.nan          # no earnings calendar at all
    feat["days_to_earn"] = np.nan
    feat["dow"] = 2.0                   # constant
    ev = evaluate(feat)
    assert ev.results[ev.winner]["accuracy"] > 0.65


def test_score_and_buckets_basics():
    y = np.array([1, 0, 1, 1, 0, 1])
    p = np.array([0.7, 0.3, 0.6, 0.55, 0.45, 0.4])
    r = score(y, p)
    assert r["calls"] == 4 and r["call_accuracy"] == pytest.approx(0.75)
    assert r["always_up"] == pytest.approx(4 / 6)
    assert sum(n for _, n, _, _ in buckets(y, p)) == 6


def test_end_to_end_predictions_and_bot_text(tmp_path):
    feat = synthetic_features(signal=1.0)
    ev = evaluate(feat)
    model, name = train_final(feat, ev)
    preds = predict_latest(model, feat)
    assert set(preds["ticker"]) == set(TICKERS)
    assert preds["p_up"].between(0, 1).all()

    s = Settings(watchlist=TICKERS, database_path=Path(tmp_path / "m.db"))
    conn = db.connect(s.database_path)
    save(conn, feat)
    save_predictions(conn, preds, name, ev)
    conn.close()

    text = today_text(s)
    assert "AAA" in text and "not backtested" in text
    one = odds_text(s, "bbb")
    assert "BBB" in one and "chance up" in one
    assert "isn't on the watchlist" in odds_text(s, "ZZZ")


def test_today_text_without_predictions(tmp_path):
    s = Settings(watchlist=TICKERS, database_path=Path(tmp_path / "e.db"))
    assert "No predictions yet" in today_text(s)
