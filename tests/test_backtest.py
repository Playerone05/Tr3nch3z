"""Phase 4 tests: walk-forward finds a planted edge, calls noise 'no signal', and never peeks."""

import numpy as np
import pandas as pd
import pytest

from signalbot.backtest import add_targets, format_report, make_folds, run, split_fold, summarize, peer_spread
from signalbot.features import FEATURES, LABEL, NEWS

TICKERS = ["A1", "A2", "A3", "A4", "A5", "A6"]


def synthetic(n_days=760, signal=0.0, news_signal=0.0, news_days=300, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-02", periods=n_days)
    rows = []
    for i, d in enumerate(days):
        for t in TICKERS:
            r = {c: rng.normal() for c in FEATURES}
            covered = i >= n_days - news_days
            if not covered:
                for c in NEWS:
                    r[c] = np.nan
            drift = signal * r["ret_1d"] + (news_signal * r["news_sent"] if covered else 0.0)
            nxt = 0.01 * (drift + rng.normal())
            r.update(ticker=t, date=d, close=100.0, news_covered=int(covered), volat_20=0.01, usual_move=0.0067,
                     next_ret=nxt, next_ret_5=nxt + 0.01 * rng.normal() * 2)
            r[LABEL] = float(nxt > 0)
            rows.append(r)
    f = pd.DataFrame(rows)
    last = f["date"] == days[-1]
    f.loc[last, ["next_ret", "next_ret_5", LABEL]] = np.nan
    return f


def test_targets_are_built_correctly():
    f = add_targets(synthetic(n_days=30))
    day = f[f["date"] == f["date"].iloc[0]]
    assert day["y_beat_peers"].sum() == 3                    # 6 stocks: top half beats the median
    assert set(f["y_big_move"].dropna().unique()) <= {0.0, 1.0}
    assert f.loc[f["date"] == f["date"].max(), "y_direction"].isna().all()


def test_folds_never_train_on_the_test_window_or_its_gap():
    f = add_targets(synthetic(n_days=500))
    days = sorted(f["date"].unique())
    for i0, i1 in make_folds(days, "2020-06-01", 63):
        train, cal = split_fold(f, days, i0, "y_direction_5d")
        assert train["date"].max() < cal["date"].min()
        # 5-day labels on the last training day must end before the test block starts
        assert days.index(cal["date"].max()) + 5 < i0


def test_planted_signal_is_found():
    oos, news = run(synthetic(signal=0.6), start="2021-01-01", step=63, models=("logistic",), progress=None)
    s = summarize(oos, news)
    for target in ("direction", "beat_peers", "direction_5d"):
        r = s["targets"][target]["results"]["logistic"]
        assert r["verdict"] == "signal", target
        assert r["edge_lo"] > 0
        assert r["auc"] > 0.6
    assert s["targets"]["beat_peers"]["results"]["logistic"]["extra"]["spread_bps_per_day"] > 0
    assert "SIGNAL" in format_report(s, "a", "b")


def test_noise_is_not_called_a_signal():
    oos, news = run(synthetic(signal=0.0), start="2021-01-01", step=63, models=("logistic",), progress=None)
    s = summarize(oos, news)
    for target, block in s["targets"].items():
        r = block["results"]["logistic"]
        assert r["verdict"] != "signal", target
        assert r["edge_lo"] <= 0, target            # the 90% range includes zero


def test_news_addon_detects_real_news_signal():
    oos, news = run(synthetic(signal=0.2, news_signal=1.0), start="2021-01-01", step=63,
                    models=("logistic",), targets=("direction",), progress=None)
    s = summarize(oos, news)
    assert s["news"]["direction"]["verdict"] == "helps"
    assert s["news"]["direction"]["acc_with"] > s["news"]["direction"]["acc_without"]


def test_peer_spread_sign():
    oos = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-05")] * 4,
        "p": [0.9, 0.8, 0.2, 0.1],
        "next_ret": [0.02, 0.01, -0.01, -0.02],
    })
    out = peer_spread(oos, k=2)
    assert out["spread_bps_per_day"] == pytest.approx(300.0)
