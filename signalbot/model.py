"""Phase 3: train the first model, check it honestly, and predict tomorrow.

    python -m signalbot.model          # evaluate, train, save, predict
    python -m signalbot.model --quiet  # just train + predict (for nightly runs)

What it does
1. Holds out the most recent stretch of days the model never trains on.
2. Trains two price + market models (logistic regression, gradient-boosted
   trees) on everything before it, then calibrates their probabilities so
   "60%" means up about 60% of the time.
3. Trains a news add-on on top of the best one, using only the days that have
   news, and checks whether it actually helps on the held-out days.
4. Runs a shuffle check: with randomly shuffled labels, accuracy must fall to
   the base rate. If it doesn't, something is leaking.
5. Retrains the winner on all data and stores tomorrow's probabilities.

This is a single train/test split, not a full backtest. Phase 4 replays every
past year walk-forward before any of these numbers should be trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import db
from .config import Settings, load_settings
from .features import CONTEXT, EVENTS, LABEL, NEWS, TECH, XS

BASE_FEATURES = TECH + CONTEXT + XS + EVENTS
CALL_UP, CALL_DOWN = 0.58, 0.42        # decision rule from the Signal Map
EMBARGO_DAYS = 5                        # gap between train and test so nothing bleeds across
EPS = 1e-4

PRED_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    date       TEXT NOT NULL,   -- trading day the features come from (its close)
    ticker     TEXT NOT NULL,
    p_up       REAL,            -- chance it closes higher the NEXT trading day
    model      TEXT,
    created_at TEXT,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS model_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    model      TEXT,
    report     TEXT              -- JSON metrics from the held-out check
);
"""


# ---------------------------------------------------------------- models

def make_logistic():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=3000),
    )


def make_trees(seed: int = 0):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.04, max_depth=3, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


MODELS = {"logistic": make_logistic, "trees": make_trees}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrated:
    """A fitted model plus Platt scaling learned on later, unseen days."""
    name: str
    features: list[str]
    model: object
    a: float = 1.0
    b: float = 0.0

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = self.a * logit(self.raw(X)) + self.b
        return 1 / (1 + np.exp(-z))


def usable_features(features: list[str], train: pd.DataFrame) -> list[str]:
    """Skip columns that are blank or constant in this training window (e.g. earnings
    dates in years before the calendar data starts); they carry no information."""
    return [c for c in features if train[c].nunique(dropna=True) > 1]


def fit_calibrated(name: str, features: list[str], train: pd.DataFrame, cal: pd.DataFrame,
                   label: str = LABEL) -> Calibrated:
    features = usable_features(features, train)
    m = MODELS[name]()
    m.fit(train[features], train[label].astype(int))
    c = Calibrated(name, features, m)
    if len(cal) >= 200 and cal[label].nunique() == 2:
        platt = LogisticRegression(C=1.0)
        platt.fit(logit(c.raw(cal)).reshape(-1, 1), cal[label].astype(int))
        c.a, c.b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    return c


@dataclass
class NewsAddOn:
    """Adjusts a base model's probability using news features."""
    base: Calibrated
    model: object
    name: str = "news-add-on"

    @staticmethod
    def inputs(base: Calibrated, X: pd.DataFrame) -> np.ndarray:
        return addon_inputs(base.predict(X), X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = self.base.predict(X)
        covered = (X["news_covered"] == 1).to_numpy()
        if covered.any():
            out = out.copy()
            out[covered] = self.model.predict_proba(self.inputs(self.base, X[covered]))[:, 1]
        return out


COUNT_COLUMNS = {"news_n", "news_focus_n", "news_n_ratio"}


def addon_inputs(p_base: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """Base model's log-odds plus news columns. Counts are log-scaled so one huge
    news day (hundreds of headlines) can't dominate."""
    cols = []
    for c in NEWS:
        v = X[c].fillna(0.0).to_numpy(dtype=float)
        cols.append(np.log1p(np.clip(v, 0, None)) if c in COUNT_COLUMNS else v)
    return np.column_stack([logit(p_base), *cols])


def make_addon_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


def fit_news_addon(base: Calibrated, rows: pd.DataFrame) -> NewsAddOn:
    """Train and predict with the SAME base model. (Phase 3 trained with one base model
    and predicted with another; the mismatch stretched its inputs and made it overconfident.)"""
    m = make_addon_model()
    m.fit(NewsAddOn.inputs(base, rows), rows[LABEL].astype(int))
    return NewsAddOn(base, m)


# ---------------------------------------------------------------- scoring

def score(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    base = y.mean()
    calls = (p >= CALL_UP) | (p <= CALL_DOWN)
    hit = ((p >= 0.5) == (y == 1))
    call_hit = ((p >= CALL_UP) & (y == 1)) | ((p <= CALL_DOWN) & (y == 0))
    return {
        "rows": int(len(y)),
        "accuracy": float(hit.mean()),
        "always_up": float(base),
        "edge_pts": float((hit.mean() - max(base, 1 - base)) * 100),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "log_loss_base": float(log_loss(y, np.full(len(y), base), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        "calls": int(calls.sum()),
        "call_rate": float(calls.mean()),
        "call_accuracy": float(call_hit[calls].mean()) if calls.any() else float("nan"),
        "mean_p": float(p.mean()),
    }


def buckets(y: np.ndarray, p: np.ndarray) -> list[tuple[str, int, float, float]]:
    """Calibration check: within each probability band, how often did it actually go up?"""
    edges = [0, 0.42, 0.48, 0.52, 0.58, 1.0001]
    names = ["<42%", "42-48%", "48-52%", "52-58%", "58%+"]
    out = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append((name, int(m.sum()), float(p[m].mean()), float(np.asarray(y)[m].mean())))
    return out


def row_log_loss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_win_share(y, p_old, p_new, dates, seed: int = 0, n: int = 1000) -> float:
    """Resample whole days with replacement; share of resamples where p_new has lower log loss.
    Guards against switching on a feature group that only won by luck."""
    diff = row_log_loss(y, p_old) - row_log_loss(y, p_new)        # >0 means new is better
    by_day = pd.Series(diff).groupby(pd.Series(dates)).agg(["sum", "count"])
    sums, counts = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float((means > 0).mean())


# ---------------------------------------------------------------- splitting

def trading_days(feat: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(feat["date"].unique())


def split_by_days(feat: pd.DataFrame, test_days: int, cal_days: int):
    """[ ...train... | embargo | cal | embargo | test ]  (all by trading day)"""
    days = trading_days(feat[feat[LABEL].notna()])
    if len(days) < test_days + cal_days + 2 * EMBARGO_DAYS + 250:
        raise ValueError("Not enough history to train and test. Run refresh with an earlier history_start.")
    test_start = days[-test_days]
    cal_end = days[-test_days - EMBARGO_DAYS - 1]
    cal_start = days[-test_days - EMBARGO_DAYS - cal_days]
    train_end = days[-test_days - 2 * EMBARGO_DAYS - cal_days - 1]
    labeled = feat[feat[LABEL].notna()]
    train = labeled[labeled["date"] <= train_end]
    cal = labeled[(labeled["date"] >= cal_start) & (labeled["date"] <= cal_end)]
    test = labeled[labeled["date"] >= test_start]
    return train, cal, test


@dataclass
class Evaluation:
    results: dict = field(default_factory=dict)       # name -> score dict
    buckets: list = field(default_factory=list)
    winner: str = ""
    use_news: bool = False
    shuffle_accuracy: float = float("nan")
    shuffle_bar: float = float("nan")     # accuracy of always guessing the more common direction
    test_from: str = ""
    test_to: str = ""
    news_note: str = ""


def evaluate(feat: pd.DataFrame, test_days: int = 120, cal_days: int = 120, seed: int = 0) -> Evaluation:
    train, cal, test = split_by_days(feat, test_days, cal_days)
    ev = Evaluation(test_from=f"{test['date'].min():%Y-%m-%d}", test_to=f"{test['date'].max():%Y-%m-%d}")
    y = test[LABEL].astype(int).to_numpy()

    fitted = {}
    for name in MODELS:
        m = fit_calibrated(name, BASE_FEATURES, train, cal)
        fitted[name] = m
        ev.results[name] = score(y, m.predict(test))
    best = min(fitted, key=lambda n: ev.results[n]["log_loss"])
    ev.winner = best

    # News add-on: learn from news-covered days before the test window.
    covered = feat[(feat["news_covered"] == 1) & feat[LABEL].notna()]
    news_train = covered[covered["date"] < test["date"].min() - pd.Timedelta(days=EMBARGO_DAYS + 2)]
    news_test = test[test["news_covered"] == 1]
    if news_train["date"].nunique() >= 40 and len(news_test) >= 100:
        # Inputs for training come from a base model that never saw those days.
        first_news = covered["date"].min()
        pre = train[train["date"] < first_news - pd.Timedelta(days=EMBARGO_DAYS + 2)]
        pre_cal = pre[pre["date"] >= pre["date"].max() - pd.Timedelta(days=180)]
        pre_train = pre[pre["date"] < pre_cal["date"].min()]
        base_inputs = fit_calibrated(best, BASE_FEATURES, pre_train, pre_cal)
        addon = fit_news_addon(base_inputs, news_train)
        yn = news_test[LABEL].astype(int).to_numpy()
        p_without = fitted[best].predict(news_test)
        p_with = addon.predict(news_test)
        ev.results[f"{best} (news days)"] = score(yn, p_without)
        ev.results[f"{best} + news"] = score(yn, p_with)
        win_share = bootstrap_win_share(yn, p_without, p_with, news_test["date"].to_numpy(), seed)
        better = win_share >= 0.90
        ev.use_news = bool(better)
        ev.news_note = (f"trained on {news_train['date'].nunique()} news days, tested on "
                        f"{news_test['date'].nunique()}; better in {win_share:.0%} of resamples "
                        f"(needs 90%) -> {'using it' if better else 'not using it yet'}")
    else:
        ev.news_note = "not enough scored news history yet to test the add-on"

    p_best = fitted[best].predict(test)
    ev.buckets = buckets(y, p_best)

    # Shuffle check: same features, labels shuffled at random.
    rng = np.random.default_rng(seed)
    shuf = train.copy()
    shuf[LABEL] = rng.permutation(shuf[LABEL].to_numpy())
    cols = usable_features(BASE_FEATURES, shuf)
    m = MODELS["logistic"]().fit(shuf[cols], shuf[LABEL].astype(int))
    ev.shuffle_accuracy = float(((m.predict_proba(test[cols])[:, 1] >= 0.5) == (y == 1)).mean())
    ev.shuffle_bar = float(max(y.mean(), 1 - y.mean()))
    return ev


# ---------------------------------------------------------------- final model

def train_final(feat: pd.DataFrame, ev: Evaluation, cal_days: int = 120):
    labeled = feat[feat[LABEL].notna()]
    days = trading_days(labeled)
    cal_start = days[-cal_days]
    train = labeled[labeled["date"] < days[-cal_days - EMBARGO_DAYS]]
    cal = labeled[labeled["date"] >= cal_start]
    base = fit_calibrated(ev.winner, BASE_FEATURES, train, cal)
    if not ev.use_news:
        return base, ev.winner
    covered = labeled[labeled["news_covered"] == 1]
    first_news = covered["date"].min()
    pre = labeled[labeled["date"] < first_news - pd.Timedelta(days=EMBARGO_DAYS + 2)]
    pre_cal = pre[pre["date"] >= pre["date"].max() - pd.Timedelta(days=180)]
    base_inputs = fit_calibrated(ev.winner, BASE_FEATURES, pre[pre["date"] < pre_cal["date"].min()], pre_cal)
    return fit_news_addon(base_inputs, covered), f"{ev.winner}+news"


def predict_latest(model, feat: pd.DataFrame) -> pd.DataFrame:
    latest = feat["date"].max()
    rows = feat[feat["date"] == latest].copy()
    rows["p_up"] = model.predict(rows)
    return rows[["date", "ticker", "p_up"]].sort_values("p_up", ascending=False)


def save_predictions(conn, preds: pd.DataFrame, model_name: str, ev: Evaluation | None) -> None:
    conn.executescript(PRED_SCHEMA)
    stamp = db.now_utc()
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (date, ticker, p_up, model, created_at) VALUES (?, ?, ?, ?, ?)",
        [(f"{r.date:%Y-%m-%d}", r.ticker, float(r.p_up), model_name, stamp) for r in preds.itertuples()],
    )
    if ev is not None:
        conn.execute("INSERT INTO model_runs (created_at, model, report) VALUES (?, ?, ?)",
                     (stamp, model_name, json.dumps({
                         "results": ev.results, "winner": ev.winner, "use_news": ev.use_news,
                         "shuffle_accuracy": ev.shuffle_accuracy, "test": [ev.test_from, ev.test_to],
                     })))
    conn.commit()


def load_features(conn) -> pd.DataFrame:
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'").fetchone()
    if not has:
        return pd.DataFrame()
    feat = pd.read_sql("SELECT * FROM features", conn)
    feat["date"] = pd.to_datetime(feat["date"])
    return feat


# ---------------------------------------------------------------- output

def call_label(p: float) -> str:
    if p >= CALL_UP:
        return "lean up"
    if p <= CALL_DOWN:
        return "lean down"
    return "no call"


def format_report(ev: Evaluation) -> str:
    L = [f"Held-out check: {ev.test_from} to {ev.test_to} (the model never trained on these days)", ""]
    L.append(f"  {'model':<24}{'rows':>6}{'accuracy':>10}{'always up':>11}{'edge':>8}"
             f"{'log loss':>10}{'vs base':>9}{'calls':>7}{'call acc':>10}")
    for name, r in ev.results.items():
        vs = r["log_loss_base"] - r["log_loss"]
        L.append(f"  {name:<24}{r['rows']:>6}{r['accuracy']:>10.1%}{r['always_up']:>11.1%}"
                 f"{r['edge_pts']:>+7.1f}p{r['log_loss']:>10.4f}{vs:>+9.4f}{r['calls']:>7}"
                 + (f"{r['call_accuracy']:>10.1%}" if r["calls"] else f"{'—':>10}"))
    L += [
        "",
        "  edge = accuracy minus the better of always-up / always-down, in points",
        "  vs base = log-loss improvement over just predicting the base rate (positive is better)",
        "  calls = predictions at 58%+ or 42%-; call acc = how often those were right",
        "",
        f"Calibration ({ev.winner}): when the model said X, how often did it go up?",
    ]
    for name, n, mp, actual in ev.buckets:
        L.append(f"  {name:>7}: {n:>5} predictions, avg {mp:.1%} -> actually up {actual:.1%}")
    L += [
        "",
        f"Shuffle check: {ev.shuffle_accuracy:.1%} with random labels; should not beat "
        f"always-guessing-the-common-direction ({ev.shuffle_bar:.1%})"
        f"{' -- OK' if ev.shuffle_accuracy <= ev.shuffle_bar + 0.03 else ' -- INVESTIGATE, possible leak'}",
        f"News add-on: {ev.news_note}",
        f"Using: {ev.winner}{' + news' if ev.use_news else ''}",
    ]
    return "\n".join(L)


def format_predictions(preds: pd.DataFrame) -> str:
    d = preds["date"].max()
    L = [f"Odds for the session after {d:%a %b %d} (from that day's close):"]
    for r in preds.itertuples():
        L.append(f"  {r.ticker:<6} {r.p_up:6.1%}  {call_label(r.p_up)}")
    return "\n".join(L)


def market_not_closed_yet() -> bool:
    now = pd.Timestamp.now(tz="America/New_York")
    return now.weekday() < 5 and (now.hour, now.minute) < (16, 15)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train, check, and run the prediction model.")
    p.add_argument("--quiet", action="store_true", help="skip the held-out report; train and predict only")
    p.add_argument("--test-days", type=int, default=120, help="trading days held out for the check")
    args = p.parse_args(argv)

    s = load_settings()
    conn = db.connect(s.database_path)
    feat = load_features(conn)
    if feat.empty:
        print("No feature table yet. Run `python -m signalbot.features` first.")
        return 1

    print("Checking the model on held-out days...", flush=True)
    ev = evaluate(feat, test_days=args.test_days)
    if not args.quiet:
        print()
        print(format_report(ev))
        print()

    model, name = train_final(feat, ev)
    preds = predict_latest(model, feat)
    save_predictions(conn, preds, name, ev)
    print(format_predictions(preds))
    if market_not_closed_yet() and preds["date"].max().date() == pd.Timestamp.now(tz="America/New_York").date():
        print("\nNote: the market hasn't closed yet, so today's prices (and these odds) aren't final."
              "\nRerun after 4:15 PM ET (3:15 PM Central) for the real numbers.")
    print("\nThis is one train/test split, not a backtest. Don't trade on it yet.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
