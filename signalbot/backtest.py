"""Phase 4: walk-forward backtest. Replays history the way the bot would have lived it.

    python -m signalbot.backtest                 # all four questions, quarterly since 2021
    python -m signalbot.backtest --start 2023-01-01 --models trees

For each quarter since --start, the models train only on days before it (with a
gap), calibrate on the latest 120 of those days, then predict the quarter. Every
prediction in the report is therefore out-of-sample.

It asks four different questions, because some are much easier than "up or down":

  direction     Will the stock close higher tomorrow?
  beat_peers    Will it do better than the other watchlist stocks tomorrow?
  big_move      Will tomorrow's move (either way) be bigger than its usual daily size?
  direction_5d  Will it be higher 5 trading days from now?

A question gets a "signal" verdict only if the model beats the simple base rate
in at least 95% of bootstrap resamples of the test days.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import db
from .config import load_settings
from .features import LABEL, NEWS
from .model import (
    BASE_FEATURES, CALL_DOWN, CALL_UP, EMBARGO_DAYS, EPS, MODELS, addon_inputs, bootstrap_win_share,
    fit_calibrated, load_features, logit, make_addon_model, row_log_loss,
)

TARGETS = {
    "direction": "Up tomorrow",
    "beat_peers": "Beats the other watchlist stocks tomorrow",
    "big_move": "Moves more than its typical day (1-year median move) tomorrow",
    "direction_5d": "Up 5 trading days from now",
}
SIGNAL_SHARE, WEAK_SHARE = 0.95, 0.80
CAL_DAYS = 120

SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    params TEXT,
    summary TEXT
);
"""


# ---------------------------------------------------------------- labels

def add_targets(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["y_direction"] = f[LABEL]
    med = f.groupby("date")["next_ret"].transform("median")
    peers = f.groupby("date")["next_ret"].transform("count")
    f["y_beat_peers"] = np.where(f["next_ret"].isna() | (peers < 3), np.nan, (f["next_ret"] > med).astype(float))
    usual = f["usual_move"]
    f["y_big_move"] = np.where(f["next_ret"].isna() | usual.isna(), np.nan,
                               (f["next_ret"].abs() > usual).astype(float))
    f["y_direction_5d"] = np.where(f["next_ret_5"].isna(), np.nan, (f["next_ret_5"] > 0).astype(float))
    return f


# ---------------------------------------------------------------- folds

def make_folds(days: list, start: str, step: int) -> list[tuple[int, int]]:
    """(first test index, last test index + 1) for each block of `step` trading days."""
    first = next((i for i, d in enumerate(days) if d >= pd.Timestamp(start)), None)
    if first is None:
        return []
    need = CAL_DAYS + 2 * EMBARGO_DAYS + 250
    first = max(first, need)
    return [(i, min(i + step, len(days))) for i in range(first, len(days), step)]


def split_fold(f: pd.DataFrame, days: list, i0: int, label: str):
    pool_end = days[i0 - EMBARGO_DAYS - 1]
    pool = f[(f["date"] <= pool_end) & f[label].notna()]
    pool_days = sorted(pool["date"].unique())
    cal_start = pool_days[-CAL_DAYS]
    train_end = pool_days[-CAL_DAYS - EMBARGO_DAYS - 1]
    return pool[pool["date"] <= train_end], pool[pool["date"] >= cal_start]


# ---------------------------------------------------------------- scoring

def day_bootstrap(values_by_row: np.ndarray, dates: np.ndarray, seed: int = 0, n: int = 1000):
    """Mean of a per-row quantity, resampling whole days. Returns (5th pct, 95th pct)."""
    g = pd.Series(values_by_row).groupby(pd.Series(dates)).agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


@dataclass
class TargetResult:
    target: str
    model: str
    rows: int
    days: int
    base_rate: float
    accuracy: float
    base_accuracy: float          # always guessing whichever side the training data favored
    edge_pts: float
    edge_lo: float
    edge_hi: float
    ll_gain: float                # log-loss improvement over the base-rate forecast (x1000)
    win_share: float
    auc: float
    verdict: str
    by_year: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def evaluate_oos(oos: pd.DataFrame, target: str, model: str, seed: int = 0) -> TargetResult:
    from sklearn.metrics import roc_auc_score

    y = oos["y"].to_numpy(dtype=int)
    p = np.clip(oos["p"].to_numpy(dtype=float), EPS, 1 - EPS)
    pb = np.clip(oos["p_base"].to_numpy(dtype=float), EPS, 1 - EPS)
    dates = oos["date"].to_numpy()
    years_of = pd.to_datetime(dates)
    if target == "direction_5d":
        # 5-day labels overlap, so neighbouring days aren't independent: resample whole weeks.
        rank = pd.Series(dates).rank(method="dense").to_numpy().astype(int)
        dates = rank // 5

    hit = ((p >= 0.5) == (y == 1)).astype(float)
    base_hit = ((pb >= 0.5) == (y == 1)).astype(float)
    lo, hi = day_bootstrap(hit - base_hit, dates, seed)
    win = bootstrap_win_share(y, pb, p, dates, seed)
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    # Better probabilities than the base rate in 95%+ of resamples, AND real ranking skill.
    verdict = "signal" if win >= SIGNAL_SHARE and auc > 0.51 else \
              "weak" if win >= WEAK_SHARE else "none"

    years = pd.Series(hit - base_hit).groupby(pd.Series(years_of).dt.year).mean() * 100
    res = TargetResult(
        target=target, model=model, rows=len(y), days=int(pd.Series(years_of).nunique()),
        base_rate=float(y.mean()), accuracy=float(hit.mean()), base_accuracy=float(base_hit.mean()),
        edge_pts=float((hit.mean() - base_hit.mean()) * 100), edge_lo=lo * 100, edge_hi=hi * 100,
        ll_gain=float((row_log_loss(y, pb).mean() - row_log_loss(y, p).mean()) * 1000),
        win_share=win, auc=auc,
        verdict=verdict, by_year={int(k): round(float(v), 1) for k, v in years.items()},
    )

    if target in ("direction", "direction_5d"):
        for up, dn in ((0.55, 0.45), (CALL_UP, CALL_DOWN), (0.60, 0.40)):
            m = (p >= up) | (p <= dn)
            right = ((p >= up) & (y == 1)) | ((p <= dn) & (y == 0))
            res.extra[f"calls_{int(up * 100)}"] = {
                "n": int(m.sum()), "per_day": round(float(m.sum()) / res.days, 2),
                "accuracy": round(float(right[m].mean()), 4) if m.any() else None,
            }
    if target == "beat_peers" and "next_ret" in oos:
        res.extra.update(peer_spread(oos))
    return res


def peer_spread(oos: pd.DataFrame, k: int = 2) -> dict:
    """Each day: average next-day return of the model's top-k picks minus its bottom-k picks."""
    rows = []
    for d, g in oos.groupby("date"):
        if len(g) < 2 * k:
            continue
        g = g.sort_values("p")
        rows.append(g["next_ret"].tail(k).mean() - g["next_ret"].head(k).mean())
    s = pd.Series(rows, dtype=float)
    if s.empty:
        return {}
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 and s.std() > 0 else float("nan")
    return {"spread_bps_per_day": round(float(s.mean() * 1e4), 2), "spread_t": round(float(t), 2),
            "spread_positive_days": round(float((s > 0).mean()), 4), "spread_days": int(len(s))}


# ---------------------------------------------------------------- run

def run(feat: pd.DataFrame, start: str = "2021-01-01", step: int = 63, models=("logistic", "trees"),
        targets=tuple(TARGETS), progress=print, seed: int = 0):
    f = add_targets(feat)
    days = sorted(f["date"].unique())
    folds = make_folds(days, start, step)
    if not folds:
        raise ValueError("Not enough history for a walk-forward backtest.")
    oos_parts = []
    t0 = time.time()
    for k, (i0, i1) in enumerate(folds, 1):
        test_days = days[i0:i1]
        test_all = f[f["date"].isin(test_days)]
        for target in targets:
            label = f"y_{target}"
            test = test_all[test_all[label].notna()]
            if test.empty:
                continue
            train, cal = split_fold(f, days, i0, label)
            # Baseline = the up-rate of the most recent 120 days. The model's calibration sees
            # those same days, so beating this means real skill, not just tracking the trend.
            p_base = float(cal[label].mean())
            for name in models:
                m = fit_calibrated(name, BASE_FEATURES, train, cal, label=label)
                oos_parts.append(pd.DataFrame({
                    "date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy(),
                    "target": target, "model": name,
                    "p": m.predict(test), "p_base": p_base, "y": test[label].to_numpy(),
                    "next_ret": test["next_ret"].to_numpy(),
                    "news_covered": test["news_covered"].to_numpy(),
                }))
        if progress:
            done = time.time() - t0
            left = done / k * (len(folds) - k)
            progress(f"  quarter {k}/{len(folds)} ({test_days[0]:%Y-%m-%d}) done, ~{left / 60:.1f} min left")
    oos = pd.concat(oos_parts, ignore_index=True)
    news = news_addon_oos(oos, f, folds, days)
    return oos, news


def news_addon_oos(oos: pd.DataFrame, f: pd.DataFrame, folds, days) -> pd.DataFrame:
    """For days with news: does adding news to the base model's out-of-sample probability help?
    Each quarter's add-on learns only from EARLIER quarters' out-of-sample rows."""
    parts = []
    cols = ["date", "ticker", *NEWS]
    news_cols = f[cols]
    for (target, model), g in oos.groupby(["target", "model"]):
        g = g[g["news_covered"] == 1].merge(news_cols, on=["date", "ticker"], how="left")
        if g.empty:
            continue
        for i0, i1 in folds:
            block = g[(g["date"] >= days[i0]) & (g["date"] <= days[i1 - 1])]
            past = g[g["date"] < days[max(i0 - EMBARGO_DAYS, 0)]]
            if block.empty or past["date"].nunique() < 40:
                continue
            m = make_addon_model()
            m.fit(addon_inputs(past["p"].to_numpy(), past), past["y"].astype(int))
            out = block[["date", "ticker", "target", "model", "p", "p_base", "y", "next_ret"]].copy()
            out["p_news"] = m.predict_proba(addon_inputs(block["p"].to_numpy(), block))[:, 1]
            parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize(oos: pd.DataFrame, news: pd.DataFrame, seed: int = 0) -> dict:
    out = {"targets": {}, "news": {}}
    for target in TARGETS:
        sub = oos[oos["target"] == target]
        if sub.empty:
            continue
        results = {m: evaluate_oos(g, target, m, seed) for m, g in sub.groupby("model")}
        best = max(results.values(), key=lambda r: r.ll_gain)
        out["targets"][target] = {"best": best.model, "results": {m: r.__dict__ for m, r in results.items()}}
        if not news.empty:
            n = news[(news["target"] == target) & (news["model"] == best.model)]
            if len(n) >= 200:
                y = n["y"].to_numpy(dtype=int)
                win = bootstrap_win_share(y, n["p"].to_numpy(), n["p_news"].to_numpy(), n["date"].to_numpy(), seed)
                gain = (row_log_loss(y, n["p"]).mean() - row_log_loss(y, n["p_news"]).mean()) * 1000
                acc_without = float(((n["p"] >= 0.5) == (y == 1)).mean())
                acc_with = float(((n["p_news"] >= 0.5) == (y == 1)).mean())
                out["news"][target] = {
                    "days": int(n["date"].nunique()), "rows": int(len(n)), "win_share": win,
                    "ll_gain": float(gain), "acc_without": acc_without, "acc_with": acc_with,
                    "verdict": "helps" if win >= SIGNAL_SHARE else "maybe" if win >= WEAK_SHARE else "no help",
                }
    return out


# ---------------------------------------------------------------- report

VERDICT_TEXT = {"signal": "SIGNAL", "weak": "weak / unproven", "none": "no signal"}


def format_report(summary: dict, start: str, end: str) -> str:
    L = [f"Walk-forward backtest, {start} to {end}. Every prediction is out-of-sample.", ""]
    for target, block in summary["targets"].items():
        L.append(f"[{target}] {TARGETS[target]}")
        for m, r in block["results"].items():
            star = "*" if m == block["best"] else " "
            L.append(
                f" {star}{m:<9} acc {r['accuracy']:.1%} vs base {r['base_accuracy']:.1%}  "
                f"edge {r['edge_pts']:+.1f}p (90% range {r['edge_lo']:+.1f} to {r['edge_hi']:+.1f})  "
                f"log-loss gain {r['ll_gain']:+.2f}  AUC {r['auc']:.3f}  -> {VERDICT_TEXT[r['verdict']]} "
                f"({r['win_share']:.0%} of resamples)")
        best = block["results"][block["best"]]
        years = "  ".join(f"{y}: {v:+.1f}" for y, v in best["by_year"].items())
        L.append(f"   edge by year ({block['best']}): {years}")
        ex = best["extra"]
        if "spread_bps_per_day" in ex:
            L.append(f"   top-2 minus bottom-2 picks: {ex['spread_bps_per_day']:+.1f} bps/day "
                     f"(t = {ex['spread_t']:.2f}, positive on {ex['spread_positive_days']:.0%} of "
                     f"{ex['spread_days']} days, before costs)")
        for key in ("calls_55", "calls_58", "calls_60"):
            if key in ex and ex[key]["n"]:
                th = key.split("_")[1]
                L.append(f"   calls at {th}%+ / {100 - int(th)}%-: {ex[key]['n']} "
                         f"({ex[key]['per_day']}/day), right {ex[key]['accuracy']:.1%}")
        if target in summary["news"]:
            n = summary["news"][target]
            L.append(f"   + news ({n['days']} days): acc {n['acc_without']:.1%} -> {n['acc_with']:.1%}, "
                     f"log-loss gain {n['ll_gain']:+.2f}, better in {n['win_share']:.0%} of resamples "
                     f"-> {n['verdict']}")
        L.append("")
    L += [
        "How to read it:",
        "  edge      = accuracy minus always guessing the side the last 120 days favored, in points",
        "  90% range = where the edge lands in 90% of day-level resamples; if it includes 0, it could be luck",
        "  log-loss gain = how much better the probabilities are than the base rate (x1000; positive = better)",
        "  AUC       = 0.5 means no ranking skill; 0.52-0.55 is typical of a real but small edge",
        "  SIGNAL    = better probabilities than the base rate in 95%+ of resamples, and AUC above 0.51",
    ]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest of every target.")
    ap.add_argument("--start", default="2021-01-01", help="first test day (default 2021-01-01)")
    ap.add_argument("--step", type=int, default=63, help="trading days per test block (63 = a quarter)")
    ap.add_argument("--models", default="logistic,trees", help="comma list: logistic,trees")
    ap.add_argument("--targets", default=",".join(TARGETS), help="comma list of targets")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    targets = [t.strip() for t in args.targets.split(",") if t.strip() in TARGETS]
    s = load_settings()
    conn = db.connect(s.database_path)
    feat = load_features(conn)
    if feat.empty or "next_ret_5" not in feat.columns:
        print("Feature table is missing or old. Run `python -m signalbot.features` first.")
        return 1

    print(f"Backtesting {len(targets)} questions x {len(models)} models, quarterly from {args.start}...", flush=True)
    oos, news = run(feat, args.start, args.step, models, targets, progress=lambda m: print(m, flush=True))
    summary = summarize(oos, news)
    start, end = f"{oos['date'].min():%Y-%m-%d}", f"{oos['date'].max():%Y-%m-%d}"
    print()
    print(format_report(summary, start, end))

    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO backtest_runs (created_at, params, summary) VALUES (?, ?, ?)", (
        db.now_utc(), json.dumps({"start": start, "end": end, "step": args.step, "models": models}),
        json.dumps(summary, default=float)))
    save = oos.copy()
    save["date"] = save["date"].dt.strftime("%Y-%m-%d")
    save.to_sql("backtest_predictions", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("\nSaved. Send /backtest in Telegram for the summary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
