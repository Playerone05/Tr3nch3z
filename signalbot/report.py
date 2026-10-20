"""Backtest charts.

    python -m signalbot.report       # redraw the Trench backtest chart from the last run

The chart is saved to data/trench_backtest.png. Open it with:
    xdg-open data/trench_backtest.png
or send /chart to the bot.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .config import ROOT

CHART_PATH = ROOT / "data" / "trench_backtest.png"

# Validated categorical slots (blue, orange, aqua) in fixed order; aqua is below 3:1 on the
# light surface, so every line also carries a direct text label.
SERIES = {"stock": "#2a78d6", "crypto": "#eb6834", "commodity": "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#52514c", "#8a897f", "#e6e5df"
KIND_NAME = {"stock": "Stocks", "crypto": "Crypto", "commodity": "Commodities"}


def _style(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


def best_per_kind(plan: dict) -> dict:
    """The hold the bot would use for each market type: tradeable first, then best per-trade result."""
    out = {}
    for v in plan.values():
        if not v.get("best"):
            continue
        k = v["kind"]
        key = (v["tradeable"], v["best"]["stake_ret"])
        if k not in out or key > (out[k]["tradeable"], out[k]["best"]["stake_ret"]):
            out[k] = v
    return out


def trench_chart(oos: pd.DataFrame, plan: dict, t, path=CHART_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .intraday import per_trade

    path.parent.mkdir(parents=True, exist_ok=True)
    picks = best_per_kind(plan)
    cut = next(iter(plan.values()))["judged_on"].split()[0] if plan else None
    fee_txt = f"fee {t.fee_bps:.1f} bps" if t.fee_bps is not None else "before Trench's fee (not set yet)"
    kinds = [k for k in ("stock", "crypto", "commodity") if any(v["kind"] == k for v in plan.values())]
    holds = sorted({v["hold"] for v in plan.values()})

    fig = plt.figure(figsize=(15, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.62, wspace=0.28,
                          left=0.06, right=0.97, top=0.86, bottom=0.08)
    gs_top = gs[0, :].subgridspec(1, 1)
    fig.text(0.06, 0.95, "Trench backtest: how the bot's calls would have done", fontsize=16,
             color=INK, fontweight="bold")
    fig.text(0.06, 0.915, f"{oos['day'].min()} to {oos['day'].max()} · {t.multiplier:g}x · {fee_txt} · "
             f"entering {t.entry_delay} min after each alert · every prediction out-of-sample",
             fontsize=10, color=INK2)

    # A: cumulative result -------------------------------------------------------------
    ax = fig.add_subplot(gs_top[0, 0])
    ax.set_position([0.06, ax.get_position().y0, 0.70, ax.get_position().height])   # room for end labels
    _style(ax, "If you'd bet the same stake on every call",
           "Running total in stakes (1.0 = one bet's size won). Each line uses that market type's best hold time.")
    ax.axhline(0, color=MUTED, linewidth=1)
    all_days = sorted(oos["day"].unique())
    x_of = {d: i for i, d in enumerate(all_days)}
    ends = []
    for k in kinds:
        v = picks.get(k)
        if not v:
            continue
        g = oos[(oos["kind"] == k) & (oos["hold"] == v["hold"])].sort_values("day")
        call, _, _, stake = per_trade(g["p"], g["fwd"], g["mae_long"], g["mae_short"],
                                      v["best"]["delta"], t.multiplier, t.fee_bps)
        daily = pd.Series(stake).groupby(g["day"].to_numpy()[call]).sum().reindex(all_days, fill_value=0.0)
        cum = daily.cumsum()
        xs = [x_of[d] for d in cum.index]
        ax.plot(xs, cum.to_numpy(), color=SERIES[k], linewidth=2, solid_capstyle="round")
        verdict = "TRADE" if v["tradeable"] else "no trade"
        ends.append((cum.iloc[-1], f"{KIND_NAME[k]} · {v['hold']} min: {cum.iloc[-1]:+.1f} stakes ({verdict})",
                     SERIES[k]))
    # Direct labels at the right edge, nudged apart so they don't collide
    if ends:
        ends.sort(key=lambda e: e[0])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.07
        placed = []
        for y, label, col in ends:
            y2 = max([y] + [p + gap for p in placed])
            placed.append(y2)
            ax.annotate(label, xy=(len(all_days) - 1, y), xytext=(len(all_days) + 1, y2), color=INK,
                        fontsize=9.5, va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col, linewidth=1.5))
    if cut in x_of:
        xc = x_of[cut]
        ax.axvline(xc, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(xc - 0.5, 1.0, "confidence bar picked here  ", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=8.5, color=INK2)
        ax.text(xc + 0.5, 1.0, "  judged here (never used to pick)", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8.5, color=INK2)
    step = max(1, len(all_days) // 8)
    ax.set_xticks(range(0, len(all_days), step))
    ax.set_xticklabels([pd.Timestamp(all_days[i]).strftime("%b %d") for i in range(0, len(all_days), step)])
    ax.set_xlim(-0.5, len(all_days) - 0.5)
    ax.set_ylabel("stakes", color=INK2, fontsize=9)

    width = 0.8 / max(len(kinds), 1)
    xpos = np.arange(len(holds))

    # B: hit rate ----------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _style(ax, "How often calls were right", "Judged half only. 50% = coin flip.")
    ax.axhline(50, color=MUTED, linewidth=1)
    for i, k in enumerate(kinds):
        vals = [(plan.get(f"{k}:{h}") or {}).get("best") for h in holds]
        ys = [b["hit"] * 100 if b else np.nan for b in vals]
        ax.bar(xpos + (i - (len(kinds) - 1) / 2) * width, ys, width * 0.92, color=SERIES[k],
               edgecolor=SURFACE, linewidth=2, label=KIND_NAME[k])
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{h} min" for h in holds])
    finite = [b["hit"] * 100 for v in plan.values() if (b := v.get("best"))]
    if finite:
        ax.set_ylim(min(45, min(finite) - 2), max(56, max(finite) + 2))
    ax.set_ylabel("% right", color=INK2, fontsize=9)

    # C: average move ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _style(ax, "Average move in your favor per call",
           "bps (0.01%) with 90% range. Whiskers crossing 0 = could be luck.")
    ax.axhline(0, color=MUTED, linewidth=1)
    if t.fee_bps is not None:
        ax.axhline(t.fee_bps, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(len(holds) - 0.5, t.fee_bps, " fee", color=INK2, fontsize=8.5, va="center")
    for i, k in enumerate(kinds):
        for j, h in enumerate(holds):
            b = (plan.get(f"{k}:{h}") or {}).get("best")
            if not b:
                continue
            x = j + (i - (len(kinds) - 1) / 2) * width
            ax.errorbar(x, b["bps"], yerr=[[b["bps"] - b["bps_lo"]], [b["bps_hi"] - b["bps"]]],
                        fmt="o", color=SERIES[k], markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=2, elinewidth=2, capsize=0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{h} min" for h in holds])
    ax.set_ylabel("bps", color=INK2, fontsize=9)

    # D: calibration -------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    _style(ax, "Does 'X% confident' mean X%?", "Judged half, each type's best hold. On the line = honest odds.")
    ax.plot([35, 65], [35, 65], color=MUTED, linewidth=1)
    edges = np.array([0, 0.44, 0.47, 0.49, 0.51, 0.53, 0.56, 1.0])
    for k in kinds:
        v = picks.get(k)
        if not v:
            continue
        g = oos[(oos["kind"] == k) & (oos["hold"] == v["hold"]) & (oos["day"] >= (cut or ""))]
        if g.empty:
            continue
        b = pd.cut(g["p"], edges, include_lowest=True)
        agg = g.assign(up=(g["fwd"] > 0).astype(float)).groupby(b, observed=True).agg(p=("p", "mean"),
                                                                                       up=("up", "mean"),
                                                                                       n=("up", "size"))
        agg = agg[agg["n"] >= 30]
        ax.plot(agg["p"] * 100, agg["up"] * 100, color=SERIES[k], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(38, 62)
    ax.set_ylim(38, 62)
    ax.set_xlabel("bot said (% up)", color=INK2, fontsize=9)
    ax.set_ylabel("actually went up (%)", color=INK2, fontsize=9)

    # Shared legend (identity is never color-alone: panel A is direct-labeled too)
    handles = [plt.Line2D([], [], color=SERIES[k], linewidth=3) for k in kinds]
    fig.legend(handles, [KIND_NAME[k] for k in kinds], loc="center", bbox_to_anchor=(0.5, 0.465),
               ncol=len(kinds), frameon=False, fontsize=10, labelcolor=INK)

    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    from .config import load_settings
    from .intraday import load_models, load_oos, trench_settings
    oos, art = load_oos(), load_models()
    if oos is None or art is None:
        print("No Trench backtest saved yet. Run `python -m signalbot.intraday backtest` first.")
        return 1
    path = trench_chart(oos, art["plan"], trench_settings(load_settings()))
    print(f"Chart saved: {path}\nOpen it with: xdg-open {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
