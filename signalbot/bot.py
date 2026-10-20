"""The Telegram bot. Run it and leave it running:

    python -m signalbot.bot

Main commands (Trench mode):
    /call NVDA   UP / DOWN / No trade, with a hold time of 1-5 minutes and the reasons
    /scan        every Trench market right now; lists only the live setups
    /tested      what the backtest says works, per market type and hold time

Also: /today (next-day outlook), /status, /help. It only answers your chat
(TELEGRAM_CHAT_ID); anyone else is told it's private.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from . import db
from .config import ROOT, Settings, load_settings
from .notify import escape
from .model import CALL_DOWN, CALL_UP, call_label
from .sentiment import coverage as sentiment_coverage

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
# httpx logs every request URL, and those URLs contain the bot token. Keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("signalbot")

COMMANDS = [
    ("call", "Trench call for one market, e.g. /call NVDA or /call btc"),
    ("scan", "Check every Trench market for a setup right now"),
    ("tested", "What the backtest says works, by market and hold time"),
    ("today", "Next-day outlook (daily model)"),
    ("status", "What data the bot has"),
    ("help", "List commands"),
]

_scorer = None


def get_scorer():
    """FinBERT, loaded once and kept in memory. None if it isn't installed."""
    global _scorer
    if _scorer is None:
        try:
            from .sentiment import finbert_scorer
            _scorer = finbert_scorer()
        except BaseException:
            _scorer = False
    return _scorer or None


def tested_text(s: Settings) -> str:
    import json
    from .intraday import trench_settings
    conn = db.connect(s.database_path)
    try:
        has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_runs'").fetchone()
        row = conn.execute("SELECT summary FROM intraday_runs ORDER BY id DESC LIMIT 1").fetchone() if has else None
    finally:
        conn.close()
    if not row:
        return "No Trench backtest yet. Run <code>python -m signalbot.intraday backtest</code>."
    sm = json.loads(row[0])
    t = trench_settings(s)
    fee = f"{sm['fee_bps']:.1f} bps" if sm.get("fee_bps") is not None else "not set"
    lines = [f"<b>Trench backtest</b> {escape(sm['start'])} → {escape(sm['end'])}",
             f"{sm['multiplier']:g}x · fee {escape(fee)} · out-of-sample", ""]
    for kind in ("stock", "crypto", "commodity"):
        rows = sorted((v for v in sm["plan"].values() if v["kind"] == kind), key=lambda v: v["hold"])
        if not rows:
            continue
        lines.append(f"<b>{kind}</b>")
        for v in rows:
            b = v.get("best")
            if not b:
                lines.append(f"  ⚪ {v['hold']} min: too few confident calls")
                continue
            mark = "🟢" if v["tradeable"] else "⚪"
            lines.append(f"  {mark} {v['hold']} min: right {b['hit']:.0%}, {b['bps']:+.1f} bps, "
                         f"{b['per_day']:.0f}/day, {b['bust_rate']:.1%} busts, max fee {b['max_fee_bps']:+.1f} bps")
    lines += ["", "🟢 = the bot will make calls here. ⚪ = it says No trade."]
    if sm.get("fee_bps") is None:
        lines.append("Set trench.fee_bps in config.yaml from a real ticket, then rerun the backtest.")
    return "\n".join(lines)


def status_text(s: Settings) -> str:
    conn = db.connect(s.database_path)
    try:
        sm = db.summary(conn)
        scored, total = sentiment_coverage(conn)
        has_feat = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'").fetchone()
        feat = conn.execute("SELECT COUNT(*), MAX(date) FROM features").fetchone() if has_feat else (0, None)
    finally:
        conn.close()
    if not sm["price_rows"]:
        return "No data yet. Send /refresh or run <code>python -m signalbot.refresh</code>."
    return "\n".join([
        "<b>Signal Bot status</b>",
        f"Prices: {sm['price_rows']:,} rows · {sm['price_tickers']} tickers",
        f"  {escape(sm['price_from'])} → {escape(sm['price_to'])}",
        f"News: {sm['news_rows']:,} stories · {sm['news_tickers']} tickers",
        f"  {escape(sm['news_from'] or '—')} → {escape(sm['news_to'] or '—')}",
        f"Headlines scored: {scored:,} of {total:,}",
        f"Earnings dates: {sm['earnings_rows']}",
        f"Feature rows: {feat[0]:,} (through {escape(feat[1] or '—')})",
        f"Last refresh: {escape(sm['last_refresh'] or 'never')} (UTC)",
        f"Watchlist: {escape(', '.join(s.watchlist))}",
    ])


UNTESTED = "<i>Phase 3: not backtested yet. Don't trade on these.</i>"


def _latest_predictions(conn):
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()
    if not has:
        return None, []
    day = conn.execute("SELECT MAX(date) FROM predictions").fetchone()[0]
    rows = conn.execute(
        "SELECT ticker, p_up, model FROM predictions WHERE date = ? ORDER BY p_up DESC", (day,)
    ).fetchall()
    return day, rows


def today_text(s: Settings) -> str:
    conn = db.connect(s.database_path)
    try:
        day, rows = _latest_predictions(conn)
    finally:
        conn.close()
    if not rows:
        return "No predictions yet. Run <code>python -m signalbot.model</code> or send /refresh."
    calls = [(t, p) for t, p, _ in rows if p >= CALL_UP or p <= CALL_DOWN]
    lines = [f"<b>Odds for the session after {escape(day)}</b>",
             f"{len(calls)} call{'s' if len(calls) != 1 else ''}, {len(rows) - len(calls)} no-call", ""]
    for t, p, _ in rows:
        mark = "🟢" if p >= CALL_UP else "🔴" if p <= CALL_DOWN else "⚪"
        lines.append(f"{mark} <code>{escape(t):<5} {p:5.1%}</code>  {call_label(p)}")
    lines += ["", f"Model: {escape(rows[0][2])}", UNTESTED]
    return "\n".join(lines)


ODDS_COLUMNS = [
    ("ret_1d", "Today's move", "{:+.1%}"), ("ret_5d", "5-day move", "{:+.1%}"),
    ("rsi_14", "RSI-14", "{:.0f}"), ("dist_sma20", "vs 20-day avg", "{:+.1%}"),
    ("vol_ratio_20", "Volume vs avg", "{:.1f}x"), ("rel_sector_5d", "vs sector (5d)", "{:+.1%}"),
    ("news_n", "Headlines today", "{:.0f}"), ("news_sent", "News sentiment", "{:+.2f}"),
    ("earn_next", "Earnings reaction next", "{:.0f}"), ("vix", "VIX", "{:.1f}"),
]


def odds_text(s: Settings, ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker not in s.watchlist:
        return f"{escape(ticker)} isn't on the watchlist. Add it to config.yaml first."
    conn = db.connect(s.database_path)
    try:
        day, rows = _latest_predictions(conn)
        hit = next((r for r in rows if r[0] == ticker), None)
        cols = ", ".join(c for c, _, _ in ODDS_COLUMNS)
        feat = conn.execute(
            f"SELECT {cols} FROM features WHERE ticker = ? AND date = ?", (ticker, day)
        ).fetchone() if day else None
    finally:
        conn.close()
    if not hit:
        return "No prediction for that stock yet. Run <code>python -m signalbot.model</code>."
    p = hit[1]
    lines = [f"<b>{escape(ticker)}: {p:.1%} chance up</b> after {escape(day)} · {call_label(p)}", ""]
    if feat:
        for (col, label, fmt), v in zip(ODDS_COLUMNS, feat):
            if v is None:
                continue
            if col == "earn_next":
                lines.append(f"{label}: {'yes' if v else 'no'}")
            else:
                lines.append(f"{label}: {fmt.format(v)}")
    lines += ["", UNTESTED]
    return "\n".join(lines)


def backtest_text(s: Settings) -> str:
    import json
    conn = db.connect(s.database_path)
    try:
        has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='backtest_runs'").fetchone()
        row = conn.execute(
            "SELECT created_at, params, summary FROM backtest_runs ORDER BY id DESC LIMIT 1").fetchone() if has else None
    finally:
        conn.close()
    if not row:
        return "No backtest yet. Run <code>python -m signalbot.backtest</code> (about 1-3 minutes)."
    params, summary = json.loads(row[1]), json.loads(row[2])
    icon = {"signal": "🟢", "weak": "🟡", "none": "⚪"}
    word = {"signal": "signal", "weak": "weak", "none": "no signal"}
    lines = [f"<b>Backtest</b> {escape(params['start'])} → {escape(params['end'])}", "<i>all out-of-sample</i>", ""]
    for target, block in summary["targets"].items():
        r = block["results"][block["best"]]
        lines.append(f"{icon[r['verdict']]} <b>{escape(target)}</b>: {word[r['verdict']]}")
        lines.append(f"   edge {r['edge_pts']:+.1f}p ({r['edge_lo']:+.1f} to {r['edge_hi']:+.1f}) · "
                     f"AUC {r['auc']:.3f} · {escape(block['best'])}")
        ex = r.get("extra", {})
        if "spread_bps_per_day" in ex:
            lines.append(f"   top-2 minus bottom-2: {ex['spread_bps_per_day']:+.1f} bps/day")
        n = summary.get("news", {}).get(target)
        if n:
            lines.append(f"   + news: {escape(n['verdict'])} ({n['win_share']:.0%})")
    lines += ["", "🟢 = beats the base rate in 95%+ of resamples"]
    return "\n".join(lines)


def build_app(s: Settings) -> Application:
    app = Application.builder().token(s.telegram_bot_token).build()
    owner = filters.Chat(chat_id=int(s.telegram_chat_id)) if s.telegram_chat_id else None

    async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        if owner is None:
            await update.message.reply_text(
                f"Setup mode. Your chat ID is {chat_id}.\n"
                "Add it to .env as TELEGRAM_CHAT_ID and restart the bot."
            )
        elif str(chat_id) == s.telegram_chat_id:
            await update.message.reply_text("Signal Bot is running. Send /help for commands.")
        else:
            await update.message.reply_text("This bot is private.")

    async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        lines = [f"/{name} - {desc}" for name, desc in COMMANDS]
        lines.append("\nMore: /odds TICKER, /backtest (daily model), /refresh, /ping")
        await update.message.reply_text("\n".join(lines))

    async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("pong")

    async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(status_text(s), parse_mode=ParseMode.HTML)

    async def refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Refreshing data and rerunning the model. This takes a few minutes...")
        code = 0
        steps = (("signalbot.refresh", ["--notify"]), ("signalbot.intraday", ["refresh"]), ("signalbot.sentiment", []),
                 ("signalbot.features", []), ("signalbot.model", ["--quiet"]))
        for module, extra in steps:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", module, *extra,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, cwd=str(ROOT),
            )
            rc = await proc.wait()
            code = code or rc
        if code != 0:
            await update.message.reply_text("Refresh finished with errors. Run it in the terminal to see details.")
        else:
            await update.message.reply_text(today_text(s), parse_mode=ParseMode.HTML)

    async def today(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(today_text(s), parse_mode=ParseMode.HTML)

    async def call(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        from .intraday import format_call, live_call, load_models, resolve_ticker, trench_settings
        t = trench_settings(s)
        if not ctx.args:
            await update.message.reply_text(f"Which market? e.g. /call NVDA\nYours: {', '.join(t.tickers)}")
            return
        tk = resolve_ticker(" ".join(ctx.args), t)
        if tk is None:
            await update.message.reply_text(f"Not one of your Trench markets: {', '.join(t.tickers)}")
            return
        art = load_models()
        if art is None:
            await update.message.reply_text("No trained models yet. Run: python -m signalbot.intraday backtest")
            return
        try:
            c = await asyncio.to_thread(live_call, tk, s, t, art, scorer=get_scorer())
            await update.message.reply_text(format_call(c, t, html=True), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"Couldn't make a call for {tk}: {exc}")

    async def scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        from .intraday import format_call, live_call, load_models, trench_settings
        t = trench_settings(s)
        art = load_models()
        if art is None:
            await update.message.reply_text("No trained models yet. Run: python -m signalbot.intraday backtest")
            return
        await update.message.reply_text(f"Scanning {len(t.tickers)} markets...")
        calls, quiet = [], []
        for tk in t.tickers:
            try:
                c = await asyncio.to_thread(live_call, tk, s, t, art, scorer=get_scorer())
            except Exception:
                quiet.append(f"{tk} (error)")
                continue
            (calls if c.direction != "NONE" else quiet).append(c if c.direction != "NONE" else tk)
        parts = [format_call(c, t, html=True) for c in calls] or ["<b>No setups right now.</b>"]
        if quiet:
            parts.append(f"⚪ No trade: {escape(', '.join(quiet))}")
        await update.message.reply_text("\n\n".join(parts), parse_mode=ParseMode.HTML)

    async def tested(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(tested_text(s), parse_mode=ParseMode.HTML)

    async def backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(backtest_text(s), parse_mode=ParseMode.HTML)

    async def odds(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await update.message.reply_text("Which stock? For example: /odds NVDA")
            return
        await update.message.reply_text(odds_text(s, ctx.args[0]), parse_mode=ParseMode.HTML)

    async def unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("I don't know that one yet. Send /help.")

    async def post_init(application: Application) -> None:
        await application.bot.set_my_commands([BotCommand(n, d) for n, d in COMMANDS])

    app.post_init = post_init
    app.add_handler(CommandHandler("start", start))
    if owner is not None:
        app.add_handler(CommandHandler("help", help_cmd, filters=owner))
        app.add_handler(CommandHandler("today", today, filters=owner))
        app.add_handler(CommandHandler("odds", odds, filters=owner))
        app.add_handler(CommandHandler("backtest", backtest, filters=owner))
        app.add_handler(CommandHandler("call", call, filters=owner))
        app.add_handler(CommandHandler("scan", scan, filters=owner))
        app.add_handler(CommandHandler("tested", tested, filters=owner))
        app.add_handler(CommandHandler("ping", ping, filters=owner))
        app.add_handler(CommandHandler("status", status, filters=owner))
        app.add_handler(CommandHandler("refresh", refresh, filters=owner))
        # Any other command from you gets a hint; other chats are ignored.
        app.add_handler(MessageHandler(filters.COMMAND & owner, unknown))
    return app


def main() -> int:
    s = load_settings()
    if not s.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file.")
        return 1
    if not s.telegram_chat_id:
        print("TELEGRAM_CHAT_ID isn't set, so the bot starts in setup mode.")
        print("Send /start to your bot in Telegram to see your chat ID.")
    print("Bot is running. Press Ctrl+C to stop.")
    build_app(s).run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
