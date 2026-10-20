"""The Telegram bot. Run it and leave it running:

    python -m signalbot.bot

It only answers your chat (TELEGRAM_CHAT_ID). Anyone else who finds the bot gets
told it's private. Prediction commands (/today, /odds, /log, /record) arrive in
later phases; Phase 1 has /status, /refresh, /ping, and /help.
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
from .sentiment import coverage as sentiment_coverage

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
# httpx logs every request URL, and those URLs contain the bot token. Keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("signalbot")

COMMANDS = [
    ("status", "What data the bot has"),
    ("refresh", "Pull the latest prices and news"),
    ("ping", "Check the bot is alive"),
    ("help", "List commands"),
]


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
        lines.append("\nComing later: /today, /odds TICKER, /log, /record, /pause")
        await update.message.reply_text("\n".join(lines))

    async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("pong")

    async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(status_text(s), parse_mode=ParseMode.HTML)

    async def refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Refreshing data. This can take a few minutes on the first run...")
        code = 0
        for module, extra in (("signalbot.refresh", ["--notify"]), ("signalbot.features", [])):
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", module, *extra,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, cwd=str(ROOT),
            )
            rc = await proc.wait()
            code = code or rc
        if code != 0:
            await update.message.reply_text("Refresh finished with errors. Run it in the terminal to see details.")

    async def unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("I don't know that one yet. Send /help.")

    async def post_init(application: Application) -> None:
        await application.bot.set_my_commands([BotCommand(n, d) for n, d in COMMANDS])

    app.post_init = post_init
    app.add_handler(CommandHandler("start", start))
    if owner is not None:
        app.add_handler(CommandHandler("help", help_cmd, filters=owner))
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
