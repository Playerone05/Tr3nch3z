"""Send a one-off Telegram message (used by scripts; the interactive bot is bot.py)."""

from __future__ import annotations

import html

import requests

API = "https://api.telegram.org/bot{token}/{method}"


def escape(text: str) -> str:
    return html.escape(str(text), quote=False)


def send_message(token: str, chat_id: str, text: str, timeout: int = 20) -> None:
    """`text` may contain simple HTML (<b>, <i>, <code>). Escape any outside data with escape()."""
    resp = requests.post(
        API.format(token=token, method="sendMessage"),
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=timeout,
    )
    if resp.status_code != 200:
        # Never echo the URL: it contains the bot token.
        raise RuntimeError(f"Telegram sendMessage failed (HTTP {resp.status_code}): {resp.text[:200]}")
