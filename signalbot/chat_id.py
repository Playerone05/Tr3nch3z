"""Find your Telegram chat ID.

1. Open Telegram, find your bot, and send it any message (for example "hi").
2. Run:  python -m signalbot.chat_id
3. Copy the number it prints into TELEGRAM_CHAT_ID in your .env file.

Stop the bot (python -m signalbot.bot) before running this; Telegram only lets
one program read the bot's messages at a time.
"""

from __future__ import annotations

import sys

import requests

from .config import load_settings
from .notify import API


def main() -> int:
    s = load_settings()
    if not s.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file first.")
        return 1

    resp = requests.get(API.format(token=s.telegram_bot_token, method="getUpdates"), timeout=20)
    if resp.status_code == 401:
        print("Telegram rejected the token. Copy it again from @BotFather.")
        return 1
    if resp.status_code == 409:
        print("The bot is running somewhere else. Stop `python -m signalbot.bot` and try again.")
        return 1
    if resp.status_code != 200:
        print(f"Telegram returned HTTP {resp.status_code}: {resp.text[:200]}")
        return 1

    chats = {}
    for update in resp.json().get("result", []):
        msg = update.get("message") or update.get("edited_message") or {}
        chat = msg.get("chat")
        if chat:
            name = chat.get("username") or chat.get("first_name") or chat.get("title") or "?"
            chats[chat["id"]] = name

    if not chats:
        print("No messages found. Send your bot a message in Telegram, then run this again.")
        return 1

    print("Chats that have messaged your bot:")
    for chat_id, name in chats.items():
        print(f"  {name}: {chat_id}")
    print("\nPut your number in .env as:  TELEGRAM_CHAT_ID=<number>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
