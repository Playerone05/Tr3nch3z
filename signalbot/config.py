"""Loads settings from config.yaml and secrets from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    watchlist: list[str]
    context: list[str] = field(default_factory=list)
    sectors: dict[str, str] = field(default_factory=dict)
    history_start: str = "2019-01-01"
    news_lookback_days: int = 365
    market_timezone: str = "America/New_York"
    news_cutoff: str = "16:00"
    database_path: Path = ROOT / "data" / "signalbot.db"

    trench: dict = field(default_factory=dict)

    # Secrets (from .env). None means "not set".
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    finnhub_api_key: str | None = None

    @property
    def price_tickers(self) -> list[str]:
        """Watchlist plus market-context tickers, without duplicates."""
        seen: list[str] = []
        for t in [*self.watchlist, *self.context]:
            if t not in seen:
                seen.append(t)
        return seen

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def _env(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return value or None


def load_settings(config_path: str | Path | None = None) -> Settings:
    # override=True: values in .env win over leftover (possibly empty) shell variables.
    load_dotenv(ROOT / ".env", override=True)
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    db_path = Path(raw.get("database_path", "data/signalbot.db"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    watchlist = [str(t).upper().strip() for t in raw.get("watchlist", [])]
    if not watchlist:
        raise ValueError("config.yaml needs at least one ticker under `watchlist`.")

    return Settings(
        watchlist=watchlist,
        context=[str(t).upper().strip() for t in raw.get("context", [])],
        sectors={str(k).upper(): str(v).upper() for k, v in (raw.get("sectors") or {}).items()},
        history_start=str(raw.get("history_start", "2019-01-01")),
        news_lookback_days=int(raw.get("news_lookback_days", 365)),
        market_timezone=str(raw.get("market_timezone", "America/New_York")),
        news_cutoff=str(raw.get("news_cutoff", "16:00")),
        database_path=db_path,
        trench=raw.get("trench") or {},
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
        alpaca_api_key=_env("ALPACA_API_KEY"),
        alpaca_secret_key=_env("ALPACA_SECRET_KEY"),
        finnhub_api_key=_env("FINNHUB_API_KEY"),
    )
