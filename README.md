# Tr3nch3z — Signal Bot

A bot that estimates the chance each stock on your watchlist closes higher the next day, using price action, real news, and market context, and sends its calls to you on Telegram.

**Current stage: Phase 1 (data pipeline + bot basics).** The bot collects and stores data and answers `/status`. Predictions arrive in Phase 3.

> Not financial advice. This is a learning project. Paper trade for months before risking real money, and never use high multipliers.

---

## Setup (VS Code)

You need **Python 3.11+** and **Git** installed.

### 1. Get the code
In VS Code: `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → **Git: Clone** → paste this repo's URL → open the folder.

### 2. Create a virtual environment and install packages
Open the terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Mac / Linux
source .venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

If VS Code asks whether to use the new environment for this workspace, click **Yes**.

### 3. Add your keys
Copy `.env.example` to a new file named `.env` and fill in your values. `.env` is in `.gitignore`, so it never gets uploaded. **Never paste keys into chats, issues, or commits.**

| Key | Where to get it | Needed for |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | @BotFather in Telegram → `/newbot` (or `/revoke` to replace a leaked one) | The bot |
| `TELEGRAM_CHAT_ID` | Step 4 below | The bot |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | alpaca.markets → paper trading account → API keys | News (later: paper trades) |
| `FINNHUB_API_KEY` | finnhub.io → free account | News + earnings dates |

Price history comes from Yahoo Finance and needs no key. Anything without a key is skipped, not broken.

### 4. Find your Telegram chat ID
1. Open Telegram, find your bot, and send it any message.
2. Run `python -m signalbot.chat_id`
3. Put the number it prints into `.env` as `TELEGRAM_CHAT_ID=...`

### 5. Pull the data
```bash
python -m signalbot.refresh
```
The first run downloads ~7 years of prices and a year of news, which takes a few minutes (Finnhub's free tier allows about one call per second). Later runs only fetch what's new.

### 6. Start the bot
```bash
python -m signalbot.bot
```
Leave it running, then send `/status` to your bot in Telegram.

### 7. Run the tests (optional)
```bash
python -m pytest
```

---

## Commands

| Where | Command | What it does |
|---|---|---|
| Terminal | `python -m signalbot.refresh` | Update prices, news, earnings |
| Terminal | `python -m signalbot.refresh --only prices` | One step only (`prices`, `news`, `earnings`) |
| Terminal | `python -m signalbot.refresh --notify` | Also send a summary to Telegram |
| Terminal | `python -m signalbot.bot` | Run the Telegram bot |
| Telegram | `/status` | What data the bot has |
| Telegram | `/refresh` | Run a data refresh from your phone |
| Telegram | `/ping`, `/help` | Health check, command list |

The bot only answers your chat (`TELEGRAM_CHAT_ID`). Anyone else gets "This bot is private."

## Settings

Edit `config.yaml` to change the watchlist, market-context tickers, how far back to pull history, and the news cutoff time.

## How news timing works

The bot predicts from each day's close, so a headline only counts toward a trading day if it was public **before that day's 4:00 PM ET close**. After-hours, overnight, weekend, and holiday news rolls forward to the next session. The same headline from multiple outlets is stored once, at the time it first appeared. This is what keeps future information out of the model; see `signalbot/align.py` and `tests/test_align.py`.

## Project layout

```
config.yaml            watchlist and settings
.env.example           template for your keys (copy to .env)
signalbot/
  config.py            loads config.yaml + .env
  db.py                SQLite storage (data/signalbot.db)
  align.py             news → trading-day rules, headline dedupe
  refresh.py           the data pipeline command
  sources/             Yahoo prices, Alpaca + Finnhub news, Finnhub earnings
  bot.py               Telegram bot
  chat_id.py           finds your Telegram chat ID
  notify.py            one-off Telegram messages
tests/                 runs offline with fake data
```

## Roadmap

1. **Data pipeline** ← you are here
2. Features & sentiment: technical indicators, FinBERT news scores, earnings/Fed flags, market context
3. First model: logistic regression → LightGBM, calibrated probabilities
4. Honest backtest: walk-forward testing, leakage checks, scorecard vs. "always guess up"
5. Paper trading: nightly run, `/today`, `/odds`, `/log`, `/record` commands, 3+ months of logged calls
6. Report & polish: AI news summaries, dashboard, portfolio write-up

## Ground rules

- No input may come from after the prediction time.
- The model must beat "always guess up" (~53%) to count as an edge.
- 70%+ accuracy in a backtest means a bug, not a breakthrough.
- 3+ months of paper trading before any real money.
- Low multipliers only; never a setting where breakeven is above 55%.
- Risk 1–2% of bankroll per trade; stop for the day after 3 losses.
- "No call" is a valid answer.
