# Tr3nch3z — Signal Bot

A bot that estimates the chance each stock on your watchlist closes higher the next day, using price action, real news, and market context, and sends its calls to you on Telegram.

**Current stage: Phase 2 (features + news sentiment).** The bot collects data, scores headlines with FinBERT, and builds the table the model will learn from. Predictions arrive in Phase 3.

> Not financial advice. This is a learning project. Paper trade for months before risking real money, and never use high multipliers.

---

## Setup (Linux terminal)

```bash
sudo apt update
sudo apt install -y git vim python3-venv python3-pip
git clone https://github.com/Playerone05/Tr3nch3z.git
cd Tr3nch3z
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Every time you open a new terminal:

```bash
cd ~/tr3nch3z            # or wherever you cloned it
source .venv/bin/activate
```

Your prompt starts with `(.venv)` when it's active. (Mac is the same apart from the `apt` line. On Windows, activate with `.venv\Scripts\Activate.ps1`.)

### Add your keys

```bash
vim .env
```

Arrow to a line, press `A` to type at the end of it, paste with `Ctrl+Shift+V`, press `Esc`. When done, type `:wq` and press Enter. (`:q!` quits without saving.)

`.env` is in `.gitignore`, so it never gets uploaded. **Never paste keys into chats, issues, or commits.**

| Key | Where to get it | Needed for |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | @BotFather in Telegram → `/newbot` (or `/revoke` to replace a leaked one) | The bot |
| `TELEGRAM_CHAT_ID` | Message your bot, then run `python -m signalbot.chat_id` | The bot |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | alpaca.markets → Trading API → **Paper** account → API Keys | News (later: paper trades) |
| `FINNHUB_API_KEY` | finnhub.io → free account | News + earnings dates |

Prices and Treasury yields come from Yahoo Finance and need no key. Anything without a key is skipped, not broken.

### Install FinBERT (one time, for news sentiment)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt
```

The first line installs PyTorch's CPU-only build (~200 MB instead of 2+ GB).

---

## Daily use

```bash
python -m signalbot.refresh       # 1. pull new prices, news, earnings
python -m signalbot.sentiment     # 2. score new headlines with FinBERT
python -m signalbot.features      # 3. rebuild the feature table
python -m signalbot.bot           # run the Telegram bot (leave it running)
```

The first `sentiment` run downloads the model (~440 MB) and scores every stored headline, which can take 10–20 minutes on a laptop. It saves as it goes, so Ctrl+C is safe; the next run picks up where it stopped. After that, each run takes seconds.

To peek at the table: `python -m signalbot.features --show NVDA`

## Commands

| Where | Command | What it does |
|---|---|---|
| Terminal | `python -m signalbot.refresh` | Update prices, news, earnings |
| Terminal | `python -m signalbot.refresh --only news` | One step only (`prices`, `news`, `earnings`) |
| Terminal | `python -m signalbot.refresh --notify` | Also send a summary to Telegram |
| Terminal | `python -m signalbot.sentiment` | Score unscored headlines (`--limit 200` for a quick test) |
| Terminal | `python -m signalbot.features --show NVDA` | Build features, print NVDA's latest rows |
| Terminal | `python -m signalbot.bot` | Run the Telegram bot |
| Terminal | `python -m pytest` | Run the tests (offline, fake data) |
| Telegram | `/status` | Data, sentiment, and feature coverage |
| Telegram | `/refresh` | Pull new data and rebuild features from your phone |
| Telegram | `/ping`, `/help` | Health check, command list |

The bot only answers your chat (`TELEGRAM_CHAT_ID`). Anyone else gets "This bot is private."

## Settings

Edit `config.yaml` to change the watchlist, market-context tickers, which sector ETF each stock is compared to, how far back to pull history, and the news cutoff time.

---

## What the model sees (Phase 2)

One row per stock per trading day. Every column uses only what was public by that day's 4:00 PM ET close.

| Group | Features |
|---|---|
| **Technical** (12) | 1/5/20-day returns, opening gap, RSI-14, MACD histogram, distance from 20/50-day averages, volume vs. 20-day average, ATR %, 20-day volatility, position in 20-day range |
| **Market** (12) | SPY and QQQ moves, VIX level and change, sector ETF move, stock vs. sector and vs. SPY, 10-year yield and its 5-day change, yield curve (10-yr minus 3-month) |
| **News** (9) | Headline count and how unusual it is, average/min/max FinBERT sentiment, share positive/negative, 3-day sentiment, stories focused on this stock |
| **Calendar** (4) | Earnings reaction tomorrow, earnings reaction today, trading days to next earnings, day of week |

**Label:** did the stock close higher the next trading day (1/0). The newest row has no label yet; it's the one the model will predict.

News features are blank (unknown) before the news history starts, and zero on quiet days after it starts, so the model can tell "no data" from "no news."

### How timing is kept honest

- A headline counts toward a trading day only if it was public **before that day's 4:00 PM ET close**. After-hours, overnight, weekend, and holiday news rolls forward. See `signalbot/align.py`.
- The same headline from several outlets is stored once, at the time it first appeared.
- An earnings report after Wednesday's close is flagged on **Wednesday's** row, because Thursday's move is the reaction. The date is on the calendar in advance, so this isn't peeking.
- `tests/test_features.py` multiplies all future prices by 3 and checks that no past feature changes.

## Project layout

```
config.yaml            watchlist, context tickers, sectors, settings
.env.example           template for your keys (copy to .env)
requirements-ml.txt    FinBERT extras
signalbot/
  config.py            loads config.yaml + .env
  db.py                SQLite storage (data/signalbot.db)
  align.py             news → trading-day rules, headline dedupe
  refresh.py           data pipeline command
  sources/             Yahoo prices, Alpaca + Finnhub news, Finnhub earnings
  sentiment.py         FinBERT headline scoring
  features.py          builds the feature table
  bot.py               Telegram bot
  chat_id.py           finds your Telegram chat ID
  notify.py            one-off Telegram messages
tests/                 runs offline with fake data
```

## Roadmap

1. ~~Data pipeline~~
2. **Features & sentiment** ← you are here
3. First model: logistic regression → LightGBM, calibrated probabilities
4. Honest backtest: walk-forward testing, leakage checks, scorecard vs. the base rate
5. Paper trading: nightly run, `/today`, `/odds`, `/log`, `/record` commands, 3+ months of logged calls
6. Report & polish: AI news summaries, dashboard, portfolio write-up

## Ground rules

- No input may come from after the prediction time.
- The model must beat the base rate ("always guess up," usually ~52–54%) to count as an edge.
- 70%+ accuracy in a backtest means a bug, not a breakthrough.
- 3+ months of paper trading before any real money.
- Low multipliers only; never a setting where breakeven is above 55%.
- Risk 1–2% of bankroll per trade; stop for the day after 3 losses.
- "No call" is a valid answer.
