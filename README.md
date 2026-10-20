# Tr3nch3z — Signal Bot

A Telegram bot built for Trench.io-style trades: **up or down, held 1–5 minutes**, on stocks, crypto, and commodities. Ask `/call NVDA` and it answers **UP / DOWN / No trade** with a hold time and the reasons, using live minute prices, volume, market mood, and fresh headlines. It only makes calls in setups that held up in an honest backtest at your multiplier and fee.

> Not financial advice. This is a learning project. 1–5 minute moves are mostly noise, so "No trade" is often the right answer.

## Trench mode (start here)

```bash
python -m signalbot.intraday refresh       # minute bars + news for every Trench market (first run: a few minutes)
python -m signalbot.sentiment              # score the new headlines
python -m signalbot.intraday backtest      # test every market type x hold time, then train the live models (~5-15 min)
python -m signalbot.bot                    # then in Telegram: /call NVDA, /call btc, /scan, /tested
```

Rerun `refresh` + `backtest` about once a week so the models keep up with the market.

**Set up for your trades in `config.yaml` → `trench:`**

| Setting | What it is |
|---|---|
| `multiplier` | the multiplier you use; busts and per-trade results are computed at it |
| `fee_bps` | Trench's fee per trade as basis points of the position. From a ticket: fee $ ÷ (stake $ × multiplier) × 10,000. Until it's set, the backtest shows the max fee each setup could survive. |
| `entry_delay_min` | minutes between the alert and you placing the trade (default 1) |
| `holds` | hold times the bot picks between (default 1, 2, 3, 5 min) |
| `markets` | stocks, crypto (24/7), and commodities via ETFs that track them (GLD gold, SLV silver, USO oil; US market hours only) |

**What a call looks like**

```
NVDA ⬆️ UP · hold 3 min
   57% confident · enter now, close after 3 min
   tested: right 55%, avg +1.4 bps, 0.0% busts at 25x, 12.0 calls/day
   • volume vs normal 3.2x
   • last 5 min +0.41%
   • vs market (5 min) +0.30%
   • news: "Nvidia supplier raises AI outlook" (4 min ago, tone +0.81)
   price 182.40 at 1:52 PM CT
```

**How it stays honest**
- Every backtest trade enters **1 minute after** the alert, like you would.
- Quarterly-style walk-forward by week: each week is predicted by models trained only on the 6 weeks before it, with a gap.
- The confidence bar for each market type and hold time is **picked on the first half** of the test weeks and **judged on the second half**. Without this, random data looked tradeable.
- A setup is 🟢 only if its AUC is above 0.505, the low end of its 90% range beats your fee, and its average result per trade at your multiplier (busts included) is positive.
- `tests/test_intraday.py`: a planted momentum edge is found; random data never gets a 🟢.

---

The daily model below (next-day direction, Phases 1–4) still runs and feeds `/today`.

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
python -m signalbot.model         # 4. check the model, retrain, predict tomorrow
python -m signalbot.bot           # run the Telegram bot (leave it running)
```

Rerun the backtest whenever you change features or settings (takes 1–3 minutes):

```bash
python -m signalbot.backtest
```

Run steps 1–4 **after the market closes** (4:15 PM ET / 3:15 PM Central). Before that, today's prices are still moving and the odds aren't final. Or just send `/refresh` in Telegram, which runs all four.

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
| Terminal | `python -m signalbot.model` | Held-out check, retrain, store tomorrow's odds |
| Terminal | `python -m signalbot.backtest` | Walk-forward backtest of all four questions |
| Terminal | `python -m signalbot.bot` | Run the Telegram bot |
| Terminal | `python -m pytest` | Run the tests (offline, fake data) |
| Telegram | `/call NVDA` | Trench call: UP / DOWN / No trade, hold time, reasons |
| Telegram | `/scan` | Every Trench market right now |
| Telegram | `/tested` | Which market types and hold times have a tested edge |
| Telegram | `/today` | Tomorrow's odds for the whole watchlist |
| Telegram | `/odds NVDA` | One stock's odds plus its key numbers |
| Telegram | `/backtest` | Which questions show a real out-of-sample signal |
| Telegram | `/status` | Data, sentiment, and feature coverage |
| Telegram | `/refresh` | Pull data, score news, rebuild features, rerun the model |
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
| **Peer rank** (7) | Where the stock ranks against the rest of the watchlist that day on 1/5/20-day return, RSI, distance from 20-day average, volume, and ATR |
| **Calendar** (4) | Earnings reaction tomorrow, earnings reaction today, trading days to next earnings, day of week |

**Label:** did the stock close higher the next trading day (1/0). The newest row has no label yet; it's the one the model will predict.

News features are blank (unknown) before the news history starts, and zero on quiet days after it starts, so the model can tell "no data" from "no news."

### How timing is kept honest

- A headline counts toward a trading day only if it was public **before that day's 4:00 PM ET close**. After-hours, overnight, weekend, and holiday news rolls forward. See `signalbot/align.py`.
- The same headline from several outlets is stored once, at the time it first appeared.
- An earnings report after Wednesday's close is flagged on **Wednesday's** row, because Thursday's move is the reaction. The date is on the calendar in advance, so this isn't peeking.
- `tests/test_features.py` multiplies all future prices by 3 and checks that no past feature changes.

## The model (Phase 3)

`python -m signalbot.model` does five things:

1. **Holds out** the last 120 trading days. The model never trains on them.
2. **Trains two price + market models** (logistic regression and gradient-boosted trees) on everything before that, with 5-day gaps so nothing bleeds across, then **calibrates** them on a separate 120-day window so "60%" means up about 60% of the time.
3. **Tests a news add-on** on top of the better model, trained only on days with news. It's switched on only if it beats the plain model in at least 90% of bootstrap resamples of the test days, so it can't win by luck.
4. **Shuffle check:** retrains with random labels. That version must not beat "always guess the common direction," or something is leaking.
5. **Retrains the winner on all data** and stores tomorrow's odds for `/today` and `/odds`.

How to read the report:

| Column | Meaning |
|---|---|
| accuracy | how often the model picked the right direction on held-out days |
| always up | how often just guessing "up" was right in that window (the base rate) |
| edge | accuracy minus the better of always-up / always-down, in points. **This is the number that matters.** |
| vs base | log-loss improvement over predicting the base rate every day; positive is better |
| calls / call acc | predictions at 58%+ or 42%- and how often those were right |

A single 120-day window is a small sample: a few points of edge either way can be luck. Phase 4 replays every past year to find out.

## The backtest (Phase 4)

`python -m signalbot.backtest` replays history quarter by quarter from 2021. For each quarter it trains only on earlier days (with a 5-day gap), calibrates on the latest 120 of them, and predicts the quarter. Every number in the report is out-of-sample.

It tests four questions, because some are much easier than "up or down":

| Question | Label | Why it might work |
|---|---|---|
| `direction` | closes higher tomorrow | the original question; the hardest |
| `beat_peers` | beats the watchlist median tomorrow | removes the market's daily swing, which is most of the noise |
| `big_move` | moves more than its 1-year median daily move, either way | volatility clusters, so move *size* is far more predictable than direction; matters for multiplier trades |
| `direction_5d` | higher 5 trading days from now | slower signals have time to play out |

How it decides:
- **Baseline** = the up-rate of the most recent 120 days, the same days the model's calibration sees. Beating it means skill, not just following the trend.
- **SIGNAL** = better probabilities (log loss) than the baseline in 95%+ of day-level bootstrap resamples, and AUC above 0.51. "Weak" = 80–95%.
- The 5-day question resamples whole weeks, because its labels overlap.
- **News** gets its own test on the days that have news: each quarter's news add-on learns only from earlier quarters' out-of-sample predictions.
- `beat_peers` also reports a **top-2 minus bottom-2 spread**: each day, the average next-day return of the model's 2 top picks minus its 2 bottom picks, before costs.

Sanity checks in `tests/test_backtest.py`: a planted signal is found, pure noise is never called a signal, and no fold trains on its test window or the 5-day gap before it.

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
  model.py             trains, checks, and runs the model
  backtest.py          walk-forward backtest of four questions (daily)
  intraday.py          Trench mode: minute features, 1-5 min backtest, live calls
  sources/bars.py      Alpaca minute bars (stocks + crypto)
  bot.py               Telegram bot
  chat_id.py           finds your Telegram chat ID
  notify.py            one-off Telegram messages
tests/                 runs offline with fake data
```

## Roadmap

1. ~~Data pipeline~~
2. ~~Features & sentiment~~
3. ~~First model~~
4. **Honest backtest** ← you are here: walk-forward testing, leakage checks, scorecard vs. the base rate
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
