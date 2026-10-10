# BTC & Prediction-Market Paper-Trading Research

A beginner-friendly research project for testing trading ideas **on paper only**.

> **PAPER MODE ONLY.** This project has no exchange or wallet code. It cannot send
> real orders. If `config.json` or `.env` asks for anything other than paper mode,
> or `.env` has wallet keys or exchange secrets in it, the program refuses to run.
> To trade live you would have to rewrite the project yourself. Nothing here is
> financial advice.

## What it does

| Step | What | Module |
|---|---|---|
| 2 | **Temporal bias research**: checks whether hours of the day, days of the week, trading sessions, or the hours after large moves behaved differently | `src/temporal_research.py` |
| 3 | **Risk sizing**: Kelly-style sizing that is cut down and capped at 1% risk, with no leverage | `src/risk_sizing.py` |
| 4 | **Volatility filter**: labels conditions CLEAN, ELEVATED or EXTREME and blocks trades | `src/volatility_filter.py` |
| 5 | **Bayesian updates**: a conservative win-probability estimate that moves slowly | `src/probability_update.py` |
| 6-7 | **Paper execution**: simulated fills, one position at a time, every block logged | `src/signals.py`, `src/execution.py`, `src/backtest.py` |
| 8 | **Prediction-market research**: kept separate from BTC, and skips when evidence is weak or stale | `src/prediction_markets.py` |
| 9 | **Tracking**: a journal spreadsheet plus JSON logs of every trade and every blocked trade | `src/trade_logger.py` |
| - | Terminal dashboard | `src/dashboard.py` |
| - | Paper-mode guard | `src/safety.py` |

```
/src        code (each job in its own module)
/data       historical candles + your prediction-market notes
/logs       append-only logs: every OPEN, CLOSE and BLOCKED event
/results    research tables, journal spreadsheet, backtest summary
/prompts    copy-paste agent prompts (optional, for use with an AI chat)
/tests      checks for the safety and risk rules
config.json risk limits and strategy settings (edit this, not the code)
.env.example template for your .env (no secrets needed)
```

## Setup

You need **Python 3.10 or newer**. No extra packages are needed; it only uses the standard library.

```bash
# 1. Get the code and enter the folder
cd Test

# 2. (Optional but recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Create your .env from the template (leave TRADING_MODE=paper)
cp .env.example .env

# 4. Check everything works
python -m unittest discover -s tests
```

## Quick start

```bash
# Get REAL historical BTC data (public Coinbase candles, no account or key needed)
python -m src.main fetch-data              # default: last 730 days of 1h candles

#   ...or, if you're offline, make SYNTHETIC random data to test the plumbing:
#   python -m src.main make-sample-data

python -m src.main research                # Step 2: temporal bias table
python -m src.main backtest                # Steps 3-7 + 9: walk-forward paper backtest
python -m src.main summary                 # dashboard
```

Other commands:

```bash
# Step 3: sizing calculator, shows the math and APPROVE / REJECT
python -m src.main size --account 10000 --win-prob 0.45 --avg-win 300 --avg-loss 200 \
                        --entry 62000 --stop 61000 --sample-size 150

# Step 4: volatility check on the latest candle in your data
python -m src.main vol-check

# Step 5: Bayesian update for a batch of 6 wins and 4 losses
python -m src.main update --wins 6 --losses 4 --avg-win 1.5 --avg-loss 1

# Step 8: prediction-market paper research (edit the JSON file first)
python -m src.main predict --input data/prediction_markets.example.json

# Always refuses:
python -m src.main live
```

If you use your own CSV, give it the columns `timestamp,open,high,low,close,volume` with
hourly candles. `timestamp` can be unix seconds or ISO format, in UTC. Then point
`data.csv_path` in `config.json` at it.

## How the backtest avoids fooling you

1. **Train/test split.** Research and the starting estimates use only the first part
   of history (`backtest.train_fraction`). Paper trades run only on the later part,
   which the research never saw.
2. **Research gate.** By default a window is traded only if research on the training
   data rated it a `CANDIDATE`. Set `strategy.require_research_candidate` to `false` for
   "exploration mode", which tests the mechanics anyway. Those trades are labelled
   `EXPLORATION (no research edge)`.
3. **Multiple-testing correction.** About 37 windows get tested, so a couple will look
   "significant" by luck. The bar is raised to match (Bonferroni), and a window must
   point the same way in every date range.
4. **Never "a real edge".** The best label is `CANDIDATE (needs out-of-sample paper test)`.
   If nothing qualifies, the result is **NO EDGE FOUND**.
5. **Conservative prior.** The win-probability estimate starts at the *breakeven* win rate
   (no edge). The backtest gets only partial credit (`probability.backtest_weight`).
6. **Pessimistic fills.** Fees and slippage are charged on every fill. A gap through the
   stop fills at the open. If the stop and the target are both inside one candle, the
   stop is assumed to have hit first.
7. **No look-ahead.** Signals and filters only use candles that had already closed.

A sanity check: on the synthetic random-walk data, research should say
**NO EDGE FOUND** and the backtest should block every trade. If it finds an edge in
random data, something is overfit.

## The rules (enforced in code)

- Paper mode only. `mode` must be `"paper"`, `TRADING_MODE` must be `paper`, and no wallet or exchange secrets may be set.
- Maximum 1% account risk per trade. Config can go **lower**; values above 1% are refused.
- No leverage. Position value can never exceed the paper balance.
- One open paper position at a time.
- Every trade has a stop before entry.
- Risk is never increased right after a loss.
- Kelly is multiplied down (`kelly_multiplier`, default quarter-Kelly) and then capped at the 1% rule.
- No chasing. No entry right after a candle larger than `max_entry_candle_atr_multiple` x ATR.
- No trade if the volatility filter says NO. ELEVATED is blocked by default (`allow_elevated: false`), and so are conditions outside the backtest's range.
- No trade if the edge estimate rests on fewer than `risk.min_sample_size` trades.
- Trading stops if the Bayesian estimate shows the edge is probably gone (`keep_testing = NO`).
- Every fill, close, loss and **blocked/skipped** trade is logged.

## Changing settings (`config.json`)

| Setting | Meaning |
|---|---|
| `account.starting_balance` | Paper account size |
| `risk.max_risk_per_trade_pct` | % of account risked per trade (max 1.0) |
| `risk.kelly_multiplier` | Fraction of Kelly used (0.25 = quarter Kelly) |
| `risk.min_sample_size` | Minimum trades behind an estimate before trading |
| `strategy.window_type` / `window_values` | e.g. `"hour_of_day"` + `[14]`, `"day_of_week"` + `["Sat"]`, `"session"` + `["europe"]` |
| `strategy.direction` | `"long"` or `"short"` (short is simulated only) |
| `strategy.stop_atr_multiple`, `reward_to_risk`, `max_hold_bars` | Stop distance, target, time exit |
| `volatility_filter.*` | Thresholds for ELEVATED / EXTREME |
| `probability.*` | Prior strength, batch size, when to stop testing |
| `research.*` | Holding period, large-move threshold, number of date-range splits, sessions (UTC) |
| `backtest.*` | Train fraction, fees and slippage (in %) |
| `prediction_markets.*` | Staleness limit, minimum edge, fees |

## Outputs

- `results/paper_trading_journal.csv` is the Step 9 spreadsheet. The account summary is at the top (starting and current balance, total trades, win rate, average win and loss, largest drawdown, total P&L). Below it is one row per trade or blocked trade, with every column from the plan. It opens in Excel, Google Sheets or Numbers.
- `results/temporal_research.csv` / `.json` hold the research table.
- `results/backtest_summary.json` holds the summary plus every probability update.
- `logs/paper_trades.jsonl` is append-only. It has one line per OPEN, CLOSE and BLOCKED event, tagged with the run ID.
- `logs/probability_updates.jsonl` and `logs/prediction_market_paper.jsonl` are append-only too.

## Prediction markets

Prediction-market research is separate from BTC. Copy `data/prediction_markets.example.json`.
The examples in it are **fictional**. Fill in real markets with the evidence you found yourself.
Each piece of evidence needs a `date` and a `quality` (`strong`, `moderate` or `weak`).
Give your estimate as a **range**. The tool:

- ignores evidence older than `max_source_age_days`
- returns SKIP if there is no fresh strong or moderate evidence, or no written resolution criteria
- gives a paper YES or NO only if your *whole* range beats the market price by more than `min_edge_pct + fee_pct`
- never claims certainty, and returns SKIP on LOW confidence

A market price is not wrong just because you, or an AI, disagree with it.

## Security notes

- `.env` is git-ignored. The project needs **no** secrets. Never put wallet private keys,
  seed phrases or exchange API secrets in it; the program will refuse to start if you do.
- The only network call is `fetch-data`, which reads public candle data.
  A test checks that no other module imports networking code.
