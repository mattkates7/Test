# Copy-paste prompts

These are the agent prompts from the project plan. Each one has a matching Python
module that applies the same rules with code. You can also paste a prompt into an
AI chat to get a second opinion.

| Prompt | Code that applies the same rules | CLI |
|---|---|---|
| `01_temporal_bias_research.md` | `src/temporal_research.py` | `python -m src.main research` |
| `02_paper_risk_sizing.md` | `src/risk_sizing.py` | `python -m src.main size ...` |
| `03_volatility_filter.md` | `src/volatility_filter.py` | `python -m src.main vol-check` |
| `04_probability_update.md` | `src/probability_update.py` | `python -m src.main update ...` |
| `05_paper_execution.md` | `src/execution.py` + `src/backtest.py` | `python -m src.main backtest` |
| `06_prediction_market_research.md` | `src/prediction_markets.py` | `python -m src.main predict` |

An AI model's opinion is not evidence. Never paste in API keys, wallet keys or account details.
