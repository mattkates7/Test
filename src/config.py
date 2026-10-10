"""Load config.json and .env, and validate the risk limits.

You can change any number in config.json without touching code, but some
limits have hard ceilings that only a code rewrite can raise:
  * max_risk_per_trade_pct can never be above 1.0
  * leverage must be exactly 1.0 (no leverage)
  * max_open_positions must be exactly 1
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HARD_MAX_RISK_PCT = 1.0


class ConfigError(ValueError):
    pass


def load_env(path=PROJECT_ROOT / ".env"):
    """Tiny .env reader (KEY=VALUE lines) so no extra package is needed.

    Only the project's .env file is read, plus TRADING_MODE from the shell
    (so `TRADING_MODE=live python -m src.main ...` is also caught and refused).
    """
    values = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    if "TRADING_MODE" in os.environ:
        values["TRADING_MODE"] = os.environ["TRADING_MODE"]
    return values


def load_config(path=PROJECT_ROOT / "config.json"):
    with open(path) as f:
        config = json.load(f)
    validate_config(config)
    return config


def validate_config(config):
    risk = config["risk"]
    if not 0 < risk["max_risk_per_trade_pct"] <= HARD_MAX_RISK_PCT:
        raise ConfigError(
            f"risk.max_risk_per_trade_pct must be above 0 and at most {HARD_MAX_RISK_PCT} "
            f"(got {risk['max_risk_per_trade_pct']})."
        )
    if risk["leverage"] != 1.0:
        raise ConfigError("risk.leverage must be 1.0. Leverage is not allowed in this project.")
    if risk["max_open_positions"] != 1:
        raise ConfigError("risk.max_open_positions must be 1 (one paper position at a time).")
    if not 0 < risk["kelly_multiplier"] <= 1:
        raise ConfigError("risk.kelly_multiplier must be between 0 and 1.")
    if risk["min_sample_size"] < 30:
        raise ConfigError("risk.min_sample_size below 30 is too small to mean anything.")
    if not risk.get("never_increase_risk_after_loss", True):
        raise ConfigError("risk.never_increase_risk_after_loss must stay true.")

    strat = config["strategy"]
    if strat["direction"] not in ("long", "short"):
        raise ConfigError('strategy.direction must be "long" or "short".')
    if strat["stop_atr_multiple"] <= 0 or strat["reward_to_risk"] <= 0:
        raise ConfigError("strategy.stop_atr_multiple and reward_to_risk must be positive.")
    if strat["window_type"] not in ("hour_of_day", "day_of_week", "session"):
        raise ConfigError('strategy.window_type must be "hour_of_day", "day_of_week" or "session".')

    bt = config["backtest"]
    if not 0.2 <= bt["train_fraction"] <= 0.8:
        raise ConfigError("backtest.train_fraction should be between 0.2 and 0.8.")


def resolve_path(relative):
    return PROJECT_ROOT / relative
