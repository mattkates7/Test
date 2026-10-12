"""Signal generation (research signal).

Two strategy types (config `strategy.type`):
  * "temporal_window" (default): trade the start of a time window.
  * "breakout": trend-following channel breakout with a trailing stop.

Notes for the temporal-window type:

A signal fires at the OPEN of the first candle of the configured window
(e.g. 14:00 UTC for hour_of_day=[14]). It only uses candles that have already
closed, so there is no look-ahead.

The signal PASSES only if:
  * the window was a CANDIDATE in the research run on the training period
    (unless strategy.require_research_candidate is false = exploration mode)
  * the previous candle was not unusually large (no chasing)
"""

from dataclasses import dataclass

from .temporal_research import DAY_NAMES


@dataclass
class Signal:
    index: int
    ts: object
    window: str
    direction: str
    passes: bool
    reason: str


def window_name(ts, strat, sessions):
    """Return the window label if candle `ts` is the START of a configured window."""
    wtype, values = strat["window_type"], strat["window_values"]
    if wtype == "hour_of_day" and ts.hour in values:
        return f"hour_{ts.hour:02d}"
    if wtype == "day_of_week" and ts.hour == 0 and DAY_NAMES[ts.weekday()] in values:
        return f"day_{DAY_NAMES[ts.weekday()]}"
    if wtype == "session":
        for name in values:
            if ts.hour == sessions[name][0]:
                return f"session_{name}"
    return None


def breakout_setup(candles, i, strat):
    """Trend-following breakout, decided at the OPEN of candle i from closed candles.

    Long: the previous candle CLOSED above the highest high of the
    `channel_bars` candles before it (and above the `trend_sma_bars` average,
    if set). Short is the mirror image.
    """
    n, sma_n = strat["channel_bars"], strat.get("trend_sma_bars", 0)
    if i - 1 - max(n, sma_n) < 0:
        return None
    prev = candles[i - 1]
    window = candles[i - 1 - n : i - 1]
    if strat["direction"] == "long":
        broke = prev.close > max(c.high for c in window)
        trend_ok = not sma_n or prev.close > sum(c.close for c in candles[i - 1 - sma_n : i - 1]) / sma_n
    else:
        broke = prev.close < min(c.low for c in window)
        trend_ok = not sma_n or prev.close < sum(c.close for c in candles[i - 1 - sma_n : i - 1]) / sma_n
    return breakout_name(strat) if broke and trend_ok else None


def breakout_name(strat):
    return f"breakout_{strat['channel_bars']}bar"


def configured_window_names(strat):
    if strat.get("type") == "breakout":
        return [breakout_name(strat)]
    fmt = {
        "hour_of_day": lambda v: f"hour_{int(v):02d}",
        "day_of_week": lambda v: f"day_{v}",
        "session": lambda v: f"session_{v}",
    }[strat["window_type"]]
    return [fmt(v) for v in strat["window_values"]]


def generate_signal(candles, i, atr, strat, sessions, research_verdicts):
    """Return a Signal for candle i, or None if there is no setup at candle i."""
    if strat.get("type") == "breakout":
        name = breakout_setup(candles, i, strat)
    else:
        name = window_name(candles[i].ts, strat, sessions)
    if name is None:
        return None
    direction = strat["direction"]
    make = lambda ok, why: Signal(i, candles[i].ts, name, direction, ok, why)  # noqa: E731

    verdict = research_verdicts.get(name, "NOT TESTED")
    require = strat.get("require_research_candidate", True)
    if require and not verdict.startswith("CANDIDATE"):
        return make(False, f"research verdict for {name} on training data: {verdict}")

    prev = candles[i - 1]
    prev_atr = atr[i - 1]
    if prev_atr is None:
        return make(False, "not enough history for ATR")
    candle_size = (prev.high - prev.low) / prev_atr
    if candle_size > strat["max_entry_candle_atr_multiple"]:
        return make(False, f"no chasing: previous candle was {candle_size:.1f}x ATR")

    label = "research candidate" if verdict.startswith("CANDIDATE") else "EXPLORATION (no research edge)"
    return make(True, f"{name} window open; {label}; research verdict: {verdict}")
