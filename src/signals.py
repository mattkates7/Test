"""Signal generation (research signal) for the temporal-window strategy.

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


def configured_window_names(strat):
    fmt = {
        "hour_of_day": lambda v: f"hour_{int(v):02d}",
        "day_of_week": lambda v: f"day_{v}",
        "session": lambda v: f"session_{v}",
    }[strat["window_type"]]
    return [fmt(v) for v in strat["window_values"]]


def generate_signal(candles, i, atr, strat, sessions, research_verdicts):
    """Return a Signal for candle i, or None if i is not the start of a window."""
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
