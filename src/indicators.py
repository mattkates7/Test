"""Small shared helpers (ATR, returns, drawdown, percentiles). Pure functions."""

import bisect
import math
import statistics


def true_ranges(candles):
    out = []
    for i, c in enumerate(candles):
        if i == 0:
            out.append(c.high - c.low)
        else:
            prev = candles[i - 1].close
            out.append(max(c.high - c.low, abs(c.high - prev), abs(c.low - prev)))
    return out


def atr_series(candles, period):
    """Simple-moving-average ATR. Value at i uses candles[i-period+1 .. i]."""
    trs = true_ranges(candles)
    out = [None] * len(candles)
    running = 0.0
    for i, tr in enumerate(trs):
        running += tr
        if i >= period:
            running -= trs[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def log_returns(candles):
    return [0.0] + [
        math.log(candles[i].close / candles[i - 1].close) for i in range(1, len(candles))
    ]


def rolling_std(values, window):
    out = [None] * len(values)
    for i in range(window - 1, len(values)):
        out[i] = statistics.pstdev(values[i - window + 1 : i + 1])
    return out


def max_drawdown(returns):
    """Max peak-to-trough drop (as a positive fraction) when compounding `returns`."""
    equity, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1 + r
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return worst


def max_drawdown_from_equity(equity_curve):
    peak, worst = None, 0.0
    for e in equity_curve:
        peak = e if peak is None else max(peak, e)
        if peak > 0:
            worst = max(worst, (peak - e) / peak)
    return worst


def percentile_rank(sorted_values, x):
    """Percent of reference values that are <= x (0-100)."""
    if not sorted_values:
        return 50.0
    return 100.0 * bisect.bisect_right(sorted_values, x) / len(sorted_values)


def quantile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]
