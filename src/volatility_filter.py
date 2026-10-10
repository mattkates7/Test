"""Volatility Filter (Step 4) for the PAPER trading system.

Classifies current conditions as CLEAN / ELEVATED / EXTREME and decides
TRADE ALLOWED: YES / NO. It looks at:
  * recent realized volatility, ranked against the reference (backtest) period
  * sudden candle expansion (last candle range vs ATR)
  * abnormal volume (z-score of log volume)
  * distance of the last move from the recent average move
  * whether current volatility is outside anything seen in the reference period

Default is NO whenever conditions are materially outside what the strategy was
tested on. Only closed candles are used, so there is no look-ahead.
"""

import math
import statistics
from dataclasses import dataclass

from .indicators import atr_series, log_returns, percentile_rank, rolling_std


@dataclass
class VolatilityReading:
    condition: str  # CLEAN / ELEVATED / EXTREME
    trade_allowed: bool
    reason: str
    realized_vol_pct: float | None = None
    vol_percentile: float | None = None

    def report(self):
        return (
            f"CONDITION: {self.condition}\n"
            f"TRADE ALLOWED: {'YES' if self.trade_allowed else 'NO'}\n"
            f"REASON: {self.reason}"
        )


class VolatilityFilter:
    def __init__(self, candles, cfg, atr_period):
        """Precompute indicators over all candles. `reference_end` (via
        set_reference) marks the backtest/training period the strategy was
        tested on."""
        self.cfg = cfg
        self.candles = candles
        self.lookback = cfg["lookback_bars"]
        self.rets = log_returns(candles)
        self.rv = rolling_std(self.rets, self.lookback)
        self.atr = atr_series(candles, atr_period)
        self.ref_sorted = []
        self.ref_min = self.ref_max = None

    def set_reference(self, start, end):
        ref = sorted(v for v in self.rv[start:end] if v is not None)
        if not ref:
            raise ValueError("Reference period too short for the volatility filter.")
        self.ref_sorted = ref
        self.ref_min, self.ref_max = ref[0], ref[-1]

    def classify(self, i):
        """Classify conditions using candles up to and including index i (closed)."""
        c = self.cfg
        rv, atr = self.rv[i], self.atr[i]
        if rv is None or atr is None or i < self.lookback:
            return VolatilityReading("EXTREME", False, "not enough history to judge volatility")

        reasons_extreme, reasons_elevated = [], []
        pct = percentile_rank(self.ref_sorted, rv)

        # Outside the tested range entirely -> default NO
        if rv > self.ref_max * 1.05:
            reasons_extreme.append("realized vol is above anything in the backtest period")
        elif rv < self.ref_min * 0.95:
            reasons_elevated.append("realized vol is below anything in the backtest period (untested)")

        if pct >= c["extreme_percentile"]:
            reasons_extreme.append(f"realized vol in the top {max(1, 100 - pct):.0f}% of the backtest period")
        elif pct >= c["elevated_percentile"]:
            reasons_elevated.append(f"realized vol in the top {max(1, 100 - pct):.0f}% of the backtest period")

        candle = self.candles[i]
        expansion = (candle.high - candle.low) / atr if atr > 0 else 0
        if expansion >= c["candle_expansion_extreme"]:
            reasons_extreme.append(f"last candle range {expansion:.1f}x ATR")
        elif expansion >= c["candle_expansion_elevated"]:
            reasons_elevated.append(f"last candle range {expansion:.1f}x ATR")

        vols = [math.log1p(x.volume) for x in self.candles[i - self.lookback : i]]
        sd = statistics.pstdev(vols)
        vz = (math.log1p(candle.volume) - statistics.fmean(vols)) / sd if sd > 0 else 0
        if vz >= c["volume_z_extreme"]:
            reasons_extreme.append(f"volume z-score {vz:.1f}")
        elif vz >= c["volume_z_elevated"]:
            reasons_elevated.append(f"volume z-score {vz:.1f}")

        recent = [abs(r) for r in self.rets[i - self.lookback : i]]
        avg_move = statistics.fmean(recent)
        move_ratio = abs(self.rets[i]) / avg_move if avg_move > 0 else 0
        if move_ratio >= c["move_vs_average_elevated"]:
            reasons_elevated.append(f"last move {move_ratio:.1f}x the recent average move")

        reading_kwargs = {"realized_vol_pct": 100 * rv, "vol_percentile": pct}
        if reasons_extreme:
            return VolatilityReading("EXTREME", False, "; ".join(reasons_extreme + reasons_elevated), **reading_kwargs)
        if reasons_elevated:
            untested = any("untested" in r for r in reasons_elevated)
            allowed = c["allow_elevated"] and not untested
            return VolatilityReading("ELEVATED", allowed, "; ".join(reasons_elevated), **reading_kwargs)
        return VolatilityReading(
            "CLEAN", True, f"realized vol at percentile {pct:.0f} of the backtest period, no expansion or volume spike", **reading_kwargs
        )
