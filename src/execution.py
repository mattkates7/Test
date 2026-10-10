"""Paper Execution (Steps 6 & 7). YOU CANNOT PLACE REAL TRADES WITH THIS.

There is no broker or exchange client here. 'Opening a trade' means creating
a Python object and writing it to a log file. Fills are simulated from
historical candles with slippage and fees.

A simulated trade is only created if ALL of these hold:
  * the research signal passes
  * the volatility filter allows it
  * the risk agent approves it
  * no paper position is already open
  * the probability agent has not said to stop testing
Otherwise the result is TRADE BLOCKED, with the reasons, and it is logged.
"""

from dataclasses import dataclass, field

from .safety import LIVE_TRADING_ENABLED, refuse_live_execution


@dataclass
class PaperTrade:
    trade_id: str
    opened_at: object
    asset: str
    setup: str
    direction: str
    entry: float
    stop: float
    target: float
    size: float
    max_dollar_risk: float
    est_probability: float
    confidence: str
    volatility: str
    entry_reason: str
    entry_index: int
    status: str = "OPEN"
    exit: float | None = None
    closed_at: object = None
    exit_reason: str = ""
    pnl: float = 0.0
    fees: float = 0.0
    outcome: str = ""
    learned: str = ""
    extra: dict = field(default_factory=dict)

    def ticket(self):
        return "\n".join(
            [
                f"PAPER TRADE ID: {self.trade_id}",
                f"DATE/TIME: {self.opened_at:%Y-%m-%d %H:%M} UTC",
                f"ASSET: {self.asset}",
                f"SETUP: {self.setup} ({self.direction})",
                f"SIMULATED ENTRY: {self.entry:,.2f}",
                f"SIMULATED STOP: {self.stop:,.2f}",
                f"SIMULATED TARGET: {self.target:,.2f}",
                f"SIMULATED POSITION SIZE: {self.size:.6f}",
                f"MAX DOLLAR RISK: ${self.max_dollar_risk:,.2f}",
                f"SIGNAL CONFIDENCE: {self.confidence} (est. win prob {self.est_probability:.1%})",
                f"STATUS: {self.status}",
            ]
        )


def blocked_ticket(ts, asset, setup, reasons):
    lines = [f"TRADE BLOCKED  {ts:%Y-%m-%d %H:%M} UTC  {asset}  {setup}"]
    lines += [f"  - {r}" for r in reasons]
    return "\n".join(lines)


class PaperExecutor:
    def __init__(self, asset, fee_pct, slippage_pct):
        if LIVE_TRADING_ENABLED:  # pragma: no cover
            refuse_live_execution()
        self.asset = asset
        self.fee = fee_pct / 100
        self.slip = slippage_pct / 100
        self.position = None
        self._counter = 0

    def send_live_order(self, *args, **kwargs):
        """Exists only to make the point: it always refuses."""
        refuse_live_execution()

    def check_gates(self, signal, vol_reading, sizing, stop_testing_reason):
        reasons = []
        if not signal.passes:
            reasons.append(f"research signal failed: {signal.reason}")
        if not vol_reading.trade_allowed:
            reasons.append(f"volatility filter {vol_reading.condition}: {vol_reading.reason}")
        if sizing is not None and not sizing.approved:
            reasons.append("risk agent REJECT: " + "; ".join(sizing.reasons))
        if self.position is not None:
            reasons.append(f"paper position {self.position.trade_id} already open")
        if stop_testing_reason:
            reasons.append(f"probability agent says stop testing: {stop_testing_reason}")
        return reasons

    def entry_price(self, open_price, direction):
        return open_price * (1 + self.slip) if direction == "long" else open_price * (1 - self.slip)

    def open_trade(self, ts, i, signal, entry, stop, target, sizing, est_p, confidence, vol_reading):
        if self.position is not None:
            raise RuntimeError("One open paper position at a time.")
        if stop is None or stop == entry:
            raise RuntimeError("Every paper trade needs a stop before entry.")
        self._counter += 1
        self.position = PaperTrade(
            trade_id=f"PT-{ts:%Y%m%d%H}-{self._counter:04d}",
            opened_at=ts,
            asset=self.asset,
            setup=signal.window,
            direction=signal.direction,
            entry=entry,
            stop=stop,
            target=target,
            size=sizing.position_size,
            max_dollar_risk=sizing.dollar_risk,
            est_probability=est_p,
            confidence=confidence,
            volatility=vol_reading.condition,
            entry_reason=signal.reason,
            entry_index=i,
        )
        return self.position

    def check_exit(self, candle, i, max_hold_bars):
        """Simulate the open position through candle i. Returns closed trade or None.

        Conservative assumptions: a gap through the stop fills at the open; if
        stop and target are both inside one candle, the STOP is assumed hit first.
        """
        t = self.position
        if t is None:
            return None
        long = t.direction == "long"
        exit_price, reason = None, ""
        gapped = candle.open <= t.stop if long else candle.open >= t.stop
        if i > t.entry_index and gapped:
            exit_price, reason = candle.open, "gap through stop"
        elif (candle.low <= t.stop) if long else (candle.high >= t.stop):
            exit_price, reason = t.stop, "stop hit"
        elif (candle.high >= t.target) if long else (candle.low <= t.target):
            exit_price, reason = t.target, "target hit"
        elif i - t.entry_index + 1 >= max_hold_bars:
            exit_price, reason = candle.close, f"time exit after {max_hold_bars} bars"
        if exit_price is None:
            return None
        if reason != "target hit":  # market-style exits pay slippage; target is a limit
            exit_price = exit_price * (1 - self.slip) if long else exit_price * (1 + self.slip)
        return self._close(t, candle.ts, exit_price, reason)

    def force_close(self, candle, reason="end of data"):
        t = self.position
        long = t.direction == "long"
        price = candle.close * (1 - self.slip) if long else candle.close * (1 + self.slip)
        return self._close(t, candle.ts, price, reason)

    def _close(self, t, ts, price, reason):
        gross = (price - t.entry) * t.size if t.direction == "long" else (t.entry - price) * t.size
        fees = self.fee * t.size * (t.entry + price)
        t.exit, t.closed_at, t.exit_reason = price, ts, reason
        t.fees = fees
        t.pnl = gross - fees
        planned = t.max_dollar_risk or 1e-9
        if abs(t.pnl) <= 0.02 * planned:
            t.outcome = "Breakeven"
        else:
            t.outcome = "Win" if t.pnl > 0 else "Loss"
        t.status = "CLOSED"
        r = t.pnl / planned
        if t.outcome == "Loss" and r < -1.2:
            t.learned = f"Loss {r:.2f}R exceeded planned 1R ({reason}); gaps/slippage are real costs."
        elif t.outcome == "Loss":
            t.learned = f"Loss {r:.2f}R, within planned risk."
        elif t.outcome == "Win":
            t.learned = f"Win {r:.2f}R. One win proves nothing; see probability updates."
        else:
            t.learned = "Breakeven after fees; costs ate the move."
        self.position = None
        return t
