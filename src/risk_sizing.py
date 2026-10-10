"""Paper Risk Sizing (Step 3). PAPER MODE ONLY - this is a simulation, not financial advice.

Rules enforced here:
  * Never risk more than max_risk_per_trade_pct (hard ceiling 1%) of the paper account.
  * Never use leverage: position value can never exceed the paper account balance.
  * Never increase the risk fraction right after a loss.
  * Kelly is only an input. It is multiplied down (e.g. quarter-Kelly) and then
    capped by the 1% rule, so it can never push risk above 1%.
  * Reject when inputs are unreliable or the sample size is too small.
"""

from dataclasses import dataclass, field

from .config import HARD_MAX_RISK_PCT


@dataclass
class SizingResult:
    decision: str  # "APPROVE" or "REJECT"
    max_dollar_risk: float = 0.0
    raw_kelly: float | None = None
    paper_fraction: float = 0.0
    position_size: float = 0.0
    position_value: float = 0.0
    dollar_risk: float = 0.0
    reasons: list = field(default_factory=list)
    math: list = field(default_factory=list)

    @property
    def approved(self):
        return self.decision == "APPROVE"

    def report(self):
        lines = ["PAPER RISK SIZING (simulation, not financial advice)"]
        lines += [f"  {m}" for m in self.math]
        lines.append(f"DECISION: {self.decision}")
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def size_position(
    account_size,
    win_probability,
    avg_win,
    avg_loss,
    entry,
    stop,
    sample_size,
    risk_cfg,
    previous_fraction=None,
    previous_was_loss=False,
):
    """Return a SizingResult. `avg_win` and `avg_loss` are positive numbers in the
    same unit (dollars or R-multiples); only their ratio is used."""
    res = SizingResult(decision="REJECT")
    max_pct = min(risk_cfg["max_risk_per_trade_pct"], HARD_MAX_RISK_PCT) / 100
    min_n = risk_cfg["min_sample_size"]

    # ---- 1. Input checks -------------------------------------------------
    problems = []
    if account_size <= 0:
        problems.append("account size must be positive")
    if not 0 < win_probability < 1:
        problems.append("win probability must be strictly between 0 and 1")
    if avg_win <= 0 or avg_loss <= 0:
        problems.append("average win and average loss must both be positive")
    if entry <= 0 or stop <= 0:
        problems.append("entry and stop must be positive prices")
    if entry == stop:
        problems.append("stop equals entry - there is no defined risk")
    if sample_size < min_n:
        problems.append(f"sample size {sample_size} is below the minimum {min_n}; estimate unreliable")
    if problems:
        res.reasons = problems
        res.math.append("Inputs rejected before any math was done.")
        return res

    # ---- 2. Max dollar risk at the % cap ----------------------------------
    res.max_dollar_risk = account_size * max_pct
    res.math.append(
        f"1) Max dollar risk = ${account_size:,.2f} x {max_pct:.2%} = ${res.max_dollar_risk:,.2f}"
    )

    # ---- 3. Raw Kelly -----------------------------------------------------
    p, q = win_probability, 1 - win_probability
    b = avg_win / avg_loss
    kelly = p - q / b
    res.raw_kelly = kelly
    res.math.append(
        f"2) Payoff ratio b = avg win / avg loss = {avg_win:.4f} / {avg_loss:.4f} = {b:.3f}"
    )
    res.math.append(f"   Raw Kelly f* = p - (1-p)/b = {p:.3f} - {q:.3f}/{b:.3f} = {kelly:.4f}")
    if kelly <= 0:
        res.reasons.append(
            f"Kelly is {kelly:.4f} (<= 0): with these numbers the expected value is not positive"
        )
        return res

    # ---- 4. Heavily capped paper fraction ---------------------------------
    mult = risk_cfg["kelly_multiplier"]
    fraction = min(kelly * mult, max_pct)
    note = f"min({kelly:.4f} x {mult}, {max_pct:.2%})"
    if previous_was_loss and previous_fraction is not None and fraction > previous_fraction:
        fraction = previous_fraction
        note += f", then held at previous {previous_fraction:.4%} (no risk increase after a loss)"
    res.paper_fraction = fraction
    res.math.append(f"3) Paper fraction = {note} = {fraction:.4%} of account")

    # ---- 5. Position size from stop distance ------------------------------
    stop_distance = abs(entry - stop)
    dollar_risk = account_size * fraction
    size = dollar_risk / stop_distance
    res.math.append(
        f"4) Position size = dollar risk / stop distance = ${dollar_risk:,.2f} / "
        f"${stop_distance:,.2f} = {size:.6f} units"
    )
    value = size * entry
    if value > account_size:  # would need leverage -> shrink
        size = account_size / entry
        dollar_risk = size * stop_distance
        res.math.append(
            f"   Position value ${value:,.2f} > account (leverage needed). "
            f"Shrunk to {size:.6f} units (${account_size:,.2f}); risk now ${dollar_risk:,.2f}"
        )
        value = size * entry
    res.position_size = size
    res.position_value = value
    res.dollar_risk = dollar_risk
    res.math.append(f"   Position value = {size:.6f} x ${entry:,.2f} = ${value:,.2f} (no leverage)")
    res.decision = "APPROVE"
    res.reasons.append(
        f"risk ${dollar_risk:,.2f} <= cap ${res.max_dollar_risk:,.2f}, no leverage, sample n={sample_size}"
    )
    return res
