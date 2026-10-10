"""Bayesian-style Probability Updates (Step 5) for the PAPER experiment.

The win probability is tracked as a Beta distribution:
    Beta(alpha, beta)  with  mean = alpha / (alpha + beta)

Start (prior):
    Centred on the BREAKEVEN win rate (the rate at which the strategy makes
    nothing), i.e. we assume NO EDGE until data says otherwise.
    `prior_strength` acts like that many imaginary trades, so a handful of
    real trades can't swing the estimate much.
    Backtest results are only partially trusted (`backtest_weight`, e.g. 25%).

Update:
    alpha += new wins, beta += new losses.
"""

import random
from dataclasses import dataclass


@dataclass
class BetaEstimate:
    alpha: float
    beta: float
    observed_trades: int = 0  # real paper trades only (not prior pseudo-counts)

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def interval(self, level=0.9, draws=20000, seed=0):
        rng = random.Random(seed)
        xs = sorted(rng.betavariate(self.alpha, self.beta) for _ in range(draws))
        lo = xs[int((1 - level) / 2 * draws)]
        hi = xs[int((1 + level) / 2 * draws) - 1]
        return lo, hi

    def prob_above(self, threshold, draws=20000, seed=1):
        rng = random.Random(seed)
        return sum(rng.betavariate(self.alpha, self.beta) > threshold for _ in range(draws)) / draws


def breakeven_win_rate(avg_win, avg_loss):
    """Win rate where expected value is zero: p*W = (1-p)*L  ->  p = L / (W + L)."""
    return avg_loss / (avg_win + avg_loss)


def make_prior(prob_cfg, breakeven, backtest_wins=0, backtest_losses=0):
    prior_mean = breakeven if prob_cfg["prior_mean"] == "breakeven" else float(prob_cfg["prior_mean"])
    k = prob_cfg["prior_strength"]
    w = prob_cfg["backtest_weight"]
    return BetaEstimate(
        alpha=prior_mean * k + w * backtest_wins,
        beta=(1 - prior_mean) * k + w * backtest_losses,
    )


def update(estimate, wins, losses, prob_cfg, breakeven):
    """Apply a batch of outcomes. Returns (new_estimate, report_dict)."""
    prior_mean = estimate.mean
    new = BetaEstimate(estimate.alpha + wins, estimate.beta + losses, estimate.observed_trades + wins + losses)
    lo, hi = new.interval()
    width = hi - lo
    n = new.observed_trades
    p_edge = new.prob_above(breakeven)

    if n < prob_cfg["min_trades_for_medium"] or width > 0.15:
        confidence = "LOW"
    elif n < prob_cfg["min_trades_for_high"] or width > 0.08:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    warnings = []
    if wins + losses < 20:
        warnings.append("This batch is small; do not read anything into it on its own.")
    if n < prob_cfg["min_trades_for_medium"]:
        warnings.append(f"Only {n} paper trades in total; too few for a strong conclusion.")

    keep_testing = True
    stop_reason = ""
    if n >= prob_cfg["min_trades_for_medium"] and p_edge < prob_cfg["stop_if_prob_above_breakeven_below"]:
        keep_testing = False
        stop_reason = f"Only {p_edge:.0%} chance the win rate beats breakeven after {n} trades."

    report = {
        "prior_estimate": round(prior_mean, 4),
        "new_sample_size": wins + losses,
        "new_wins": wins,
        "new_losses": losses,
        "updated_estimate": round(new.mean, 4),
        "credible_interval_90": [round(lo, 4), round(hi, 4)],
        "breakeven_win_rate": round(breakeven, 4),
        "prob_above_breakeven": round(p_edge, 4),
        "total_paper_trades": n,
        "confidence_level": confidence,
        "keep_testing": "YES" if keep_testing else "NO",
        "stop_reason": stop_reason,
        "warnings": warnings,
    }
    return new, report


def format_report(r):
    lines = [
        f"PRIOR ESTIMATE: {r['prior_estimate']:.1%}",
        f"NEW SAMPLE SIZE: {r['new_sample_size']}",
        f"NEW WINS: {r['new_wins']}",
        f"NEW LOSSES: {r['new_losses']}",
        f"UPDATED ESTIMATE: {r['updated_estimate']:.1%}  (90% range {r['credible_interval_90'][0]:.1%} - "
        f"{r['credible_interval_90'][1]:.1%}; breakeven {r['breakeven_win_rate']:.1%}; "
        f"P(beats breakeven) {r['prob_above_breakeven']:.0%})",
        f"CONFIDENCE LEVEL: {r['confidence_level']}",
        f"SHOULD THE SYSTEM KEEP TESTING: {r['keep_testing']}" + (f" - {r['stop_reason']}" if r["stop_reason"] else ""),
    ]
    lines += [f"  ! {w}" for w in r["warnings"]]
    return "\n".join(lines)
