"""Temporal Bias Research (Step 2). Historical data only, no trading.

Question: did some BTC time windows behave differently from the rest?

Rule tested for every window (kept identical so windows are comparable):
    Enter at the window's first candle OPEN in `strategy.direction`,
    exit at the window's last candle CLOSE, pay round-trip fees + slippage.
    A sample is a WIN if that net return is > 0, otherwise a LOSS.

Windows tested:
    * hour of day (UTC)           hold for `research.hold_hours` candles
    * day of week (UTC)           00:00 open -> 23:00 close of that day
    * session windows (UTC)       from config `research.sessions_utc`
    * time after large moves      after a 1h candle moved >= threshold, hold N hours

Each window is compared with a BASELINE: every non-overlapping return of the
same length over the same dates. A window only matters if it beats the
baseline, not just if BTC happened to rise.

Guardrails against fooling ourselves:
    * Small samples are labelled INSUFFICIENT SAMPLE, never an edge.
    * Many windows are tested, so the significance bar is raised
      (Bonferroni correction). Testing ~40 windows will produce ~2 'significant'
      ones at 5% by pure chance.
    * The result has to point the same way in every separate date range.
    * Even then the best label is CANDIDATE - it still needs an out-of-sample
      paper test. Nothing here is ever labelled a "real edge".
"""

import csv
import json
import math
import statistics
from datetime import timedelta
from statistics import NormalDist

from .indicators import max_drawdown

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOUR = timedelta(hours=1)


class _Index:
    def __init__(self, candles):
        self.candles = candles
        self.pos = {c.ts: i for i, c in enumerate(candles)}

    def span_return(self, start_i, length):
        """Open of candle start_i -> close of candle start_i+length-1, if no gaps."""
        end_i = start_i + length - 1
        if end_i >= len(self.candles):
            return None
        c0, c1 = self.candles[start_i], self.candles[end_i]
        if c1.ts - c0.ts != (length - 1) * HOUR:
            return None  # data gap inside the window, skip it
        return c1.close / c0.open - 1


def _net(gross, direction, cost):
    r = gross if direction == "long" else -gross
    return r - cost


def _collect_windows(candles, cfg):
    """Return {window_name: (length_in_hours, [(ts, net_return), ...])}."""
    research, strat, bt = cfg["research"], cfg["strategy"], cfg["backtest"]
    direction = strat["direction"]
    cost = 2 * (bt["fee_pct"] + bt["slippage_pct"]) / 100
    hold = research["hold_hours"]
    idx = _Index(candles)
    windows = {}

    def add(name, length, ts, gross):
        if gross is not None:
            windows.setdefault(name, (length, []))[1].append((ts, _net(gross, direction, cost)))

    for i, c in enumerate(candles):
        # Hour of day
        add(f"hour_{c.ts.hour:02d}", hold, c.ts, idx.span_return(i, hold))
        # Day of week + sessions start at specific hours
        if c.ts.hour == 0:
            add(f"day_{DAY_NAMES[c.ts.weekday()]}", 24, c.ts, idx.span_return(i, 24))
        for name, (start_h, end_h) in research["sessions_utc"].items():
            if c.ts.hour == start_h:
                add(f"session_{name}", end_h - start_h, c.ts, idx.span_return(i, end_h - start_h))

    # Time after large moves (non-overlapping events)
    threshold = research["large_move_threshold_pct"] / 100
    i = 0
    while i < len(candles) - 1:
        c = candles[i]
        move = c.close / c.open - 1
        if abs(move) >= threshold and candles[i + 1].ts - c.ts == HOUR:
            name = "after_large_up" if move > 0 else "after_large_down"
            add(name, hold, candles[i + 1].ts, idx.span_return(i + 1, hold))
            i += hold + 1
        else:
            i += 1
    return windows


def _baseline(candles, length, direction, cost):
    """Non-overlapping returns of `length` hours across all data, as (ts, net)."""
    idx = _Index(candles)
    out = []
    i = 0
    while i < len(candles):
        r = idx.span_return(i, length)
        if r is not None:
            out.append((candles[i].ts, _net(r, direction, cost)))
            i += length
        else:
            i += 1
    return out


def _mean_var(values):
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0
    return statistics.fmean(values), statistics.variance(values)


def _excess_t(sample, base):
    m1, v1 = _mean_var(sample)
    m2, v2 = _mean_var(base)
    se = math.sqrt(v1 / max(len(sample), 1) + v2 / max(len(base), 1))
    return (m1 - m2), ((m1 - m2) / se if se > 0 else 0.0)


def analyze(candles, cfg):
    research, strat, bt = cfg["research"], cfg["strategy"], cfg["backtest"]
    direction = strat["direction"]
    cost = 2 * (bt["fee_pct"] + bt["slippage_pct"]) / 100
    min_n = research["min_sample_size"]
    splits = research["date_range_splits"]

    windows = _collect_windows(candles, cfg)
    num_tests = len(windows)
    z_needed = NormalDist().inv_cdf(1 - research["family_alpha"] / (2 * max(num_tests, 1)))

    first_ts, last_ts = candles[0].ts, candles[-1].ts
    span = (last_ts - first_ts) / splits
    ranges = [(first_ts + k * span, first_ts + (k + 1) * span) for k in range(splits)]

    baselines = {}
    rows = []
    for name, (length, samples) in sorted(windows.items()):
        if length not in baselines:
            baselines[length] = _baseline(candles, length, direction, cost)
        base = baselines[length]
        rets = [r for _, r in samples]
        base_rets = [r for _, r in base]
        n = len(rets)
        wins = sum(1 for r in rets if r > 0)
        excess, t = _excess_t(rets, base_rets)

        split_excess = []
        for lo, hi in ranges:
            s = [r for ts, r in samples if lo <= ts <= hi]
            b = [r for ts, r in base if lo <= ts <= hi]
            if len(s) >= 10 and b:
                split_excess.append(statistics.fmean(s) - statistics.fmean(b))
            else:
                split_excess.append(None)
        survives = all(e is not None and e * excess > 0 for e in split_excess)

        flags = []
        if n < min_n:
            verdict = "INSUFFICIENT SAMPLE"
            flags.append(f"only {n} samples (< {min_n})")
        elif abs(t) < z_needed:
            verdict = "NO EDGE FOUND"
        elif not survives:
            verdict = "UNSTABLE - LIKELY OVERFIT"
            flags.append("direction of the effect flips between date ranges")
        else:
            verdict = "CANDIDATE (needs out-of-sample paper test)"
            flags.append("still could be chance or a past regime; confirm out of sample")
        if excess < 0 and verdict.startswith("CANDIDATE"):
            flags.append(f"window UNDERPERFORMS baseline - not a {direction} signal")

        rows.append(
            {
                "window": name,
                "hold_hours": length,
                "sample_size": n,
                "wins": wins,
                "losses": n - wins,
                "win_rate": wins / n if n else 0.0,
                "avg_return_pct": 100 * statistics.fmean(rets) if rets else 0.0,
                "median_return_pct": 100 * statistics.median(rets) if rets else 0.0,
                "baseline_avg_return_pct": 100 * statistics.fmean(base_rets) if base_rets else 0.0,
                "excess_vs_baseline_pct": 100 * excess,
                "t_stat_vs_baseline": t,
                "max_drawdown_pct": 100 * max_drawdown(rets),
                "excess_by_date_range_pct": [
                    None if e is None else round(100 * e, 4) for e in split_excess
                ],
                "survives_all_date_ranges": survives,
                "verdict": verdict,
                "flags": flags,
            }
        )

    candidates = [r for r in rows if r["verdict"].startswith("CANDIDATE") and r["excess_vs_baseline_pct"] > 0]
    days = (last_ts - first_ts).days
    warnings = [
        f"{num_tests} windows tested; at alpha={research['family_alpha']} roughly "
        f"{num_tests * research['family_alpha']:.1f} could look 'significant' by luck alone. "
        f"The bar was raised to |t| >= {z_needed:.2f} (Bonferroni).",
        "Past behaviour of a time window can disappear once others notice it.",
    ]
    if days < 365:
        warnings.append(f"Only {days} days of data. Use at least 1-2 years before trusting anything.")
    return {
        "data_first": first_ts.isoformat(),
        "data_last": last_ts.isoformat(),
        "days_of_data": days,
        "direction_tested": direction,
        "round_trip_cost_pct": 100 * cost,
        "windows_tested": num_tests,
        "t_stat_required": z_needed,
        "overall_verdict": "CANDIDATES FOUND (unconfirmed)" if candidates else "NO EDGE FOUND",
        "candidates": [r["window"] for r in candidates],
        "overfitting_warnings": warnings,
        "windows": rows,
    }


def save_report(report, json_path, csv_path):
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    fields = [k for k in report["windows"][0].keys()] if report["windows"] else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in report["windows"]:
            row = dict(row)
            row["flags"] = "; ".join(row["flags"])
            w.writerow(row)


def print_report(report):
    print(f"\nTEMPORAL BIAS RESEARCH  ({report['data_first'][:10]} .. {report['data_last'][:10]}, "
          f"{report['days_of_data']} days, direction={report['direction_tested']}, "
          f"costs={report['round_trip_cost_pct']:.2f}% round trip)")
    header = f"{'window':<24}{'n':>6}{'win%':>7}{'avg%':>8}{'med%':>8}{'xs%':>8}{'t':>7}{'maxDD%':>8}  stable  verdict"
    print(header)
    print("-" * len(header))
    for r in report["windows"]:
        print(
            f"{r['window']:<24}{r['sample_size']:>6}{100 * r['win_rate']:>7.1f}"
            f"{r['avg_return_pct']:>8.3f}{r['median_return_pct']:>8.3f}"
            f"{r['excess_vs_baseline_pct']:>8.3f}{r['t_stat_vs_baseline']:>7.2f}"
            f"{r['max_drawdown_pct']:>8.1f}  {'yes' if r['survives_all_date_ranges'] else 'no ':<6}  {r['verdict']}"
        )
    print("\nxs% = average return minus the baseline average for the same holding length.")
    print("Overfitting warnings:")
    for w in report["overfitting_warnings"]:
        print(f"  ! {w}")
    print(f"\nOVERALL: {report['overall_verdict']}")
    if report["candidates"]:
        print("  Candidates (NOT proven edges): " + ", ".join(report["candidates"]))
