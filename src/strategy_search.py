"""Strategy search with an honest walk-forward check. Historical data only.

Searching many strategies on the same data almost guarantees that one looks
great by luck. So this module reports two things:

1. FULL-PERIOD TABLE: every candidate's results across all history. Useful
   for understanding, but the 'best' row here is biased upward by selection.

2. WALK-FORWARD (the honest number): at the start of each calendar year,
   pick the best candidate using ONLY trades that finished before that year,
   then trade it during that year. The yearly picks are stitched together.
   This tests the whole process 'search, pick, trade', on unseen data.

All candidates are long-only (or flat), trade daily candles, enter at the
next day's open, always carry a stop, pay fees + slippage, and are sized at
1% account risk with no leverage (same rules as the paper system).
"""

import csv
import json
import statistics
from dataclasses import dataclass

from .data_loader import resample
from .indicators import atr_series


# ---------------------------------------------------------------- indicators
def _sma(values, n):
    out, run = [None] * len(values), 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def _rolling(values, n, fn):
    return [fn(values[i - n + 1 : i + 1]) if i >= n - 1 else None for i in range(len(values))]


def _rsi(closes, n):
    out = [None] * len(closes)
    gains = losses = 0.0
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g, l = max(ch, 0), max(-ch, 0)
        if i <= n:
            gains += g / n
            losses += l / n
        else:
            gains = (gains * (n - 1) + g) / n
            losses = (losses * (n - 1) + l) / n
        if i >= n:
            out[i] = 100.0 if losses == 0 else 100 - 100 / (1 + gains / losses)
    return out


class Ind:
    """Lazily computed, cached indicators on daily candles."""

    def __init__(self, d):
        self.d = d
        self.close = [c.close for c in d]
        self.high = [c.high for c in d]
        self.low = [c.low for c in d]
        self.atr = atr_series(d, 14)
        self._cache = {}

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def sma(self, n):
        return self.get(("sma", n), lambda: _sma(self.close, n))

    def hh(self, n):
        return self.get(("hh", n), lambda: _rolling(self.high, n, max))

    def ll(self, n):
        return self.get(("ll", n), lambda: _rolling(self.low, n, min))

    def rsi(self, n):
        return self.get(("rsi", n), lambda: _rsi(self.close, n))


# ---------------------------------------------------------------- candidates
# Each candidate: entry(i) / exit(i) decide at the OPEN of day i using only
# data up to the close of day i-1. stop_k = initial stop in ATRs.
@dataclass
class Candidate:
    name: str
    family: str
    entry: object
    exit: object
    stop_k: float
    max_hold: int = 10_000


def build_candidates(ind):
    C, cands = ind.close, []

    for n, m in [(10, 5), (20, 10), (55, 20), (100, 50)]:
        hh, ll = ind.hh(n), ind.ll(m)
        cands.append(Candidate(
            f"breakout {n}d high / exit {m}d low", "breakout",
            lambda i, hh=hh: hh[i - 2] is not None and C[i - 1] > hh[i - 2],
            lambda i, ll=ll: ll[i - 2] is not None and C[i - 1] < ll[i - 2],
            2.0))

    for n in [20, 50, 100, 200]:
        s = ind.sma(n)
        cands.append(Candidate(
            f"close above {n}d SMA", "trend MA",
            lambda i, s=s: s[i - 1] is not None and C[i - 1] > s[i - 1],
            lambda i, s=s: s[i - 1] is not None and C[i - 1] < s[i - 1],
            3.0))

    for f, sl in [(10, 50), (20, 100), (50, 200)]:
        a, b = ind.sma(f), ind.sma(sl)
        cands.append(Candidate(
            f"SMA {f}/{sl} cross", "trend MA",
            lambda i, a=a, b=b: b[i - 1] is not None and a[i - 1] > b[i - 1],
            lambda i, a=a, b=b: b[i - 1] is not None and a[i - 1] < b[i - 1],
            3.0))

    for lb in [14, 30, 90]:
        cands.append(Candidate(
            f"momentum: {lb}d return > 0", "momentum",
            lambda i, lb=lb: i - 1 - lb >= 0 and C[i - 1] > C[i - 1 - lb],
            lambda i, lb=lb: i - 1 - lb >= 0 and C[i - 1] < C[i - 1 - lb],
            3.0))

    s5, s200 = ind.sma(5), ind.sma(200)
    for rn, lvl in [(2, 10), (14, 30)]:
        r = ind.rsi(rn)
        cands.append(Candidate(
            f"RSI{rn} < {lvl} buy dip, exit > 5d SMA", "mean reversion",
            lambda i, r=r, lvl=lvl: r[i - 1] is not None and r[i - 1] < lvl,
            lambda i: s5[i - 1] is not None and C[i - 1] > s5[i - 1],
            2.0, max_hold=10))
        cands.append(Candidate(
            f"RSI{rn} < {lvl} dip in uptrend (>200d SMA)", "mean reversion",
            lambda i, r=r, lvl=lvl: r[i - 1] is not None and s200[i - 1] is not None
            and r[i - 1] < lvl and C[i - 1] > s200[i - 1],
            lambda i: s5[i - 1] is not None and C[i - 1] > s5[i - 1],
            2.0, max_hold=10))

    for drop, hold in [(0.07, 3), (0.07, 7), (0.10, 5)]:
        cands.append(Candidate(
            f"buy after {drop:.0%} down day, hold {hold}d", "mean reversion",
            lambda i, drop=drop: i >= 2 and C[i - 1] / C[i - 2] - 1 <= -drop,
            lambda i: False,
            2.0, max_hold=hold))

    weekday = [c.ts.weekday() for c in ind.d]
    cands.append(Candidate(
        "weekend: hold Fri open -> Mon open", "calendar",
        lambda i: weekday[i] == 4, lambda i: weekday[i] == 0, 2.0))
    cands.append(Candidate(
        "weekdays only: Mon open -> Fri open", "calendar",
        lambda i: weekday[i] == 0, lambda i: weekday[i] == 4, 2.0))
    return cands


# ---------------------------------------------------------------- simulator
@dataclass
class SimTrade:
    entry_i: int
    exit_i: int
    entry_ts: object
    exit_ts: object
    r: float          # result in R (multiples of initial risk), after costs
    stop_pct: float   # initial stop distance as a fraction of entry
    reason: str


def simulate(cand, ind, fee, slip):
    d, atr = ind.d, ind.atr
    trades, pos = [], None
    for i in range(2, len(d)):
        c = d[i]
        if pos:
            e, stop, ei = pos
            px = why = None
            if cand.exit(i) or i - ei >= cand.max_hold:
                px, why = c.open * (1 - slip), "exit signal" if cand.exit(i) else "time exit"
            elif c.open <= stop:
                px, why = c.open * (1 - slip), "gap through stop"
            if px is None and c.low <= stop:
                px, why = stop * (1 - slip), "stop"
            if px is not None:
                risk = e - pos_stop0
                trades.append(SimTrade(ei, i, d[ei].ts, c.ts, (px - e - fee * (e + px)) / risk, risk / e, why))
                pos = None
            continue
        if atr[i - 1] is None or not cand.entry(i):
            continue
        e = c.open * (1 + slip)
        pos_stop0 = e - cand.stop_k * atr[i - 1]
        pos = (e, pos_stop0, i)
        if c.low <= pos_stop0:  # stopped out on the entry day
            px = pos_stop0 * (1 - slip)
            trades.append(SimTrade(i, i, c.ts, c.ts, (px - e - fee * (e + px)) / (e - pos_stop0),
                                   (e - pos_stop0) / e, "stop"))
            pos = None
    return trades


# ---------------------------------------------------------------- scoring
def stats(trades, risk_pct=0.01):
    rs = [t.r for t in trades]
    n = len(rs)
    if n == 0:
        return {"trades": 0, "win_rate": 0, "avg_r": 0, "total_r": 0, "t_stat": 0, "return_pct": 0, "max_dd_pct": 0}
    sd = statistics.stdev(rs) if n > 1 else 0
    eq, peak, dd = 1.0, 1.0, 0.0
    for t in trades:
        eq *= 1 + min(risk_pct, t.stop_pct) * t.r  # 1% risk, but no leverage
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak)
    return {
        "trades": n,
        "win_rate": sum(r > 0 for r in rs) / n,
        "avg_r": statistics.fmean(rs),
        "total_r": sum(rs),
        "t_stat": statistics.fmean(rs) / (sd / n ** 0.5) if sd > 0 else 0,
        "return_pct": 100 * (eq - 1),
        "max_dd_pct": 100 * dd,
    }


def _score(trades, min_trades):
    """Selection score: t-stat of R per trade (rewards consistency AND sample size)."""
    if len(trades) < min_trades:
        return None
    s = stats(trades)
    return s["t_stat"] if s["avg_r"] > 0 else None


def walk_forward(cands, results, first_year, last_year, min_trades):
    picks, chosen_trades = [], []
    last_exit = -1
    for year in range(first_year, last_year + 1):
        best, best_score = None, None
        for c in cands:
            past = [t for t in results[c.name] if t.exit_ts.year < year]
            sc = _score(past, min_trades)
            if sc is not None and (best_score is None or sc > best_score):
                best, best_score = c, sc
        if best is None:
            picks.append({"year": year, "pick": "STAY FLAT (nothing qualified)", "score": None})
            continue
        yr = [t for t in results[best.name] if t.entry_ts.year == year and t.entry_i > last_exit]
        if yr:
            last_exit = yr[-1].exit_i
        chosen_trades += yr
        s = stats(yr)
        picks.append({"year": year, "pick": best.name, "score": round(best_score, 2),
                      "trades": s["trades"], "total_r": round(s["total_r"], 2),
                      "return_pct": round(s["return_pct"], 2)})
    return picks, chosen_trades


def run(hourly, cfg):
    bt = cfg["backtest"]
    fee, slip = bt["fee_pct"] / 100, bt["slippage_pct"] / 100
    d = resample(hourly, 24)
    ind = Ind(d)
    cands = build_candidates(ind)
    results = {c.name: simulate(c, ind, fee, slip) for c in cands}

    mid = d[len(d) // 2].ts
    table = []
    for c in cands:
        tr = results[c.name]
        full = stats(tr)
        h1 = stats([t for t in tr if t.entry_ts < mid])
        h2 = stats([t for t in tr if t.entry_ts >= mid])
        table.append({"strategy": c.name, "family": c.family, **{k: round(v, 3) for k, v in full.items()},
                      "avg_r_first_half": round(h1["avg_r"], 3), "avg_r_second_half": round(h2["avg_r"], 3),
                      "positive_in_both_halves": h1["avg_r"] > 0 and h2["avg_r"] > 0})
    table.sort(key=lambda r: r["t_stat"], reverse=True)

    first_year = d[0].ts.year + 2  # need ~1.5 years of history before the first pick
    picks, wf_trades = walk_forward(cands, results, first_year, d[-1].ts.year, min_trades=15)
    wf = stats(wf_trades)
    wf_start = next((t.entry_ts for t in wf_trades), None)
    bh_from = next((c for c in d if c.ts.year >= first_year), d[0])
    return {
        "data": f"{d[0].ts:%Y-%m-%d} .. {d[-1].ts:%Y-%m-%d} ({len(d)} daily candles)",
        "round_trip_cost_pct": 2 * (bt["fee_pct"] + bt["slippage_pct"]),
        "candidates_tested": len(cands),
        "table": table,
        "walk_forward": {"from_year": first_year, "picks": picks,
                         **{k: round(v, 3) for k, v in wf.items()},
                         "first_trade": str(wf_start)},
        "buy_and_hold_same_period_pct": round(100 * (d[-1].close / bh_from.open - 1), 1),
        "_wf_trades": wf_trades,
    }


def save(report, json_path, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report["table"][0].keys()))
        w.writeheader()
        w.writerows(report["table"])
    out = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)


def print_report(r):
    print(f"\nSTRATEGY SEARCH  {r['data']}, costs {r['round_trip_cost_pct']:.2f}% round trip, "
          f"{r['candidates_tested']} candidates, long-only, 1% risk/trade, no leverage")
    print("\nA) FULL-PERIOD TABLE (biased: the best row was picked AFTER seeing all the data)")
    h = f"{'strategy':<44}{'n':>5}{'win%':>6}{'avgR':>7}{'totR':>7}{'t':>6}{'ret%':>8}{'DD%':>6}  both halves +"
    print(h)
    print("-" * len(h))
    for x in r["table"]:
        print(f"{x['strategy']:<44}{x['trades']:>5}{100 * x['win_rate']:>6.0f}{x['avg_r']:>7.2f}{x['total_r']:>7.1f}"
              f"{x['t_stat']:>6.2f}{x['return_pct']:>8.1f}{x['max_dd_pct']:>6.1f}  {'yes' if x['positive_in_both_halves'] else 'no'}")
    wf = r["walk_forward"]
    print(f"\nB) WALK-FORWARD (honest): each year, pick the best candidate using only earlier data, trade it that year")
    for p in wf["picks"]:
        extra = "" if p["score"] is None else f"  -> {p['trades']} trades, {p['total_r']:+.1f}R, {p['return_pct']:+.1f}%"
        print(f"  {p['year']}: {p['pick']}{extra}")
    print(f"  TOTAL: {wf['trades']} trades, win rate {100 * wf['win_rate']:.0f}%, avg {wf['avg_r']:+.2f}R, "
          f"t={wf['t_stat']:.2f}, return {wf['return_pct']:+.1f}% on the paper account, max drawdown {wf['max_dd_pct']:.1f}%")
    print(f"  Buy & hold BTC over the same years: {r['buy_and_hold_same_period_pct']:+.1f}%")
