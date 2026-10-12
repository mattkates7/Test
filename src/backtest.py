"""Walk-forward paper backtest that wires all modules together.

1. Split history into TRAIN (first part) and TEST (the rest, never seen).
2. On TRAIN: run temporal research and a 'shadow' run of the strategy to
   estimate win rate and average win/loss (in R = multiples of the risk).
3. Start the probability estimate from a conservative prior (no edge).
4. Walk through TEST candle by candle, as if live but simulated:
     signal -> volatility filter -> risk sizing -> paper execution
   logging every fill, close, and blocked trade.
5. Update the probability estimate in batches.
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from . import probability_update as pu
from . import temporal_research
from .execution import PaperExecutor, blocked_ticket
from .data_loader import resample
from .indicators import atr_series
from .risk_sizing import size_position
from .signals import configured_window_names, generate_signal
from .trade_logger import TradeLogger
from .volatility_filter import VolatilityFilter


def _stop_and_target(entry, atr_value, strat):
    dist = strat["stop_atr_multiple"] * atr_value
    rr = strat.get("reward_to_risk")
    sign = 1 if strat["direction"] == "long" else -1
    target = entry + sign * rr * dist if rr else None  # None = trailing-stop exit only
    return entry - sign * dist, target


def _trail_levels(candles, strat):
    """Trailing exit level known at each candle's close: lowest low (long) or
    highest high (short) of the last `trail_exit_bars` candles. None if unused."""
    n = strat.get("trail_exit_bars")
    if not n:
        return [None] * len(candles)
    long = strat["direction"] == "long"
    out = []
    for i in range(len(candles)):
        w = candles[max(0, i - n + 1) : i + 1]
        out.append(min(c.low for c in w) if long else max(c.high for c in w))
    return out


def _breakout_verdict(r_list, min_n):
    """Research verdict for a non-window strategy, from its TRAIN shadow trades."""
    import statistics
    n = len(r_list)
    if n < min_n:
        return f"INSUFFICIENT SAMPLE (n={n} < {min_n})"
    mean = statistics.fmean(r_list)
    t = mean / (statistics.stdev(r_list) / n ** 0.5) if n > 1 and statistics.stdev(r_list) > 0 else 0
    if mean <= 0 or t < 2:
        return f"NO EDGE FOUND (avg {mean:+.2f}R, t={t:.2f})"
    return f"CANDIDATE (needs out-of-sample paper test; avg {mean:+.2f}R, t={t:.2f})"


def _shadow_r_multiples(candles, start, end, atr, cfg, trail=None):
    """Mechanically trade every setup in [start, end) with 1 unit. Returns R results."""
    strat, bt = cfg["strategy"], cfg["backtest"]
    trail = trail or [None] * len(candles)
    ex = PaperExecutor("shadow", bt["fee_pct"], bt["slippage_pct"])
    mechanical = dict(strat, require_research_candidate=False)
    results = []
    for i in range(max(start, 1), end):
        if ex.position is not None:
            t = ex.check_exit(candles[i], i, strat["max_hold_bars"], trail[i])
            if t:
                results.append(t.pnl / t.max_dollar_risk)
            continue
        sig = generate_signal(candles, i, atr, mechanical, cfg["research"]["sessions_utc"], {})
        if sig is None or not sig.passes:
            continue
        entry = ex.entry_price(candles[i].open, sig.direction)
        stop, target = _stop_and_target(entry, atr[i - 1], strat)
        sizing = SimpleNamespace(position_size=1.0, dollar_risk=abs(entry - stop))
        vol = SimpleNamespace(condition="n/a")
        ex.open_trade(candles[i].ts, i, sig, entry, stop, target, sizing, 0.0, "n/a", vol)
        t = ex.check_exit(candles[i], i, strat["max_hold_bars"], trail[i])
        if t:
            results.append(t.pnl / t.max_dollar_risk)
    return results


def _win_loss_stats(r_list, fallback_rr):
    wins = [r for r in r_list if r > 0]
    losses = [-r for r in r_list if r <= 0]
    avg_win = sum(wins) / len(wins) if wins else fallback_rr
    avg_loss = sum(losses) / len(losses) if losses else 1.0
    return len(wins), len(losses), avg_win, max(avg_loss, 1e-6)


def run_backtest(candles, cfg, paths, data_label, verbose_tickets=3):
    strat, bt, risk = cfg["strategy"], cfg["backtest"], cfg["risk"]
    prob_cfg, asset = cfg["probability"], cfg["data"]["product"]
    sessions = cfg["research"]["sessions_utc"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    is_breakout = strat.get("type") == "breakout"
    if is_breakout:
        candles = resample(candles, strat["timeframe_hours"])
        data_label += f", {strat['timeframe_hours']}h bars"
    fallback_rr = strat.get("reward_to_risk") or 1.0

    split = int(len(candles) * bt["train_fraction"])
    train, test_start = candles[:split], split
    atr = atr_series(candles, strat["atr_period"])
    trail = _trail_levels(candles, strat)

    # --- Shadow run on TRAIN to estimate win rate / payoff ---
    train_r = _shadow_r_multiples(candles, 0, split, atr, cfg, trail)
    tw, tl, avg_win_r, avg_loss_r = _win_loss_stats(train_r, fallback_rr)

    # --- Research on TRAIN only (no peeking at the test period) ---
    if is_breakout:
        verdict = _breakout_verdict(train_r, risk["min_sample_size"])
        verdicts = {configured_window_names(strat)[0]: verdict}
        research = {"overall_verdict": verdict}
    else:
        research = temporal_research.analyze(train, cfg)
        verdicts = {w["window"]: w["verdict"] for w in research["windows"]}
    breakeven = pu.breakeven_win_rate(avg_win_r, avg_loss_r)
    estimate = pu.make_prior(prob_cfg, breakeven, tw, tl)
    confidence = "LOW"
    stop_reason = ""

    vf = VolatilityFilter(candles, cfg["volatility_filter"], strat["atr_period"])
    vf.set_reference(0, split)

    executor = PaperExecutor(asset, bt["fee_pct"], bt["slippage_pct"])
    log = TradeLogger(paths["logs"], paths["results"], run_id, cfg["account"]["starting_balance"])
    balance = cfg["account"]["starting_balance"]
    prev_fraction, prev_was_loss = None, False
    pending_w = pending_l = 0
    test_r = []
    shown_open = shown_blocked = 0
    updates = []

    print(f"\nPAPER BACKTEST  run {run_id}  data: {data_label}")
    print(f"  TRAIN {train[0].ts:%Y-%m-%d} .. {train[-1].ts:%Y-%m-%d}  ({len(train_r)} shadow trades, "
          f"win rate {tw / max(len(train_r), 1):.1%}, avg win {avg_win_r:.2f}R, avg loss {avg_loss_r:.2f}R)")
    print(f"  TEST  {candles[test_start].ts:%Y-%m-%d} .. {candles[-1].ts:%Y-%m-%d}")
    print(f"  Breakeven win rate {breakeven:.1%}; starting estimate {estimate.mean:.1%} (conservative prior)")
    for w in configured_window_names(strat):
        print(f"  Research verdict on TRAIN for {w}: {verdicts.get(w, 'NOT TESTED')}")

    def close_trade(t):
        nonlocal balance, prev_fraction, prev_was_loss, pending_w, pending_l
        balance += t.pnl
        log.log_close(t, balance)
        test_r.append(t.pnl / t.max_dollar_risk)
        prev_was_loss = t.outcome == "Loss"
        if t.outcome == "Win":
            pending_w += 1
        else:  # breakeven counts as not-a-win (conservative)
            pending_l += 1

    def maybe_update(force=False):
        nonlocal estimate, confidence, stop_reason, pending_w, pending_l
        if pending_w + pending_l == 0 or (not force and pending_w + pending_l < prob_cfg["batch_size"]):
            return
        estimate, report = pu.update(estimate, pending_w, pending_l, prob_cfg, breakeven)
        report["as_of_trade"] = len(log.closed)
        log.log_probability_update(report)
        updates.append(report)
        confidence = report["confidence_level"]
        if report["keep_testing"] == "NO":
            stop_reason = report["stop_reason"]
        pending_w = pending_l = 0

    for i in range(test_start, len(candles)):
        c = candles[i]
        was_open = executor.position is not None
        if was_open:
            t = executor.check_exit(c, i, strat["max_hold_bars"], trail[i])
            if t:
                close_trade(t)
                maybe_update()

        sig = generate_signal(candles, i, atr, strat, sessions, verdicts)
        if sig is None:
            continue

        vol = vf.classify(i - 1)  # only the last CLOSED candle is known at this open
        entry = executor.entry_price(c.open, sig.direction)
        stop, target = _stop_and_target(entry, atr[i - 1], strat)
        all_r = train_r + test_r
        _, _, aw, al = _win_loss_stats(all_r, fallback_rr)
        sizing = size_position(
            account_size=balance,
            win_probability=estimate.mean,
            avg_win=aw,
            avg_loss=al,
            entry=entry,
            stop=stop,
            sample_size=len(all_r),
            risk_cfg=risk,
            previous_fraction=prev_fraction,
            previous_was_loss=prev_was_loss,
        )
        reasons = executor.check_gates(sig, vol, sizing, stop_reason)
        if was_open and executor.position is None:
            reasons.append("previous paper position was still open at this candle's open")

        if reasons:
            learned = "Skipped. Not trading is a valid outcome."
            log.log_blocked(c.ts, asset, sig.window, reasons, vol.condition, estimate.mean, learned)
            if shown_blocked < verbose_tickets:
                print("\n" + blocked_ticket(c.ts, asset, sig.window, reasons))
                shown_blocked += 1
            continue

        trade = executor.open_trade(c.ts, i, sig, entry, stop, target, sizing, estimate.mean, confidence, vol)
        prev_fraction = sizing.paper_fraction
        log.log_open(trade)
        if shown_open < verbose_tickets:
            print("\n" + trade.ticket())
            print(sizing.report())
            shown_open += 1
        t = executor.check_exit(c, i, strat["max_hold_bars"], trail[i])
        if t:
            close_trade(t)
            maybe_update()

    if executor.position is not None:
        close_trade(executor.force_close(candles[-1]))
    maybe_update(force=True)

    journal = log.write_journal()
    summary = log.summary()
    summary.update(
        {
            "data": data_label,
            "breakeven_win_rate": round(breakeven, 4),
            "final_estimated_win_probability": round(estimate.mean, 4),
            "final_confidence": confidence,
            "research_overall_verdict_on_train": research["overall_verdict"],
            "probability_updates": len(updates),
        }
    )
    with open(paths["results"] / "backtest_summary.json", "w") as f:
        json.dump({"summary": summary, "probability_updates": updates}, f, indent=2)

    if updates:
        print("\nLATEST PROBABILITY UPDATE")
        print(pu.format_report(updates[-1]))
    print(f"\nJournal written to {journal}")
    return summary
