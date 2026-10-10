"""Terminal dashboard: reads the latest results and prints a summary."""

import json
from collections import Counter
from pathlib import Path

SPARK = " ▁▂▃▄▅▆▇█"


def sparkline(values, width=60):
    if len(values) < 2:
        return ""
    step = max(1, len(values) // width)
    pts = values[::step]
    lo, hi = min(pts), max(pts)
    span = hi - lo or 1
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in pts)


def show(results_dir, logs_dir):
    results_dir, logs_dir = Path(results_dir), Path(logs_dir)
    summary_file = results_dir / "backtest_summary.json"
    print("=" * 64)
    print(" PAPER TRADING DASHBOARD - SIMULATION ONLY, NO REAL MONEY")
    print("=" * 64)
    if not summary_file.exists():
        print("No backtest yet. Run: python -m src.main backtest")
    else:
        s = json.loads(summary_file.read_text())["summary"]
        rows = [
            ("Data", s["data"]),
            ("Run", s["run_id"]),
            ("Starting paper balance", f"${s['starting_paper_balance']:,.2f}"),
            ("Current paper balance", f"${s['current_paper_balance']:,.2f}"),
            ("Total simulated P&L", f"${s['total_simulated_pnl']:,.2f}"),
            ("Total trades", s["total_trades"]),
            ("Blocked / skipped", s["blocked_or_skipped"]),
            ("Wins / Losses / Breakeven", f"{s['wins']} / {s['losses']} / {s['breakevens']}"),
            ("Win rate", f"{s['win_rate']:.1%}"),
            ("Average win", f"${s['average_win']:,.2f}"),
            ("Average loss", f"${s['average_loss']:,.2f}"),
            ("Largest drawdown", f"{s['largest_drawdown_pct']:.2f}%"),
            ("Breakeven win rate", f"{s['breakeven_win_rate']:.1%}"),
            ("Estimated win probability", f"{s['final_estimated_win_probability']:.1%} ({s['final_confidence']} confidence)"),
            ("Research verdict (train)", s["research_overall_verdict_on_train"]),
        ]
        for k, v in rows:
            print(f" {k:<28}{v}")

    trades_log = logs_dir / "paper_trades.jsonl"
    if trades_log.exists():
        events = [json.loads(line) for line in trades_log.read_text().splitlines() if line.strip()]
        if events:
            latest = events[-1]["run_id"]
            events = [e for e in events if e["run_id"] == latest]
            equity = [e["balance_after"] for e in events if e["event"] == "CLOSE"]
            if equity:
                print(f"\n Equity curve: {sparkline(equity)}")
            blocked = Counter()
            for e in events:
                if e["event"] == "BLOCKED":
                    for r in e["reasons"]:
                        blocked[r.split(":")[0]] += 1
            if blocked:
                print("\n Why trades were blocked (latest run):")
                for reason, n in blocked.most_common(8):
                    print(f"   {n:>5}  {reason}")

    research = results_dir / "temporal_research.json"
    if research.exists():
        r = json.loads(research.read_text())
        print(f"\n Latest temporal research: {r['overall_verdict']}"
              + (f"  (candidates: {', '.join(r['candidates'])})" if r["candidates"] else ""))

    pm_log = logs_dir / "prediction_market_paper.jsonl"
    if pm_log.exists():
        lines = [json.loads(x) for x in pm_log.read_text().splitlines() if x.strip()]
        decisions = Counter(x["paper_decision"] for x in lines)
        print(f"\n Prediction-market paper decisions logged: {dict(decisions)}")
    print("=" * 64)
