"""Logging (Step 9). Every simulated fill, close, loss and BLOCKED/skipped trade.

Outputs:
  logs/paper_trades.jsonl         one JSON event per line (OPEN / CLOSE / BLOCKED), append-only
  logs/probability_updates.jsonl  each Bayesian batch update
  results/paper_trading_journal.csv
      the 'spreadsheet': account summary block at the top, then one row per
      trade or blocked trade. Opens in Excel / Google Sheets / Numbers.
"""

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .indicators import max_drawdown_from_equity

JOURNAL_COLUMNS = [
    "Trade ID",
    "Date and time (UTC)",
    "Asset or market",
    "Signal window",
    "Entry",
    "Exit",
    "Stop",
    "Target",
    "Paper position size",
    "Maximum dollar risk",
    "Volatility classification",
    "Estimated probability before trade",
    "Profit/loss",
    "Win / Loss / Breakeven / Blocked",
    "Reason for entry",
    "Reason for exit",
    "What the system learned",
]


def _iso(ts):
    return ts.isoformat() if hasattr(ts, "isoformat") else ts


class TradeLogger:
    def __init__(self, logs_dir, results_dir, run_id, starting_balance):
        self.logs_dir = Path(logs_dir)
        self.results_dir = Path(results_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.run_id = run_id
        self.starting_balance = starting_balance
        self.rows = []
        self.closed = []
        self.blocked = 0
        self.equity = [starting_balance]

    def _event(self, filename, payload):
        payload = {"run_id": self.run_id, "logged_at": datetime.now(timezone.utc).isoformat(), **payload}
        with open(self.logs_dir / filename, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def log_open(self, t):
        self._event("paper_trades.jsonl", {"event": "OPEN", **_trade_dict(t)})

    def log_close(self, t, balance_after):
        self._event("paper_trades.jsonl", {"event": "CLOSE", "balance_after": balance_after, **_trade_dict(t)})
        self.closed.append(t)
        self.equity.append(balance_after)
        self.rows.append(
            {
                "Trade ID": t.trade_id,
                "Date and time (UTC)": f"{t.opened_at:%Y-%m-%d %H:%M} -> {t.closed_at:%Y-%m-%d %H:%M}",
                "Asset or market": t.asset,
                "Signal window": t.setup,
                "Entry": round(t.entry, 2),
                "Exit": round(t.exit, 2),
                "Stop": round(t.stop, 2),
                "Target": round(t.target, 2) if t.target else "trailing",
                "Paper position size": round(t.size, 6),
                "Maximum dollar risk": round(t.max_dollar_risk, 2),
                "Volatility classification": t.volatility,
                "Estimated probability before trade": round(t.est_probability, 4),
                "Profit/loss": round(t.pnl, 2),
                "Win / Loss / Breakeven / Blocked": t.outcome,
                "Reason for entry": t.entry_reason,
                "Reason for exit": t.exit_reason,
                "What the system learned": t.learned,
            }
        )

    def log_blocked(self, ts, asset, window, reasons, volatility="", est_p=None, learned=""):
        self.blocked += 1
        self._event(
            "paper_trades.jsonl",
            {"event": "BLOCKED", "ts": _iso(ts), "asset": asset, "window": window, "reasons": reasons,
             "volatility": volatility, "est_probability": est_p},
        )
        self.rows.append(
            {
                "Trade ID": "",
                "Date and time (UTC)": f"{ts:%Y-%m-%d %H:%M}",
                "Asset or market": asset,
                "Signal window": window,
                "Volatility classification": volatility,
                "Estimated probability before trade": "" if est_p is None else round(est_p, 4),
                "Profit/loss": 0,
                "Win / Loss / Breakeven / Blocked": "Blocked",
                "Reason for entry": "",
                "Reason for exit": " | ".join(reasons),
                "What the system learned": learned,
            }
        )

    def log_probability_update(self, report):
        self._event("probability_updates.jsonl", report)

    def summary(self):
        pnls = [t.pnl for t in self.closed]
        wins = [p for t, p in zip(self.closed, pnls) if t.outcome == "Win"]
        losses = [p for t, p in zip(self.closed, pnls) if t.outcome == "Loss"]
        current = self.equity[-1]
        return {
            "run_id": self.run_id,
            "starting_paper_balance": round(self.starting_balance, 2),
            "current_paper_balance": round(current, 2),
            "total_trades": len(self.closed),
            "blocked_or_skipped": self.blocked,
            "wins": len(wins),
            "losses": len(losses),
            "breakevens": len(self.closed) - len(wins) - len(losses),
            "win_rate": round(len(wins) / len(self.closed), 4) if self.closed else 0.0,
            "average_win": round(statistics.fmean(wins), 2) if wins else 0.0,
            "average_loss": round(statistics.fmean(losses), 2) if losses else 0.0,
            "largest_drawdown_pct": round(100 * max_drawdown_from_equity(self.equity), 2),
            "total_simulated_pnl": round(current - self.starting_balance, 2),
        }

    def write_journal(self, path=None):
        path = Path(path or self.results_dir / "paper_trading_journal.csv")
        s = self.summary()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["PAPER TRADING JOURNAL - SIMULATION ONLY, NOT REAL MONEY"])
            w.writerow(["Run ID", s["run_id"]])
            w.writerow(["Starting paper balance", s["starting_paper_balance"]])
            w.writerow(["Current paper balance", s["current_paper_balance"]])
            w.writerow(["Total trades", s["total_trades"]])
            w.writerow(["Blocked / skipped", s["blocked_or_skipped"]])
            w.writerow(["Win rate", s["win_rate"]])
            w.writerow(["Average win", s["average_win"]])
            w.writerow(["Average loss", s["average_loss"]])
            w.writerow(["Largest drawdown %", s["largest_drawdown_pct"]])
            w.writerow(["Total simulated P&L", s["total_simulated_pnl"]])
            w.writerow([])
            dw = csv.DictWriter(f, fieldnames=JOURNAL_COLUMNS)
            dw.writeheader()
            for row in self.rows:
                dw.writerow(row)
        return path


def _trade_dict(t):
    d = {k: _iso(v) for k, v in t.__dict__.items() if k != "extra"}
    return d
