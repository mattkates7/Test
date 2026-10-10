"""Checks for the project's hard rules. Run: python -m unittest discover -s tests"""

import copy
import json
import re
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from src import data_loader, prediction_markets, temporal_research
from src import probability_update as pu
from src.config import ConfigError, load_config, validate_config
from src.execution import PaperExecutor
from src.main import main
from src.risk_sizing import size_position
from src.safety import LiveTradingRefused, assert_paper_mode

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()


class SafetyTests(unittest.TestCase):
    def test_config_live_mode_refused(self):
        cfg = dict(CFG, mode="live")
        with self.assertRaises(LiveTradingRefused):
            assert_paper_mode(cfg, {})

    def test_env_live_mode_refused(self):
        with self.assertRaises(LiveTradingRefused):
            assert_paper_mode(CFG, {"TRADING_MODE": "live"})

    def test_wallet_secret_in_env_refused(self):
        with self.assertRaises(LiveTradingRefused):
            assert_paper_mode(CFG, {"TRADING_MODE": "paper", "WALLET_PRIVATE_KEY": "abc"})

    def test_live_command_refuses(self):
        self.assertEqual(main(["live"]), 2)

    def test_executor_live_order_refuses(self):
        with self.assertRaises(LiveTradingRefused):
            PaperExecutor("BTC-USD", 0.1, 0.05).send_live_order()

    def test_only_data_loader_touches_network(self):
        pattern = re.compile(r"^\s*(import|from)\s+(urllib|socket|http|requests|websocket|ccxt|web3)", re.M)
        offenders = [
            p.name for p in (ROOT / "src").glob("*.py")
            if p.name != "data_loader.py" and pattern.search(p.read_text())
        ]
        self.assertEqual(offenders, [])


class ConfigTests(unittest.TestCase):
    def test_risk_above_one_percent_rejected(self):
        cfg = copy.deepcopy(CFG)
        cfg["risk"]["max_risk_per_trade_pct"] = 2.0
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_leverage_rejected(self):
        cfg = copy.deepcopy(CFG)
        cfg["risk"]["leverage"] = 3.0
        with self.assertRaises(ConfigError):
            validate_config(cfg)


class SizingTests(unittest.TestCase):
    def size(self, **kw):
        args = dict(account_size=10_000, win_probability=0.6, avg_win=2, avg_loss=1,
                    entry=60_000, stop=59_000, sample_size=500, risk_cfg=CFG["risk"])
        args.update(kw)
        return size_position(**args)

    def test_kelly_capped_at_one_percent(self):
        r = self.size(win_probability=0.9)
        self.assertEqual(r.decision, "APPROVE")
        self.assertGreater(r.raw_kelly, 0.5)
        self.assertLessEqual(r.dollar_risk, 100.0 + 1e-9)

    def test_no_leverage(self):
        r = self.size(stop=59_990)  # tiny stop would need a huge position
        self.assertLessEqual(r.position_value, 10_000 + 1e-6)

    def test_small_sample_rejected(self):
        self.assertEqual(self.size(sample_size=10).decision, "REJECT")

    def test_negative_edge_rejected(self):
        self.assertEqual(self.size(win_probability=0.3, avg_win=1, avg_loss=1).decision, "REJECT")

    def test_no_risk_increase_after_loss(self):
        r = self.size(win_probability=0.9, previous_fraction=0.002, previous_was_loss=True)
        self.assertAlmostEqual(r.paper_fraction, 0.002)


class ProbabilityTests(unittest.TestCase):
    def test_prior_starts_at_breakeven(self):
        be = pu.breakeven_win_rate(1.5, 1.0)
        self.assertAlmostEqual(pu.make_prior(CFG["probability"], be).mean, be)

    def test_small_batch_is_low_confidence(self):
        prior = pu.make_prior(CFG["probability"], 0.4)
        _, rep = pu.update(prior, 5, 0, CFG["probability"], 0.4)
        self.assertEqual(rep["confidence_level"], "LOW")
        self.assertLess(rep["updated_estimate"], 0.55)  # 5 straight wins can't swing it much

    def test_clear_losing_record_stops_testing(self):
        prior = pu.make_prior(CFG["probability"], 0.4)
        _, rep = pu.update(prior, 10, 60, CFG["probability"], 0.4)
        self.assertEqual(rep["keep_testing"], "NO")


class ExecutionTests(unittest.TestCase):
    def test_one_position_at_a_time(self):
        ex = PaperExecutor("BTC-USD", 0.1, 0.05)
        sig = SimpleNamespace(window="hour_14", direction="long", reason="test")
        sizing = SimpleNamespace(position_size=0.01, dollar_risk=10)
        vol = SimpleNamespace(condition="CLEAN")
        ex.open_trade(date(2024, 1, 1), 1, sig, 100, 99, 102, sizing, 0.5, "LOW", vol)
        with self.assertRaises(RuntimeError):
            ex.open_trade(date(2024, 1, 1), 2, sig, 100, 99, 102, sizing, 0.5, "LOW", vol)


class ResearchTests(unittest.TestCase):
    def test_random_walk_has_no_edge(self):
        candles = data_loader.make_synthetic_candles(days=400, seed=11)
        report = temporal_research.analyze(candles, CFG)
        self.assertEqual(report["overall_verdict"], "NO EDGE FOUND")

    def test_small_sample_never_an_edge(self):
        candles = data_loader.make_synthetic_candles(days=40, seed=2)
        report = temporal_research.analyze(candles, CFG)
        for w in report["windows"]:
            self.assertEqual(w["verdict"], "INSUFFICIENT SAMPLE")


class PredictionMarketTests(unittest.TestCase):
    def test_stale_or_weak_evidence_skips(self):
        m = {"market": "x", "implied_probability": 0.3, "resolution_criteria": "y",
             "evidence_yes": [{"claim": "c", "source": "s", "date": "2020-01-01", "quality": "strong"}],
             "my_estimate_low": 0.6, "my_estimate_high": 0.8}
        r = prediction_markets.analyze_market(m, CFG["prediction_markets"], date(2026, 10, 10))
        self.assertEqual(r["paper_decision"], "SKIP")

    def test_example_file_decisions(self):
        data = json.loads((ROOT / "data/prediction_markets.example.json").read_text())
        as_of = date.fromisoformat(data["as_of"])
        decisions = [prediction_markets.analyze_market(m, CFG["prediction_markets"], as_of)["paper_decision"]
                     for m in data["markets"]]
        self.assertEqual(decisions, ["SKIP", "SKIP", "YES"])


if __name__ == "__main__":
    unittest.main()
