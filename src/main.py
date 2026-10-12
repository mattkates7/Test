"""Command-line entry point. Run `python -m src.main --help`.

Every command first checks that the project is in PAPER MODE.
"""

import argparse
import sys
from pathlib import Path

from . import dashboard, data_loader, prediction_markets, strategy_search, temporal_research
from . import probability_update as pu
from .backtest import run_backtest
from .config import load_config, load_env, resolve_path
from .risk_sizing import size_position
from .safety import LiveTradingRefused, assert_paper_mode, refuse_live_execution
from .volatility_filter import VolatilityFilter

BANNER = "*** PAPER MODE - SIMULATION ONLY. No real orders. Not financial advice. ***"


def _paths():
    return {"logs": resolve_path("logs"), "results": resolve_path("results"), "data": resolve_path("data")}


def _load_data(cfg):
    path = resolve_path(cfg["data"]["csv_path"])
    candles = data_loader.load_candles(path)
    source = data_loader.data_source(path)
    label = f"{path.name} [{source}]"
    if source == "synthetic":
        print("!" * 72)
        print("! WARNING: SYNTHETIC random-walk data. Results only test the plumbing. !")
        print("! Run `python -m src.main fetch-data` on your machine for real data.   !")
        print("!" * 72)
    return candles, label


def cmd_fetch_data(cfg, args):
    days = args.days or cfg["data"]["days_to_fetch"]
    print(f"Downloading {days} days of hourly {cfg['data']['product']} candles (public data, no key)...")
    candles = data_loader.fetch_coinbase_hourly(cfg["data"]["product"], days)
    path = resolve_path(cfg["data"]["csv_path"])
    data_loader.save_candles(candles, path, "coinbase")
    print(f"Saved {len(candles)} candles to {path}")


def cmd_import_csv(cfg, args):
    candles = data_loader.import_cryptodatadownload(args.file)
    path = resolve_path(cfg["data"]["csv_path"])
    data_loader.save_candles(candles, path, f"imported: {Path(args.file).name}")
    gaps = sum(1 for a, b in zip(candles, candles[1:]) if (b.ts - a.ts).total_seconds() != 3600)
    print(f"Imported {len(candles)} candles ({candles[0].ts:%Y-%m-%d} .. {candles[-1].ts:%Y-%m-%d}), "
          f"{gaps} gaps, saved to {path}")


def cmd_make_sample_data(cfg, args):
    candles = data_loader.make_synthetic_candles(days=args.days or 730)
    path = resolve_path(cfg["data"]["csv_path"])
    data_loader.save_candles(candles, path, "synthetic")
    print(f"Saved {len(candles)} SYNTHETIC candles to {path} (for testing the pipeline only).")


def cmd_research(cfg, args):
    candles, _ = _load_data(cfg)
    report = temporal_research.analyze(candles, cfg)
    res = _paths()["results"]
    temporal_research.save_report(report, res / "temporal_research.json", res / "temporal_research.csv")
    temporal_research.print_report(report)
    print(f"\nSaved to {res / 'temporal_research.csv'} and .json")


def cmd_search(cfg, args):
    candles, _ = _load_data(cfg)
    report = strategy_search.run(candles, cfg)
    res = _paths()["results"]
    strategy_search.save(report, res / "strategy_search.json", res / "strategy_search.csv")
    strategy_search.print_report(report)
    print(f"\nSaved to {res / 'strategy_search.csv'} and .json")


def cmd_backtest(cfg, args):
    candles, label = _load_data(cfg)
    run_backtest(candles, cfg, _paths(), label, verbose_tickets=args.show)
    dashboard.show(_paths()["results"], _paths()["logs"])


def cmd_size(cfg, args):
    r = size_position(
        account_size=args.account,
        win_probability=args.win_prob,
        avg_win=args.avg_win,
        avg_loss=args.avg_loss,
        entry=args.entry,
        stop=args.stop,
        sample_size=args.sample_size,
        risk_cfg=cfg["risk"],
    )
    print(r.report())


def cmd_vol_check(cfg, args):
    candles, _ = _load_data(cfg)
    vf = VolatilityFilter(candles, cfg["volatility_filter"], cfg["strategy"]["atr_period"])
    split = int(len(candles) * cfg["backtest"]["train_fraction"])
    vf.set_reference(0, split)
    print(f"Latest closed candle: {candles[-1].ts:%Y-%m-%d %H:%M} UTC, close {candles[-1].close:,.2f}")
    print(f"Reference period: {candles[0].ts:%Y-%m-%d} .. {candles[split - 1].ts:%Y-%m-%d}")
    print(vf.classify(len(candles) - 1).report())


def cmd_update(cfg, args):
    breakeven = pu.breakeven_win_rate(args.avg_win, args.avg_loss)
    prior = pu.make_prior(cfg["probability"], breakeven, args.backtest_wins, args.backtest_losses)
    if args.prior_trades_wins or args.prior_trades_losses:
        prior, _ = pu.update(prior, args.prior_trades_wins, args.prior_trades_losses, cfg["probability"], breakeven)
    _, report = pu.update(prior, args.wins, args.losses, cfg["probability"], breakeven)
    print(pu.format_report(report))


def cmd_predict(cfg, args):
    pm_cfg = cfg["prediction_markets"]
    path = resolve_path(args.input or pm_cfg["input_path"])
    as_of, results = prediction_markets.run(pm_cfg, path, _paths()["logs"])
    print(f"Prediction-market PAPER research, as of {as_of}. Market prices are not assumed wrong.\n")
    for r in results:
        print(prediction_markets.format_result(r))
        print("-" * 60)


def cmd_summary(cfg, args):
    dashboard.show(_paths()["results"], _paths()["logs"])


def cmd_live(cfg, args):
    refuse_live_execution()


def build_parser():
    p = argparse.ArgumentParser(prog="python -m src.main", description=BANNER)
    p.add_argument("--config", default="config.json", help="config file to use (default config.json)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("fetch-data", help="download real historical BTC candles (public, read-only)")
    s.add_argument("--days", type=int)
    s.set_defaults(func=cmd_fetch_data)

    s = sub.add_parser("import-csv", help="import a CryptoDataDownload hourly CSV (e.g. Bitstamp_BTCUSD_1h.csv)")
    s.add_argument("file")
    s.set_defaults(func=cmd_import_csv)

    s = sub.add_parser("make-sample-data", help="create SYNTHETIC data to test the pipeline offline")
    s.add_argument("--days", type=int)
    s.set_defaults(func=cmd_make_sample_data)

    sub.add_parser("research", help="temporal bias research (Step 2)").set_defaults(func=cmd_research)

    sub.add_parser("search", help="test many strategy types + honest walk-forward").set_defaults(func=cmd_search)

    s = sub.add_parser("backtest", help="walk-forward paper backtest (Steps 3-7, 9)")
    s.add_argument("--show", type=int, default=3, help="how many trade/blocked tickets to print")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("size", help="paper risk sizing calculator (Step 3)")
    s.add_argument("--account", type=float, required=True)
    s.add_argument("--win-prob", type=float, required=True, help="e.g. 0.45")
    s.add_argument("--avg-win", type=float, required=True)
    s.add_argument("--avg-loss", type=float, required=True, help="positive number")
    s.add_argument("--entry", type=float, required=True)
    s.add_argument("--stop", type=float, required=True)
    s.add_argument("--sample-size", type=int, required=True, help="trades behind the win-prob estimate")
    s.set_defaults(func=cmd_size)

    sub.add_parser("vol-check", help="volatility filter on the latest candle (Step 4)").set_defaults(func=cmd_vol_check)

    s = sub.add_parser("update", help="Bayesian probability update calculator (Step 5)")
    s.add_argument("--wins", type=int, required=True, help="wins in the new batch")
    s.add_argument("--losses", type=int, required=True, help="losses in the new batch")
    s.add_argument("--avg-win", type=float, default=1.5, help="average win (R or $)")
    s.add_argument("--avg-loss", type=float, default=1.0, help="average loss (R or $), positive")
    s.add_argument("--backtest-wins", type=int, default=0)
    s.add_argument("--backtest-losses", type=int, default=0)
    s.add_argument("--prior-trades-wins", type=int, default=0, help="earlier paper-trade wins")
    s.add_argument("--prior-trades-losses", type=int, default=0, help="earlier paper-trade losses")
    s.set_defaults(func=cmd_update)

    s = sub.add_parser("predict", help="prediction-market paper research (Step 8)")
    s.add_argument("--input", help="JSON file with markets (default from config)")
    s.set_defaults(func=cmd_predict)

    sub.add_parser("summary", help="terminal dashboard").set_defaults(func=cmd_summary)
    sub.add_parser("live", help="always refuses: this project cannot trade live").set_defaults(func=cmd_live)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(resolve_path(args.config))
        assert_paper_mode(cfg, load_env())
        print(BANNER)
        args.func(cfg, args)
    except LiveTradingRefused as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
