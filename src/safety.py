"""Paper-mode guard.

This project is SIMULATION ONLY. There is no broker, exchange, or wallet code
anywhere in it, and this module makes sure nobody can switch it to live mode by
flipping a setting. Enabling live trading would require rewriting the project.
"""

LIVE_TRADING_ENABLED = False  # Hard-coded. Not read from config or .env.

# .env variable names that suggest someone is trying to plug in real credentials.
FORBIDDEN_SECRET_HINTS = (
    "PRIVATE_KEY",
    "SEED_PHRASE",
    "MNEMONIC",
    "WALLET_KEY",
    "API_SECRET",
    "SECRET_KEY",
    "PASSPHRASE",
)

REFUSAL_MESSAGE = (
    "LIVE EXECUTION REFUSED. This project is PAPER MODE ONLY. It contains no "
    "exchange or wallet connection and will not send real orders. To trade live "
    "you would have to rewrite the project yourself; changing config or .env is "
    "not enough."
)


class LiveTradingRefused(RuntimeError):
    """Raised whenever anything other than paper simulation is requested."""


def refuse_live_execution(*_args, **_kwargs):
    """The only 'live' entry point in the project. It always refuses."""
    raise LiveTradingRefused(REFUSAL_MESSAGE)


def assert_paper_mode(config, env):
    """Stop the program unless config and environment are both in paper mode."""
    if LIVE_TRADING_ENABLED:  # pragma: no cover - constant is False by design
        raise LiveTradingRefused("LIVE_TRADING_ENABLED must stay False.")

    mode = str(config.get("mode", "")).strip().lower()
    if mode != "paper":
        raise LiveTradingRefused(f'config.json "mode" is "{mode}". ' + REFUSAL_MESSAGE)

    env_mode = str(env.get("TRADING_MODE", "paper")).strip().lower()
    if env_mode != "paper":
        raise LiveTradingRefused(f'TRADING_MODE (from .env or your shell) is "{env_mode}". ' + REFUSAL_MESSAGE)

    for name, value in env.items():
        if value and any(hint in name.upper() for hint in FORBIDDEN_SECRET_HINTS):
            raise LiveTradingRefused(
                f"Found {name} in your environment/.env. This project never needs wallet "
                "keys or exchange secrets. Remove it, then run again."
            )
