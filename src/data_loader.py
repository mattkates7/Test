"""Historical BTC candle data: load, download (public data only), or simulate.

CSV format (one row per hourly candle, oldest first):
    timestamp,open,high,low,close,volume
`timestamp` can be unix seconds or an ISO date-time; it is treated as UTC.

This is the ONLY module that touches the network, and it only calls a public,
read-only market-data endpoint. No API keys, no accounts, no orders.
"""

import csv
import json
import math
import random
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class Candle:
    ts: datetime  # candle open time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_ts(raw):
    raw = raw.strip()
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except ValueError:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _meta_path(csv_path):
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + ".meta.json")


def data_source(csv_path):
    """Return 'synthetic', 'coinbase', 'imported', etc. ('unknown' if no metadata)."""
    meta = _meta_path(csv_path)
    if meta.exists():
        return json.loads(meta.read_text()).get("source", "unknown")
    return "unknown"


def load_candles(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No data at {csv_path}. Run `python -m src.main fetch-data` (real public data) "
            "or `python -m src.main make-sample-data` (synthetic, for testing only)."
        )
    candles = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v for k, v in row.items()}
            candles.append(
                Candle(
                    ts=_parse_ts(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0.0),
                )
            )
    candles.sort(key=lambda c: c.ts)
    # Drop duplicate timestamps (can happen when stitching downloads).
    unique = []
    for c in candles:
        if not unique or c.ts != unique[-1].ts:
            unique.append(c)
    return unique


def save_candles(candles, csv_path, source):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.ts.isoformat(), c.open, c.high, c.low, c.close, c.volume])
    _meta_path(csv_path).write_text(
        json.dumps(
            {
                "source": source,
                "rows": len(candles),
                "first": candles[0].ts.isoformat() if candles else None,
                "last": candles[-1].ts.isoformat() if candles else None,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


def fetch_coinbase_hourly(product="BTC-USD", days=730, pause_s=0.35):
    """Download hourly candles from Coinbase's public market-data endpoint.

    Read-only, no API key. Coinbase returns at most 300 candles per request,
    so this walks backwards in 300-hour chunks.
    """
    base = f"https://api.exchange.coinbase.com/products/{product}/candles"
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_limit = end - timedelta(days=days)
    by_ts = {}
    chunk_end = end
    while chunk_end > start_limit:
        chunk_start = max(start_limit, chunk_end - timedelta(hours=300))
        url = (
            f"{base}?granularity=3600&start={chunk_start.isoformat()}"
            f"&end={chunk_end.isoformat()}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "paper-research/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read())
        # Each row: [time, low, high, open, close, volume]
        for t, low, high, open_, close, vol in rows:
            ts = datetime.fromtimestamp(t, tz=timezone.utc)
            by_ts[ts] = Candle(ts, float(open_), float(high), float(low), float(close), float(vol))
        print(f"  fetched {chunk_start:%Y-%m-%d} .. {chunk_end:%Y-%m-%d}  ({len(by_ts)} candles so far)")
        chunk_end = chunk_start
        time.sleep(pause_s)
    return [by_ts[k] for k in sorted(by_ts)]


def make_synthetic_candles(days=730, start_price=30000.0, seed=7):
    """Random-walk BTC-like hourly candles FOR TESTING THE PIPELINE ONLY.

    The data has no real time-of-day pattern on purpose, so a healthy research
    pipeline should report NO EDGE FOUND on it. If it 'finds' an edge here,
    that is a sign of overfitting.
    """
    rng = random.Random(seed)
    candles = []
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    price = start_price
    vol = 0.006
    for _ in range(days * 24):
        # Volatility clusters: occasional shocks that decay back to normal.
        if rng.random() < 0.002:
            vol = rng.uniform(0.015, 0.03)
        vol = 0.006 + (vol - 0.006) * 0.97
        ret = rng.gauss(0.00002, vol)
        open_ = price
        close = open_ * math.exp(ret)
        wick = abs(rng.gauss(0, vol * 0.6))
        high = max(open_, close) * (1 + wick)
        low = min(open_, close) * (1 - abs(rng.gauss(0, vol * 0.6)))
        volume = rng.lognormvariate(5, 0.4) * (1 + 40 * vol)
        candles.append(Candle(ts, open_, high, low, close, volume))
        price = close
        ts += timedelta(hours=1)
    return candles


def import_cryptodatadownload(src_path):
    """Read a CryptoDataDownload hourly CSV (e.g. Bitstamp_BTCUSD_1h.csv).

    Handles their quirks: a website banner on line 1, newest-first ordering,
    and older rows where the 'Volume BTC' and 'Volume USD' columns are swapped
    (BTC volume is always the smaller of the two, so we take the min).
    """
    with open(src_path, newline="") as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.lower().startswith("unix,"))
    candles = []
    for row in csv.DictReader(lines[start:]):
        vols = [float(v) for k, v in row.items() if k and k.lower().startswith("volume") and v]
        o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        if min(o, h, l, c) <= 0:
            continue  # skip broken rows
        ts = float(row["unix"])
        if ts > 1e12:  # some files use milliseconds
            ts /= 1000
        candles.append(Candle(datetime.fromtimestamp(ts, tz=timezone.utc), o, h, l, c, min(vols) if vols else 0.0))
    candles.sort(key=lambda x: x.ts)
    return candles


def resample(candles, hours):
    """Combine hourly candles into `hours`-long candles (24 = daily), aligned to UTC midnight."""
    if hours == 1:
        return candles
    out, bucket, key = [], [], None
    for c in candles:
        k = (c.ts.date(), c.ts.hour // hours)
        if bucket and k != key:
            out.append(_merge(bucket))
            bucket = []
        key = k
        bucket.append(c)
    if bucket:
        out.append(_merge(bucket))
    return out


def _merge(b):
    return Candle(b[0].ts, b[0].open, max(x.high for x in b), min(x.low for x in b),
                  b[-1].close, sum(x.volume for x in b))
