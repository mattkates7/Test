"""Prediction-Market Research (Step 8). PAPER MODE ONLY. Kept separate from BTC.

This module does NOT connect to any prediction-market site or wallet. You
write the markets and the evidence you found into a JSON file
(see data/prediction_markets.example.json) and it applies strict rules:

  * A market price is NOT wrong just because a model (or you) disagrees.
  * Your estimate must be a RANGE, not a single number.
  * Evidence older than `max_source_age_days` counts as stale.
  * No fresh strong/moderate evidence -> SKIP.
  * Paper YES/NO only if the WHOLE estimated range clears the implied
    probability by more than min_edge + fees. Otherwise SKIP.
  * Confidence is never 'certain'. Ranges are clamped to 1%..99%.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path

QUALITY_SCORE = {"strong": 2, "moderate": 1, "weak": 0}


def _parse_date(s):
    return date.fromisoformat(str(s)[:10])


def _fresh(evidence, as_of, max_age):
    out = []
    for e in evidence:
        try:
            age = (as_of - _parse_date(e["date"])).days
        except (KeyError, ValueError):
            continue
        if 0 <= age <= max_age:
            out.append(e)
    return out


def analyze_market(m, cfg, as_of):
    max_age = cfg["max_source_age_days"]
    min_edge = cfg["min_edge_pct"] / 100
    fee = cfg["fee_pct"] / 100

    implied = float(m["implied_probability"])
    ev_yes, ev_no = m.get("evidence_yes", []), m.get("evidence_no", [])
    fresh_yes = _fresh(ev_yes, as_of, max_age)
    fresh_no = _fresh(ev_no, as_of, max_age)
    solid = [e for e in fresh_yes + fresh_no if e.get("quality") in ("strong", "moderate")]
    stale_count = len(ev_yes) + len(ev_no) - len(fresh_yes) - len(fresh_no)

    low, high = m.get("my_estimate_low"), m.get("my_estimate_high")
    reasons = []
    decision = "SKIP"
    if low is None or high is None:
        reasons.append("no estimated range provided")
    else:
        low, high = max(0.01, float(low)), min(0.99, float(high))
        if low > high:
            low, high = high, low
    if not 0 < implied < 1:
        reasons.append("implied probability must be between 0 and 1")
    if not solid:
        reasons.append(f"no fresh (<= {max_age} days) strong or moderate evidence")
    if stale_count:
        reasons.append(f"{stale_count} evidence item(s) are stale and were ignored")
    if not m.get("resolution_criteria"):
        reasons.append("resolution criteria not written down")

    blocking = [r for r in reasons if "stale" not in r]
    if not blocking:
        hurdle = min_edge + fee
        if low - implied > hurdle:
            decision = "YES"
            reasons.append(f"entire range {low:.0%}-{high:.0%} is above implied {implied:.0%} by > {hurdle:.0%}")
        elif implied - high > hurdle:
            decision = "NO"
            reasons.append(f"entire range {low:.0%}-{high:.0%} is below implied {implied:.0%} by > {hurdle:.0%}")
        else:
            reasons.append(
                f"range {low:.0%}-{high:.0%} does not clear implied {implied:.0%} by the "
                f"{hurdle:.0%} hurdle (edge + fees); the market may simply be right"
            )

    width = (high - low) if low is not None and high is not None else 1.0
    strong_fresh = sum(1 for e in solid if e.get("quality") == "strong")
    if width <= 0.10 and strong_fresh >= 3:
        confidence = "HIGH"
    elif width <= 0.20 and len(solid) >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    if decision != "SKIP" and confidence == "LOW":
        decision = "SKIP"
        reasons.append("confidence LOW, so no paper position")

    def best(evs):
        evs = sorted(evs, key=lambda e: QUALITY_SCORE.get(e.get("quality"), 0), reverse=True)
        return [f"[{e.get('quality')}, {e.get('date')}] {e.get('claim')} ({e.get('source')})" for e in evs[:3]]

    return {
        "market": m["market"],
        "implied_probability": implied,
        "estimated_range": None if low is None or high is None else [low, high],
        "strongest_yes": best(fresh_yes),
        "strongest_no": best(fresh_no),
        "key_uncertainties": m.get("unknowns", []),
        "upcoming_events": m.get("upcoming_events", []),
        "paper_decision": decision,
        "confidence": confidence,
        "reasons": reasons,
    }


def format_result(r):
    rng = r["estimated_range"]
    lines = [
        f"MARKET: {r['market']}",
        f"IMPLIED PROBABILITY: {r['implied_probability']:.0%}",
        f"ESTIMATED RANGE: {'n/a' if not rng else f'{rng[0]:.0%} - {rng[1]:.0%}'}",
        "KEY EVIDENCE:",
        *[f"  YES: {e}" for e in r["strongest_yes"] or ["(no fresh evidence)"]],
        *[f"  NO:  {e}" for e in r["strongest_no"] or ["(no fresh evidence)"]],
        "KEY UNCERTAINTIES:",
        *[f"  - {u}" for u in r["key_uncertainties"] or ["(none listed - that is itself a warning)"]],
        "UPCOMING EVENTS:",
        *[f"  - {u}" for u in r["upcoming_events"] or ["(none listed)"]],
        f"PAPER DECISION: {r['paper_decision']}",
        f"CONFIDENCE: {r['confidence']}",
        "WHY:",
        *[f"  - {x}" for x in r["reasons"]],
    ]
    return "\n".join(lines)


def run(cfg, input_path, logs_dir):
    data = json.loads(Path(input_path).read_text())
    as_of = _parse_date(data["as_of"]) if data.get("as_of") else datetime.now(timezone.utc).date()
    results = []
    with open(Path(logs_dir) / "prediction_market_paper.jsonl", "a") as log:
        for m in data["markets"]:
            r = analyze_market(m, cfg, as_of)
            results.append(r)
            log.write(json.dumps({"logged_at": datetime.now(timezone.utc).isoformat(),
                                  "as_of": as_of.isoformat(), **r}) + "\n")
    return as_of, results
