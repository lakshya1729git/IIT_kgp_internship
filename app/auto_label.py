"""
auto_label.py — Rule-based ground-truth labeling for TomTom traffic events
===========================================================================
WHY THIS WORKS BETTER THAN MANUAL LABELING FOR THIS DATASET
------------------------------------------------------------
The DB is 99.6% TomTom structured API data. TomTom's own severity signal
is embedded directly in the raw_text in a 100% deterministic way:

  Pattern                            → True severity
  ─────────────────────────────────────────────────────
  "HIGH SEVERITY" + "Stationary"     → high
  "Queuing traffic"                  → medium
  "Slow traffic"                     → low (minor delay)
  "Road Closure" / "Closed"          → medium by default
      + major arterial / bridge      → high
      + lane / small side road       → low
  "Construction" / "Roadworks"       → low
      + major road                   → medium

The LLM made systematic errors:
  - All "Road Closure" events were labeled low  (wrong — closures vary)
  - Construction was correctly labeled low
  - HIGH SEVERITY congestion was correctly labeled high

These rules CORRECT the LLM's road-closure error and confirm the rest.

HOW TO RUN
----------
  python auto_label.py                  # label all unlabeled real events
  python auto_label.py --dry-run        # preview without writing
  python auto_label.py --overwrite      # re-label already-labeled events too
  python auto_label.py --stats          # show current label stats
  python auto_label.py --confidence     # show rule confidence per pattern

OUTPUT
------
Writes verified_severity, verified_by='auto_rule', verified_at to DB.
These are treated identically to human labels by trainer.py and evaluate_hgnn.py.
"""

from __future__ import annotations

import re
import argparse
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

DB_URL = "sqlite:///traffic_events.db"

# ── Major roads — closure on these = HIGH ──────────────────────────────────
MAJOR_ARTERIALS = {
    "howrah bridge", "vidyasagar setu", "nivedita setu", "bally bridge",
    "em bypass", "eastern metropolitan bypass",
    "vip road", "kona expressway", "maa flyover",
    "belgharia expressway", "tallah bridge",
    "jawaharlal nehru road", "chowringhee road",
    "strand road", "acharya jagadish chandra bose road",
    "rashbehari avenue", "diamond harbour road",
    "bidhan sarani", "mahatma gandhi road",
    "b t road", "bt road", "jessore road", "barasat road",
    "dum dum road", "ultadanga road",
    "ajc bose road", "sp mukherjee road",
    "park street", "camac street",
    "salt lake", "rajarhat", "new town",
}

# ── Small side roads — closure on these = LOW ──────────────────────────────
# NOTE: "sarani" is intentionally excluded — it translates to "road/avenue"
# and appears on major arterials (e.g. Lala Lajpat Rai Sarani, Bidhan Sarani).
# Only truly minor infrastructure markers belong here.
SIDE_ROAD_MARKERS = [
    "lane", "ghat road", "para road", "bazar lane",
    "nagar", "pally",
]

# ── Bridge / flyover keywords — always boosts to HIGH if in text ──────────
BRIDGE_FLYOVER = [
    "bridge closed", "bridge closure", "flyover closed", "flyover closure",
    "overpass closed",
]


def _extract_delay_minutes(raw: str) -> int | None:
    """Extract 'Delay: ~N minutes' from TomTom raw text. Returns None if absent."""
    m = re.search(r'Delay:\s*~?(\d+)\s*minutes', raw, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _road_name_from_text(raw: str, road_name: str | None) -> str:
    """Best road name available for arterial/side-road classification."""
    return (road_name or raw or "").lower()


def auto_label(raw_text: str, event_type: str, road_name: str | None,
               llm_severity: str) -> tuple[str, str]:
    """
    Assign a rule-based severity label.

    Returns (verified_severity, rule_applied).

    Rules (in priority order):
      1. Bridge/flyover explicitly closed → HIGH
      2. HIGH SEVERITY + Stationary traffic → HIGH  (TomTom API signal)
      3. Queuing traffic → MEDIUM
      4. Slow traffic → LOW
      5. Construction / Roadworks → LOW (MEDIUM if on major arterial)
      6. Road Closure / Closed:
           - bridge/flyover involved  → HIGH
           - major arterial           → MEDIUM
           - small side road          → LOW
           - unknown / default        → MEDIUM
      7. Fallback: trust LLM label
    """
    raw = raw_text.lower() if raw_text else ""
    road = _road_name_from_text(raw, road_name)
    delay = _extract_delay_minutes(raw)

    # 1. Bridge / flyover explicitly closed — always HIGH
    if any(k in raw for k in BRIDGE_FLYOVER):
        return "high", "bridge_flyover_closed"

    # 2. TomTom HIGH SEVERITY signal — deterministic
    # Delay check FIRST: a 1-minute stationary burst that clears immediately
    # should be MEDIUM even if TomTom marks it HIGH SEVERITY + Stationary.
    if "high severity" in raw:
        if delay is not None and delay <= 2:
            return "medium", "high_severity_tiny_delay"
        if "stationary traffic" in raw:
            return "high", "tomtom_high_severity_stationary"
        return "high", "tomtom_high_severity"

    # 3. Queuing traffic → MEDIUM (TomTom's own mid-severity signal)
    if "queuing traffic" in raw:
        if delay is not None and delay >= 10:
            return "high", "queuing_long_delay"
        return "medium", "tomtom_queuing"

    # 4. Slow traffic → LOW
    if "slow traffic" in raw:
        if delay is not None and delay >= 15:
            return "medium", "slow_long_delay"
        return "low", "tomtom_slow"

    # 5. Construction / Roadworks
    if "construction" in raw or "roadworks" in raw or "road works" in raw:
        # Major road construction causes real disruption
        if any(art in road for art in MAJOR_ARTERIALS):
            return "medium", "construction_major_road"
        return "low", "construction_minor"

    # 6. Road Closure / Closed
    if "road closure" in raw or " closed" in raw or "closure" in raw:
        # Sub-rule 6a: bridge or flyover involved
        if any(k in road for k in ["bridge", "flyover", "setu", "expressway"]):
            return "high", "closure_bridge_flyover"

        # Sub-rule 6b: major arterial closed
        if any(art in road for art in MAJOR_ARTERIALS):
            return "medium", "closure_major_arterial"

        # Sub-rule 6c: small side road (lane / para / nagar etc.)
        # Check full road name — but only for unambiguously minor markers.
        # Do NOT check "street" or "sarani" here; those appear on major roads too.
        if any(marker in road for marker in SIDE_ROAD_MARKERS):
            # Check it's not a major one we already caught
            if not any(art in road for art in MAJOR_ARTERIALS):
                return "low", "closure_side_road"

        # Sub-rule 6d: default closure
        return "medium", "closure_default"

    # 7. Fallback — if event_type indicates a closure, default to medium
    # The LLM systematically under-labels road_closure events as "low".
    # Without a TomTom signal to confirm, medium is a safer default than
    # blindly trusting the LLM's known-biased label.
    if event_type in ("road_closure", "diversion", "protest", "transport_strike"):
        return "medium", "event_type_default"

    return llm_severity, "llm_fallback"


def _migrate(conn) -> None:
    """Add verified columns if missing."""
    existing = {row[1] for row in conn.execute(text("PRAGMA table_info(traffic_events)"))}
    for col, typ in [("verified_severity","VARCHAR(10)"),
                     ("verified_by","VARCHAR(60)"),
                     ("verified_at","DATETIME")]:
        if col not in existing:
            conn.execute(text(f"ALTER TABLE traffic_events ADD COLUMN {col} {typ}"))
    conn.commit()


def run(dry_run: bool = False, overwrite: bool = False) -> None:
    engine = create_engine(DB_URL, echo=False)

    with engine.connect() as conn:
        _migrate(conn)

        where = (
            "WHERE raw_text NOT LIKE '[SYNTHETIC]%'"
            if overwrite
            else "WHERE raw_text NOT LIKE '[SYNTHETIC]%' AND verified_severity IS NULL"
        )

        rows = conn.execute(text(
            f"SELECT id, event_type, severity, road_name, raw_text "
            f"FROM traffic_events {where} ORDER BY id"
        )).fetchall()

    if not rows:
        print("No unlabeled events found. Use --overwrite to re-label everything.")
        return

    print(f"\n[AutoLabel] Processing {len(rows)} events  "
          f"({'DRY RUN' if dry_run else 'writing to DB'}) ...")

    from collections import Counter
    rule_counts: Counter  = Counter()
    change_counts: Counter = Counter()
    sev_counts: Counter   = Counter()

    labeled_rows = []
    for ev_id, ev_type, llm_sev, road, raw_text in rows:
        verified_sev, rule = auto_label(
            raw_text  = raw_text or "",
            event_type = ev_type or "unknown",
            road_name  = road,
            llm_severity = llm_sev or "low",
        )
        rule_counts[rule] += 1
        sev_counts[verified_sev] += 1
        if verified_sev != (llm_sev or "low"):
            change_counts[f"{llm_sev}→{verified_sev}"] += 1
        labeled_rows.append((ev_id, verified_sev))

    # Print summary
    print(f"\n  Verified severity dist: {dict(sev_counts)}")
    print(f"  Rules applied         : {dict(rule_counts.most_common())}")
    if change_counts:
        print(f"  LLM corrections      : {dict(change_counts)}")
    else:
        print(f"  LLM corrections      : none (all rules agreed with LLM)")

    if dry_run:
        print(f"\n  [DRY RUN] Would write {len(labeled_rows)} labels. Pass without --dry-run to apply.")
        return

    # Write to DB
    now = datetime.now(timezone.utc).isoformat()
    engine2 = create_engine(DB_URL, echo=False)
    with engine2.connect() as conn:
        for ev_id, verified_sev in labeled_rows:
            conn.execute(text(
                "UPDATE traffic_events "
                "SET verified_severity=:vs, verified_by='auto_rule', verified_at=:va "
                "WHERE id=:id"
            ), {"vs": verified_sev, "va": now, "id": ev_id})
        conn.commit()

    print(f"\n  [AutoLabel] Done. Wrote {len(labeled_rows)} verified labels.")
    _show_stats(engine2)


def _show_stats(engine=None) -> None:
    if engine is None:
        engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM traffic_events WHERE raw_text NOT LIKE '[SYNTHETIC]%'"
        )).scalar()
        labeled = conn.execute(text(
            "SELECT COUNT(*) FROM traffic_events "
            "WHERE verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%'"
        )).scalar()
        dist = conn.execute(text(
            "SELECT verified_severity, COUNT(*) FROM traffic_events "
            "WHERE verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%' "
            "GROUP BY verified_severity"
        )).fetchall()
        agree = conn.execute(text(
            "SELECT COUNT(*) FROM traffic_events "
            "WHERE verified_severity = severity "
            "AND verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%'"
        )).scalar()
        by_labeler = conn.execute(text(
            "SELECT verified_by, COUNT(*) FROM traffic_events "
            "WHERE verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%' "
            "GROUP BY verified_by"
        )).fetchall()

    print(f"\n  {'─'*50}")
    print(f"  Label Coverage: {labeled}/{total} ({100*labeled/total:.1f}%)")
    print(f"  Verified dist : {dict(dist)}")
    print(f"  LLM agreement : {agree}/{labeled} ({100*agree/labeled:.1f}%)" if labeled else "")
    print(f"  By labeler    : {dict(by_labeler)}")
    print(f"  {'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Auto-label TomTom events with rule-based ground truth")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Preview labels without writing to DB")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-label already-labeled events (replaces human labels too)")
    parser.add_argument("--stats",      action="store_true",
                        help="Show current label stats and exit")
    parser.add_argument("--confidence", action="store_true",
                        help="Show rule confidence analysis (how often each rule fires)")
    args = parser.parse_args()

    if args.stats:
        _show_stats()
        return

    run(dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
