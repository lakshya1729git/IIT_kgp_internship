"""
label_events.py — Interactive ground-truth labeling tool
=========================================================
Shows real (non-synthetic) events one at a time and lets you confirm or
correct the LLM-assigned severity.  Results are written to the DB as
verified_severity + verified_by + verified_at columns.

Controls:
  l → label as LOW
  m → label as MEDIUM
  h → label as HIGH
  Enter (blank) → AGREE with LLM label (same as pressing l/m/h for current)
  s → skip this event (no label written)
  q → quit and save progress

Run from app/ directory:
  python label_events.py              # label up to 150 real events
  python label_events.py --limit 50   # label only 50
  python label_events.py --reset      # clear all verified labels and start over
  python label_events.py --stats      # show labeling progress, then exit

Events are shown in a consistent order (by id DESC) so you can quit and
resume — already-labeled events are skipped automatically.
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timezone

DB_URL = "sqlite:///traffic_events.db"

SEV_COLORS = {"low": "\033[92m", "medium": "\033[93m", "high": "\033[91m"}
RESET      = "\033[0m"
BOLD       = "\033[1m"
DIM        = "\033[2m"
CYAN       = "\033[96m"
YELLOW     = "\033[93m"
GREEN      = "\033[92m"


def _colored(text: str, color: str) -> str:
    """Apply ANSI color — falls back to plain text on Windows without color support."""
    try:
        return f"{color}{text}{RESET}"
    except Exception:
        return text


def _migrate_db(conn) -> None:
    """Add verified_severity, verified_by, verified_at columns if missing."""
    from sqlalchemy import text
    existing = {row[1] for row in conn.execute(text("PRAGMA table_info(traffic_events)"))}
    new_cols = [
        ("verified_severity", "VARCHAR(10)"),
        ("verified_by",       "VARCHAR(60)"),
        ("verified_at",       "DATETIME"),
    ]
    for col, col_type in new_cols:
        if col not in existing:
            conn.execute(text(f"ALTER TABLE traffic_events ADD COLUMN {col} {col_type}"))
            print(f"  [DB] Added column '{col}'")
    conn.commit()


def _show_stats(conn) -> None:
    """Print labeling progress to stdout."""
    from sqlalchemy import text
    total = conn.execute(text(
        "SELECT COUNT(*) FROM traffic_events WHERE raw_text NOT LIKE '[SYNTHETIC]%'"
    )).scalar()
    labeled = conn.execute(text(
        "SELECT COUNT(*) FROM traffic_events "
        "WHERE verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%'"
    )).scalar()
    agree = conn.execute(text(
        "SELECT COUNT(*) FROM traffic_events "
        "WHERE verified_severity = severity AND verified_severity IS NOT NULL "
        "AND raw_text NOT LIKE '[SYNTHETIC]%'"
    )).scalar()
    dist = conn.execute(text(
        "SELECT verified_severity, COUNT(*) FROM traffic_events "
        "WHERE verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%' "
        "GROUP BY verified_severity"
    )).fetchall()
    corrections = conn.execute(text(
        "SELECT severity, verified_severity, COUNT(*) FROM traffic_events "
        "WHERE verified_severity IS NOT NULL AND verified_severity != severity "
        "AND raw_text NOT LIKE '[SYNTHETIC]%' "
        "GROUP BY severity, verified_severity ORDER BY COUNT(*) DESC"
    )).fetchall()

    print(f"\n{'─'*55}")
    print(f"  Labeling Progress")
    print(f"{'─'*55}")
    print(f"  Total real events : {total}")
    print(f"  Labeled so far    : {labeled}  ({100*labeled/total:.1f}%)")
    print(f"  Agreements        : {agree}/{labeled}  ({100*agree/labeled:.1f}% agree)" if labeled else "  Agreements: 0/0")
    if dist:
        print(f"  Verified dist     : {dict(dist)}")
    if corrections:
        print(f"  Corrections (LLM→you):")
        for llm_sev, your_sev, cnt in corrections:
            print(f"    {llm_sev:8s} → {your_sev:8s}: {cnt} times")
    print(f"{'─'*55}\n")


def _load_unlabeled(conn, limit: int) -> list[tuple]:
    """Load unlabeled real events, most recent first, up to limit."""
    from sqlalchemy import text
    rows = conn.execute(text(
        "SELECT id, event_type, severity, confidence, road_name, location, reason, raw_text, source "
        "FROM traffic_events "
        "WHERE verified_severity IS NULL AND raw_text NOT LIKE '[SYNTHETIC]%' "
        "ORDER BY id DESC "
        "LIMIT :lim"
    ), {"lim": limit}).fetchall()
    return rows


def _write_label(conn, event_id: int, verified_sev: str, labeler: str) -> None:
    from sqlalchemy import text
    conn.execute(text(
        "UPDATE traffic_events "
        "SET verified_severity = :vs, verified_by = :vb, verified_at = :va "
        "WHERE id = :id"
    ), {
        "vs": verified_sev,
        "vb": labeler,
        "va": datetime.now(timezone.utc).isoformat(),
        "id": event_id,
    })
    conn.commit()


def _clear_labels(conn) -> None:
    from sqlalchemy import text
    conn.execute(text(
        "UPDATE traffic_events SET verified_severity = NULL, verified_by = NULL, verified_at = NULL "
        "WHERE raw_text NOT LIKE '[SYNTHETIC]%'"
    ))
    conn.commit()
    print("  All verified labels cleared.")


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    """Simple word-wrap for long reason/raw_text fields."""
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(indent + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)


def run_labeling(limit: int, labeler: str) -> None:
    from sqlalchemy import create_engine

    engine = create_engine(DB_URL, echo=False)
    conn   = engine.connect()
    _migrate_db(conn)

    rows = _load_unlabeled(conn, limit)
    total_to_label = len(rows)

    if total_to_label == 0:
        print("\n  All events in your limit are already labeled.")
        _show_stats(conn)
        conn.close()
        return

    # ── Labeling guidelines ────────────────────────────────────────────────
    print(f"""
{'═'*60}
  KOLKATA TRAFFIC EVENT — SEVERITY LABELING
{'═'*60}
  Severity Guide:
    LOW    — minor delay (<15 min), no lane closure, no diversion
             e.g. slow traffic, minor roadwork, brief congestion
    MEDIUM — noticeable delay (15–60 min), partial closure or diversion
             e.g. one lane blocked, waterlogging on side road, protest
    HIGH   — major disruption (>60 min or full closure)
             e.g. road blocked, bridge closed, flood, large protest

  Controls:
    l / Enter(agree)  → LOW
    m                 → MEDIUM
    h                 → HIGH
    s                 → skip (don't label)
    q                 → quit and save progress
{'─'*60}
  {total_to_label} events to label.  Labeler: {labeler}
{'─'*60}
""")

    labeled = skipped = 0
    agreements = corrections = 0

    for i, row in enumerate(rows, 1):
        ev_id, ev_type, llm_sev, conf, road, location, reason, raw_text, source = row

        # ── Display event ──────────────────────────────────────────────────
        os.system("cls" if os.name == "nt" else "clear")

        sev_color = SEV_COLORS.get(llm_sev, "")
        print(f"\n{'═'*60}")
        print(f"  Event {i}/{total_to_label}   (id={ev_id})   labeled={labeled}  skipped={skipped}")
        print(f"{'─'*60}")
        print(f"  {BOLD}Type     :{RESET} {ev_type}")
        print(f"  {BOLD}Road     :{RESET} {road or '(unknown)'}")
        print(f"  {BOLD}Location :{RESET} {location or '(unknown)'}")
        print(f"  {BOLD}Source   :{RESET} {source}")
        print(f"  {BOLD}LLM conf :{RESET} {conf:.2f}")
        print(f"  {BOLD}Reason   :{RESET}")
        print(_wrap(reason or "(no reason)", width=90, indent="    "))
        print(f"\n  {BOLD}Raw text :{RESET}")
        print(_wrap(str(raw_text)[:400], width=90, indent="    "))
        print(f"{'─'*60}")
        print(f"  {BOLD}LLM severity :{RESET} "
              f"{sev_color}{BOLD}{llm_sev.upper()}{RESET}")
        print(f"{'─'*60}")
        print(f"  {CYAN}[l]{RESET} LOW   "
              f"{CYAN}[m]{RESET} MEDIUM   "
              f"{CYAN}[h]{RESET} HIGH   "
              f"{DIM}[s]{RESET} skip   "
              f"{DIM}[q]{RESET} quit")
        print(f"  (Enter = agree with LLM: {llm_sev.upper()})")

        # ── Get input ──────────────────────────────────────────────────────
        while True:
            try:
                key = input("  Your label: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"

            if key == "q":
                print(f"\n  Quit. Labeled {labeled}, skipped {skipped}.")
                _show_stats(conn)
                conn.close()
                return
            elif key == "s":
                skipped += 1
                break
            elif key in ("l", ""):
                verified = "low"
                if verified == llm_sev:
                    agreements += 1
                    print(f"  {GREEN}✓ Agreed: LOW{RESET}")
                else:
                    corrections += 1
                    print(f"  {YELLOW}↻ Corrected: {llm_sev.upper()} → LOW{RESET}")
                _write_label(conn, ev_id, verified, labeler)
                labeled += 1
                break
            elif key == "m":
                verified = "medium"
                if verified == llm_sev:
                    agreements += 1
                    print(f"  {GREEN}✓ Agreed: MEDIUM{RESET}")
                else:
                    corrections += 1
                    print(f"  {YELLOW}↻ Corrected: {llm_sev.upper()} → MEDIUM{RESET}")
                _write_label(conn, ev_id, verified, labeler)
                labeled += 1
                break
            elif key == "h":
                verified = "high"
                if verified == llm_sev:
                    agreements += 1
                    print(f"  {GREEN}✓ Agreed: HIGH{RESET}")
                else:
                    corrections += 1
                    print(f"  {YELLOW}↻ Corrected: {llm_sev.upper()} → HIGH{RESET}")
                _write_label(conn, ev_id, verified, labeler)
                labeled += 1
                break
            else:
                print(f"  Unknown key '{key}'. Use l / m / h / s / q.")

        # Brief pause so feedback is visible before next clear
        if key not in ("q", "s"):
            import time
            time.sleep(0.4)

    print(f"\n{'═'*60}")
    print(f"  Done. Labeled {labeled}/{total_to_label}.  Agreements: {agreements}  Corrections: {corrections}")
    _show_stats(conn)
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Label traffic event severity for HGNN ground truth")
    parser.add_argument("--limit",  type=int,   default=150,
                        help="Max events to label in this session (default 150)")
    parser.add_argument("--labeler",type=str,   default="human",
                        help="Your name/ID stored with each label (default 'human')")
    parser.add_argument("--stats",  action="store_true",
                        help="Show labeling progress and exit")
    parser.add_argument("--reset",  action="store_true",
                        help="Clear all existing verified labels and exit")
    args = parser.parse_args()

    from sqlalchemy import create_engine
    engine = create_engine(DB_URL, echo=False)
    conn   = engine.connect()
    _migrate_db(conn)

    if args.stats:
        _show_stats(conn)
        conn.close()
        return

    if args.reset:
        confirm = input("  Clear ALL verified labels? (yes/no): ").strip().lower()
        if confirm == "yes":
            _clear_labels(conn)
        else:
            print("  Cancelled.")
        conn.close()
        return

    conn.close()
    run_labeling(limit=args.limit, labeler=args.labeler)


if __name__ == "__main__":
    main()
