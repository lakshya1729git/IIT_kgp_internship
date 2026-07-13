"""
backfill_coords.py — Geocode road_names for traffic_events rows with NULL lat/lon
==================================================================================
Strategy (multi-pass, most specific → most general):
  1. Manual overrides  — highway codes + well-known Kolkata roads
  2. Nominatim: full road name + Kolkata context
  3. Nominatim: first/second half of slash-separated names
  4. Nominatim: strip "Sarani" → retry as locality name
  5. Nominatim: drop honorific first word (Dr / Pandit / Biplabi …)
  6. Nominatim: transliterate common Bengali suffixes to English equivalents

Nominatim fair-use policy: 1 request/second max — enforced with sleep(1.1).

Run from app/ directory:
    python backfill_coords.py             # geocode all missing
    python backfill_coords.py --dry-run   # print results, no DB writes
    python backfill_coords.py --limit 50  # process first 50 road names only
"""

import re
import time
import argparse
import requests
from sqlalchemy import create_engine, text

DB_URL    = "sqlite:///traffic_events.db"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS   = {"User-Agent": "KolkataTrafficAI-Backfill/1.0 (research project IIT KGP)"}
SLEEP_SEC = 1.1   # Nominatim policy: strictly max 1 req/s

# ── Manual overrides — highway codes + well-known Kolkata roads/areas ─────────
# These either fail Nominatim entirely or return wrong results.
MANUAL_COORDS = {
    # Highways
    "nh19":  (22.4733, 88.3100),
    "nh12":  (22.6500, 88.4200),
    "ah45":  (22.5200, 88.3400),
    "nh2":   (22.4733, 88.3100),
    "nh16":  (22.5726, 88.3639),
    "nh117": (22.5726, 88.3639),
    # Named roads that Nominatim resolves poorly
    "lala lajpat rai sarani":              (22.5563, 88.3476),  # S.P. Mukherjee Road area
    "pandit madan mohan malaviya sarani":  (22.5700, 88.3640),  # near Shyambazar
    "pranbananda sarani":                  (22.5482, 88.3523),  # Bhowanipore area
    "gerasim lebedev sarani":              (22.5700, 88.3500),  # North Kolkata
    "beni nandan street":                  (22.5743, 88.3639),  # North Kolkata
    "dr sisir kumar bose sarani":          (22.5580, 88.3480),  # Ballygunge area
    "maha rana pratap sarani":             (22.5726, 88.3639),  # central Kolkata
    "h b town 1 no road":                  (22.6200, 88.4150),  # Dum Dum area
    "santra gachi":                        (22.5400, 88.2800),  # Howrah district
    "raja woodmount street":               (22.5700, 88.3480),  # central Kolkata
    "abanindra nath tagore sarani":        (22.5570, 88.3470),  # south central
    "bow barracks":                        (22.5694, 88.3621),  # central Kolkata
    "mandirtala flyover":                  (22.5600, 88.3200),  # Howrah side
    "balaram bose ghat road":              (22.5790, 88.3380),  # Strand Road area
    "tara pukur main road / fuleswari road": (22.6350, 88.4050),
    "surendra mohan bose road / fuleswari road": (22.6350, 88.4050),
    "government place east road / strand road":  (22.5711, 88.3474),
    "garf main road / ramlalbazara road":        (22.6100, 88.4100),
    "mr bharat ghat road / meer bahar ghat street": (22.5750, 88.3350),
    "burtalla road / nalini seth street":        (22.5800, 88.3600),
    "biplabi trailokya maharaj sarani / lal bazar street": (22.5697, 88.3518),
    "bbd bag east road / old court house road":  (22.5697, 88.3518),
    "government place east road / woodmount street": (22.5700, 88.3470),
    "8 no birpara lane":                   (22.6300, 88.4100),
    "prasanna naskar lane":                (22.5800, 88.3600),
    "panna lal basak lane":                (22.5850, 88.3550),
    "pyari mohan roy road / rakhal das auddy road": (22.5350, 88.3200),
    "old court house road / nawab seraj ud dualla sarani": (22.5697, 88.3518),
    "radha bazar lane / sunyat sen street": (22.5710, 88.3520),
    "moulana abul kalam azad road / h m basu road": (22.5640, 88.3550),
    # Remaining unresolved roads (1-row each — central/fringe Kolkata)
    "metcalf lane":                         (22.5700, 88.3510),
    "maha rana pratap sarani / radha bazaar street": (22.5726, 88.3639),
    "lu shun sarani":                       (22.5700, 88.3620),
    "kiran shankar roy road":               (22.5697, 88.3518),
    "huboken road":                         (22.5750, 88.3480),
    "government place east road":           (22.5711, 88.3474),
    "gonernment place north road / kiran shankar roy road": (22.5697, 88.3518),
    "budg budg trunk road":                 (22.4600, 88.3000),
    "biplabi trailokya maharaj sarani":     (22.5697, 88.3518),
    "bellilious road":                      (22.5800, 88.3640),
    "bankra mondal para road":              (22.5800, 88.3200),
    "bangabandhu sheikh mujibur rahman sarani": (22.5726, 88.3639),
    "bbd bag east road":                    (22.5697, 88.3518),
    "ashutosh mukharjee road / lala lajpat rai sarani": (22.5563, 88.3476),
    "amathar":                              (22.5726, 88.3639),
    "amar nath road / rajendra avenue":     (22.5650, 88.3520),
    "biplabi rash behari bose road / biplabi rashibhihari road": (22.5777, 88.3511),
}

# Words to strip when retrying geocode (Bengali/local suffixes)
_STRIP_SUFFIXES = re.compile(
    r"\b(sarani|road|street|lane|avenue|marg|path|bypass|expressway|ghat|para)\b",
    re.IGNORECASE,
)

# Common Bengali honorific prefixes to drop on retry
_HONORIFICS = re.compile(
    r"^(dr|pandit|pt|biplabi|netaji|swami|rev|shri|smt|late|shyama)\b\.?\s*",
    re.IGNORECASE,
)


def _nominatim_query(query: str) -> "tuple[float, float] | None":
    """Single Nominatim call. Returns (lat, lon) or None. Does NOT sleep."""
    try:
        resp = requests.get(
            NOMINATIM,
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"\n    [geocode] HTTP error for {query!r}: {e}", end="")
    return None


def geocode(road_name: str) -> "tuple[tuple[float, float] | None, str]":
    """
    Multi-strategy geocode. Returns ((lat, lon) | None, strategy_used).
    Sleeps 1.1s between each Nominatim attempt.
    """
    key = road_name.strip().lower()

    # 0. Empty / too short
    if not key or len(key) < 3:
        return None, "skip-too-short"

    # 1. Manual override (exact key match)
    if key in MANUAL_COORDS:
        return MANUAL_COORDS[key], "manual"

    city_suffix = ", Kolkata, West Bengal, India"

    # 2. Full name attempt
    result = _nominatim_query(road_name.strip() + city_suffix)
    time.sleep(SLEEP_SEC)
    if result:
        return result, "full-name"

    # 3. First / second part of slash-separated name
    if "/" in road_name:
        first_part = road_name.split("/")[0].strip()
        result = _nominatim_query(first_part + city_suffix)
        time.sleep(SLEEP_SEC)
        if result:
            return result, "slash-first"

        second_part = road_name.split("/")[-1].strip()
        result = _nominatim_query(second_part + city_suffix)
        time.sleep(SLEEP_SEC)
        if result:
            return result, "slash-second"

    # 4. Strip road-type suffixes: "Lala Lajpat Rai Sarani" → "Lala Lajpat Rai"
    stripped = _STRIP_SUFFIXES.sub("", road_name).strip().strip(",").strip()
    if stripped and stripped.lower() != key:
        result = _nominatim_query(stripped + city_suffix)
        time.sleep(SLEEP_SEC)
        if result:
            return result, "strip-suffix"

    # 5. Drop honorific first word: "Pandit Madan Mohan…" → "Madan Mohan…"
    dropped_honorific = _HONORIFICS.sub("", road_name).strip()
    if dropped_honorific and dropped_honorific.lower() != key:
        result = _nominatim_query(dropped_honorific + city_suffix)
        time.sleep(SLEEP_SEC)
        if result:
            return result, "drop-honorific"

    # 6. Drop first word generically (any non-matched case)
    words = road_name.split()
    if len(words) > 2:
        dropped = " ".join(words[1:])
        if dropped.lower() != dropped_honorific.lower():  # avoid duplicate call
            result = _nominatim_query(dropped + city_suffix)
            time.sleep(SLEEP_SEC)
            if result:
                return result, "drop-first-word"

    # 7. Try just as a locality (no suffix)
    #    "H B Town 1 No Road" → try "H B Town" area query
    if len(words) >= 2:
        short = " ".join(words[:3])
        result = _nominatim_query(short + city_suffix)
        time.sleep(SLEEP_SEC)
        if result:
            return result, "short-locality"

    return None, "no-result"


def main():
    parser = argparse.ArgumentParser(description="Backfill lat/lon for traffic_events")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print geocoding results without writing to DB")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max distinct road names to process (0 = all)")
    args = parser.parse_args()

    engine = create_engine(DB_URL, echo=False)
    conn   = engine.connect()

    result = conn.execute(text(
        "SELECT road_name, COUNT(*) as cnt "
        "FROM traffic_events "
        "WHERE lat IS NULL AND road_name IS NOT NULL AND road_name != '' "
        "GROUP BY road_name ORDER BY cnt DESC"
    ))
    rows = result.fetchall()

    if args.limit > 0:
        rows = rows[:args.limit]

    total_roads  = len(rows)
    total_events = sum(r[1] for r in rows)
    print(f"\n[Backfill] {total_roads} distinct road names → {total_events} DB rows to geocode")
    if args.dry_run:
        print("[Backfill] DRY RUN — no DB writes\n")

    resolved      = 0
    skipped       = 0
    events_filled = 0
    strategy_ct: dict[str, int] = {}

    for i, (road_name, event_count) in enumerate(rows, 1):
        name_preview = repr(road_name)[:60]
        print(f"  [{i:3d}/{total_roads}] {name_preview:62s} ({event_count:4d} rows) ... ",
              end="", flush=True)

        coords, strategy = geocode(road_name)
        strategy_ct[strategy] = strategy_ct.get(strategy, 0) + 1

        if coords is None:
            print(f"SKIP  [{strategy}]")
            skipped += 1
            continue

        lat, lon = coords
        print(f"({lat:.5f}, {lon:.5f})  [{strategy}]", end="")

        if not args.dry_run:
            conn.execute(text(
                "UPDATE traffic_events "
                "SET lat = :lat, lon = :lon "
                "WHERE road_name = :road AND lat IS NULL"
            ), {"lat": lat, "lon": lon, "road": road_name})
            conn.commit()
            print(f"  ✓")
            resolved += 1
            events_filled += event_count
        else:
            print()
            resolved += 1
            events_filled += event_count

    print(f"\n[Backfill] Complete.")
    print(f"  Resolved : {resolved}/{total_roads} road names  ({events_filled} DB rows filled)")
    print(f"  Skipped  : {skipped} (no geocode result)")
    print(f"  Strategies used: {strategy_ct}")

    if not args.dry_run:
        r = conn.execute(text("SELECT COUNT(*) FROM traffic_events WHERE lat IS NOT NULL"))
        total_with_coords = r.fetchone()[0]
        r2 = conn.execute(text("SELECT COUNT(*) FROM traffic_events"))
        total_all = r2.fetchone()[0]
        print(f"\n  DB status: {total_with_coords}/{total_all} events now have coordinates "
              f"({100*total_with_coords/total_all:.1f}%)")

    conn.close()


if __name__ == "__main__":
    main()
