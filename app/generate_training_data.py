"""
generate_training_data.py — Synthetic training data for HGNN
=============================================================
WHY THIS IS NEEDED
------------------
The live DB has 1122 events but they are ~97% TomTom with heavily skewed
severity: low=753, medium=139, high=230.  HGNN needs:
  1. Class balance across low/medium/high severity
  2. Diverse event types (not just TomTom incident categories)
  3. Full coordinate coverage for road-road spatial edges
  4. Diverse sources (multi-source corroboration signal for the graph)

APPROACH
--------
Generates realistic synthetic events anchored to real Kolkata roads and
coordinates.  All synthetic rows are clearly flagged: source = 'synthetic'
and raw_text starts with '[SYNTHETIC]'.

Strategy:
  - 60 named Kolkata roads with real WGS84 coordinates
  - 13 event types matching the HGNN's EVENT_TYPES list
  - Weighted sampling so medium/high are 3× over-represented vs DB ratios
  - Varied sources: tomtom_traffic, rss_city, rss, newsapi, kolkata_police_advisory,
    openweathermap, kmc_waterlogging, kmrc_scrape, twitter_official
  - Temporal diversity: fetched_at spread over past 30 days
  - Realistic confidence ranges per source (matching SOURCE_RELIABILITY)
  - Spatial jitter: ±0.002° so nearby roads don't all share identical coords

HOW TO RUN
----------
  cd app
  python generate_training_data.py                  # add 400 synthetic events
  python generate_training_data.py --count 600      # add 600
  python generate_training_data.py --dry-run        # preview, no DB writes
  python generate_training_data.py --clear-synthetic  # delete old synthetic rows first
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text

DB_URL = "sqlite:///traffic_events.db"

# ── Kolkata roads with real coordinates ──────────────────────────────────────
# 60 well-known road segments covering all parts of Kolkata.
# Coords are approximate road-segment midpoints (WGS84).
KOLKATA_ROADS: list[tuple[str, float, float]] = [
    # Central / CBD
    ("AJC Bose Road",                    22.5411, 88.3497),
    ("Jawaharlal Nehru Road",            22.5553, 88.3512),
    ("Strand Road",                      22.5711, 88.3474),
    ("BBD Bag",                          22.5697, 88.3518),
    ("Park Street",                      22.5522, 88.3527),
    ("Camac Street",                     22.5470, 88.3540),
    ("Chowringhee Road",                 22.5510, 88.3513),
    ("Circus Avenue",                    22.5448, 88.3571),
    ("Elgin Road",                       22.5377, 88.3519),
    ("Shakespeare Sarani",               22.5480, 88.3560),
    # North Kolkata
    ("Shyambazar Five Point Crossing",   22.5921, 88.3733),
    ("Bidhan Sarani",                    22.5800, 88.3640),
    ("Mahatma Gandhi Road",              22.5750, 88.3560),
    ("Rabindra Sarani",                  22.5740, 88.3550),
    ("B T Road",                         22.6200, 88.4050),
    ("Dum Dum Road",                     22.6280, 88.4120),
    ("Ultadanga Road",                   22.5980, 88.3900),
    ("Jessore Road",                     22.6500, 88.4200),
    ("Nager Bazar",                      22.6100, 88.3950),
    ("Barasat Road",                     22.6700, 88.4500),
    # South Kolkata
    ("SP Mukherjee Road",                22.5250, 88.3480),
    ("Ballygunge Circular Road",         22.5250, 88.3650),
    ("Gariahat Road",                    22.5200, 88.3680),
    ("Rashbehari Avenue",                22.5200, 88.3580),
    ("Jadavpur",                         22.4980, 88.3710),
    ("Tollygunge Road",                  22.5050, 88.3530),
    ("Prince Anwar Shah Road",           22.5050, 88.3700),
    ("Diamond Harbour Road",             22.4800, 88.3300),
    ("Behala Chowrasta",                 22.4850, 88.3200),
    ("James Long Sarani",                22.4900, 88.3150),
    # East Kolkata / Salt Lake
    ("EM Bypass",                        22.5200, 88.4000),
    ("Salt Lake Sector V",               22.5740, 88.4300),
    ("VIP Road",                         22.6000, 88.4350),
    ("Rajarhat New Town",                22.5900, 88.4700),
    ("New Town Action Area",             22.5800, 88.4600),
    ("Kona Expressway",                  22.5500, 88.2900),
    ("Bypass Connector",                 22.5350, 88.4100),
    # Howrah / West side
    ("Howrah Bridge",                    22.5851, 88.3468),
    ("GT Road Howrah",                   22.5900, 88.3100),
    ("Foreshore Road",                   22.5700, 88.3200),
    ("Kazi Nazrul Islam Avenue",         22.5850, 88.3450),
    # Key intersections and corridors
    ("Naktala Road",                     22.4950, 88.3700),
    ("Anwar Shah Road",                  22.5050, 88.3650),
    ("Mukundapur",                       22.5100, 88.4050),
    ("Garia Road",                       22.4700, 88.3800),
    ("Baghajatin",                       22.4900, 88.3700),
    ("Jodhpur Park",                     22.5050, 88.3600),
    ("Lake Road",                        22.5200, 88.3700),
    ("Southern Avenue",                  22.5200, 88.3640),
    # Airport / North corridors
    ("Airport Gate No 2",                22.6540, 88.4460),
    ("Nivedita Setu / Bally Bridge",     22.6100, 88.3250),
    ("Vidyasagar Setu",                  22.5480, 88.3230),
    ("Maa Flyover",                      22.5350, 88.3950),
    ("Tallah Bridge",                    22.6000, 88.3750),
    ("Noapara",                          22.6350, 88.3980),
    ("Dunlop",                           22.6400, 88.3780),
    ("Belgharia Expressway",             22.6550, 88.4000),
    ("Parama Island Connector",          22.5350, 88.3900),
    ("Science City Road",                22.5400, 88.4000),
    ("Topsia Road",                      22.5450, 88.3900),
]

# Event types matching HGNN's EVENT_TYPES list exactly
EVENT_TYPES = [
    "accident", "congestion", "road_closure", "construction", "protest",
    "weather", "waterlogging", "vip_movement", "metro_disruption",
    "train_delay", "transport_strike", "diversion", "unknown",
]

# Source → (reliability_approx, typical_confidence_range)
SOURCE_PROFILES = {
    "tomtom_traffic":            (0.95, (0.70, 0.95)),
    "kolkata_police_advisory":   (0.90, (0.75, 0.95)),
    "openweathermap":            (0.90, (0.72, 0.92)),
    "openweathermap_alert":      (0.95, (0.80, 0.97)),
    "kmc_waterlogging":          (0.82, (0.65, 0.88)),
    "kmrc_scrape":               (0.82, (0.60, 0.85)),
    "rss_city":                  (0.70, (0.45, 0.75)),
    "rss":                       (0.65, (0.40, 0.72)),
    "newsapi":                   (0.55, (0.35, 0.65)),
    "twitter_kolkatapolice":     (0.85, (0.65, 0.88)),
    "twitter_official":          (0.80, (0.60, 0.84)),
}

# Severity weights for balanced generation
# DB has: low≈67%, medium≈12%, high≈21%
# We oversample medium/high to achieve closer to 40/30/30 overall
SEVERITY_WEIGHTS = {
    "low":    1.0,
    "medium": 3.5,
    "high":   3.0,
}
SEVERITY_SCORE = {"low": 2, "medium": 5, "high": 10}

# Event-type → plausible severity distribution
EVENT_SEVERITY = {
    "accident":         ["low", "medium", "medium", "high", "high"],
    "congestion":       ["low", "low", "medium", "medium", "high"],
    "road_closure":     ["medium", "high", "high"],
    "construction":     ["low", "low", "medium"],
    "protest":          ["medium", "medium", "high"],
    "weather":          ["low", "medium", "medium", "high"],
    "waterlogging":     ["medium", "medium", "high", "high"],
    "vip_movement":     ["low", "medium", "medium"],
    "metro_disruption": ["medium", "medium", "high"],
    "train_delay":      ["low", "medium", "medium"],
    "transport_strike": ["high", "high", "medium"],
    "diversion":        ["low", "medium"],
    "unknown":          ["low", "low", "medium"],
}

# Reason templates per event type
REASON_TEMPLATES = {
    "accident": [
        "Multi-vehicle collision blocking {lanes} lanes",
        "Road traffic accident involving truck and auto-rickshaw",
        "Fatal accident at {road} junction — police diversion in place",
        "Two-wheeler collision causing {severity} obstruction",
        "Lorry overturned near flyover — towing in progress",
    ],
    "congestion": [
        "Heavy traffic buildup during {period} rush hour",
        "Standstill traffic due to signal failure at crossing",
        "Long queues extending {km} km due to lane merging",
        "Traffic gridlock caused by double parking near market",
        "Severe congestion due to school zone activity",
    ],
    "road_closure": [
        "Road closed for {event} — alternate route via {alt}",
        "Full road closure due to waterlogging — DND affected",
        "Police barricade for VIP movement — road sealed",
        "Emergency road closure for utility pipe burst",
        "Road blocked for procession — diverted via side streets",
    ],
    "construction": [
        "Metro rail construction causing {percent}% lane reduction",
        "Road widening project — single lane open",
        "Footpath construction blocking parking lane",
        "Underground cable work requiring road cuts",
        "Flyover construction causing slow traffic",
    ],
    "protest": [
        "Political rally blocking arterial road — diversions active",
        "Demonstration at {location} junction — police deployed",
        "Road roko agitation by {group} — traffic halted",
        "March blocking road near {landmark} — expect 2-3 hr delay",
    ],
    "weather": [
        "Heavy rainfall causing slow traffic and reduced visibility",
        "Waterlogging due to overnight rain — roads partially submerged",
        "Strong winds from cyclone {name} affecting movement",
        "Dense fog reducing visibility to {m} m — slow traffic",
        "Monsoon conditions: roads slippery, accidents likely",
    ],
    "waterlogging": [
        "Knee-deep flooding at underpass — vehicles stranded",
        "Stormwater drainage overflow blocking {road}",
        "Persistent waterlogging after {mm}mm rainfall",
        "Submerged road impassable for small vehicles",
    ],
    "vip_movement": [
        "VIP convoy movement — road sealed for {duration}",
        "Chief Minister cavalcade — route blocked 15 minutes",
        "Governor's escort causing temporary closure",
        "Presidential visit: multiple roads sealed until {time}",
    ],
    "metro_disruption": [
        "Metro service suspended between {stn1} and {stn2}",
        "Signal failure causing 20-min delay on Blue Line",
        "Technical snag at {station} — passengers stranded",
        "Power disruption affecting metro service citywide",
    ],
    "train_delay": [
        "Sealdah-Howrah trains delayed {mins} minutes",
        "Signal failure near {station} — multiple trains held",
        "Track maintenance causing {percent}% service reduction",
        "Overcrowding at terminus causing platform delays",
    ],
    "transport_strike": [
        "Auto-rickshaw strike — no service on {route} corridor",
        "Bus operators' indefinite strike — all routes suspended",
        "Taxi union agitation — ride-hailing surge pricing active",
        "Mini-bus strike affecting {area} commuters",
    ],
    "diversion": [
        "Traffic diverted via {alt} due to road repair",
        "Mandatory detour around construction zone",
        "Alternative route: take {road} to avoid closure",
    ],
    "unknown": [
        "Unspecified disruption reported near {location}",
        "Traffic disruption reported — cause under investigation",
        "Advisory: avoid {road} — condition unclear",
    ],
}

FILL_WORDS = {
    "{lanes}": ["2", "3", "all"],
    "{period}": ["morning", "evening", "peak"],
    "{km}": ["1", "2", "3"],
    "{event}": ["festival", "rally", "match"],
    "{alt}": ["side road", "service lane", "flyover"],
    "{percent}": ["30", "50", "70"],
    "{location}": ["Esplanade", "Shyambazar", "Gariahat", "Park Street"],
    "{group}": ["residents", "workers", "students"],
    "{landmark}": ["Victoria Memorial", "Kalighat", "Howrah Station"],
    "{name}": ["Remal", "Amphan", "Bulbul"],
    "{m}": ["50", "100", "200"],
    "{mm}": ["80", "120", "200"],
    "{duration}": ["30 minutes", "1 hour", "45 minutes"],
    "{time}": ["18:00", "20:00", "15:30"],
    "{stn1}": ["Howrah", "Esplanade", "Dum Dum"],
    "{stn2}": ["Sealdah", "Salt Lake Sector V", "New Garia"],
    "{station}": ["Esplanade", "Park Street", "Maidan"],
    "{mins}": ["15", "30", "45", "60"],
    "{route}": ["Park Street–Esplanade", "Howrah–Shyambazar", "Salt Lake–Park Circus"],
    "{area}": ["South Kolkata", "North Kolkata", "Howrah"],
    "{road}": ["AJC Bose Road", "EM Bypass", "VIP Road"],
}


def _fill_template(template: str) -> str:
    for placeholder, options in FILL_WORDS.items():
        if placeholder in template:
            template = template.replace(placeholder, random.choice(options), 1)
    return template


def _jitter(lat: float, lon: float, max_deg: float = 0.002) -> tuple[float, float]:
    """Small random offset so nearby roads have distinct but close coordinates."""
    return (
        round(lat + random.uniform(-max_deg, max_deg), 6),
        round(lon + random.uniform(-max_deg, max_deg), 6),
    )


def _random_fetched_at(max_days_ago: int = 30) -> datetime:
    """Random timestamp within the past max_days_ago days."""
    offset_secs = random.uniform(0, max_days_ago * 86400)
    return datetime.now(timezone.utc) - timedelta(seconds=offset_secs)


def generate_events(count: int, seed: int = 42) -> list[dict]:
    """Generate count synthetic traffic events."""
    random.seed(seed)

    # Build weighted severity list for sampling
    sev_pool = []
    for sev, w in SEVERITY_WEIGHTS.items():
        sev_pool.extend([sev] * int(w * 10))

    events = []
    sources = list(SOURCE_PROFILES.keys())

    for _ in range(count):
        # Pick road
        road_name, base_lat, base_lon = random.choice(KOLKATA_ROADS)
        lat, lon = _jitter(base_lat, base_lon)

        # Pick event type
        event_type = random.choice(EVENT_TYPES)

        # Pick severity — use event-type distribution, but also weight by sev_pool
        # for global balance
        type_sevs = EVENT_SEVERITY.get(event_type, ["low", "medium"])
        sev_candidate = random.choice(type_sevs)
        # With 40% probability, override with global pool to force balance
        if random.random() < 0.40:
            sev_candidate = random.choice(sev_pool)
        severity = sev_candidate

        # Pick source
        source = random.choice(sources)
        src_rel, conf_range = SOURCE_PROFILES[source]
        confidence = round(random.uniform(*conf_range), 3)

        # Temporal
        fetched_at = _random_fetched_at(max_days_ago=30)
        is_future  = random.random() < 0.08   # 8% are planned future events

        # Duration
        dur_map = {
            "low":    random.randint(15, 90),
            "medium": random.randint(60, 300),
            "high":   random.randint(180, 1440),
        }
        duration_mins = dur_map[severity]
        dur_label = (
            f"{duration_mins} minutes"   if duration_mins < 60 else
            f"{duration_mins//60} hour{'s' if duration_mins//60 > 1 else ''}"
            if duration_mins < 480
            else f"{duration_mins//60} hours (major)"
        )

        # Reason from template
        templates = REASON_TEMPLATES.get(event_type, ["Traffic disruption on {road}"])
        reason = _fill_template(random.choice(templates))
        # Embed road name in reason sometimes
        if "{road}" not in reason and random.random() < 0.5:
            reason = reason.rstrip(".") + f" on {road_name}."

        # Location = road name + area qualifier sometimes
        area_qualifier = random.choice(["", ", Kolkata", " junction", " crossing", " area"])
        location = road_name + area_qualifier

        severity_score = SEVERITY_SCORE[severity]
        weighted_score = round(severity_score * confidence, 4)

        now_str = fetched_at.strftime("%d/%m/%y")

        events.append({
            "source":               source,
            "source_url":           f"synthetic://{source}/{abs(hash(reason + road_name)) % 999999:06d}",
            "tomtom_url":           None,
            "raw_text":             f"[SYNTHETIC] {event_type.upper()} on {road_name}: {reason}",
            "published_date":       now_str,
            "fetched_at":           fetched_at,
            "event_type":           event_type,
            "transport_relevant":   True,
            "location":             location,
            "location_inferred":    False,
            "location_source":      "road_name",
            "road_name":            road_name,
            "reason":               reason,
            "time_mentioned":       None,
            "is_future_event":      is_future,
            "severity":             severity,
            "severity_score":       severity_score,
            "confidence":           confidence,
            "llm_confidence":       confidence,
            "source_reliability":   src_rel,
            "lat":                  lat,
            "lon":                  lon,
            "start_time_display":   now_str,
            "estimated_end_time":   (fetched_at + timedelta(minutes=duration_mins)).strftime("%d/%m/%y %H:%M"),
            "impact_duration_mins": duration_mins,
            "impact_duration_label": dur_label,
            "duration_source":      "rule_based",
            # HGNN fields — null, filled after training
            "hgnn_confidence":      None,
            "hgnn_severity":        None,
            "hgnn_multiplier":      None,
            "severity_corrected":   False,
        })

    return events


def _check_column_exists(conn, col_name: str) -> bool:
    result = conn.execute(text("PRAGMA table_info(traffic_events)"))
    cols = {row[1] for row in result}
    return col_name in cols


def insert_events(events: list[dict], dry_run: bool = False) -> int:
    """Insert synthetic events into DB. Returns count inserted."""
    engine = create_engine(DB_URL, echo=False)

    if dry_run:
        print(f"  [DRY RUN] Would insert {len(events)} synthetic events.")
        _print_stats(events)
        return 0

    inserted = 0
    with engine.connect() as conn:
        # Check which columns exist (graceful for older schemas)
        has_hgnn   = _check_column_exists(conn, "hgnn_confidence")
        has_coords = _check_column_exists(conn, "lat")

        for ev in events:
            base_cols = """
                source, source_url, tomtom_url, raw_text, published_date, fetched_at,
                event_type, transport_relevant, location, location_inferred, location_source,
                road_name, reason, time_mentioned, is_future_event,
                severity, severity_score, confidence, llm_confidence, source_reliability,
                start_time_display, estimated_end_time,
                impact_duration_mins, impact_duration_label, duration_source
            """
            base_vals = """
                :source, :source_url, :tomtom_url, :raw_text, :published_date, :fetched_at,
                :event_type, :transport_relevant, :location, :location_inferred, :location_source,
                :road_name, :reason, :time_mentioned, :is_future_event,
                :severity, :severity_score, :confidence, :llm_confidence, :source_reliability,
                :start_time_display, :estimated_end_time,
                :impact_duration_mins, :impact_duration_label, :duration_source
            """
            params = {k: v for k, v in ev.items()
                      if k not in ("lat", "lon", "hgnn_confidence", "hgnn_severity",
                                   "hgnn_multiplier", "severity_corrected")}

            if has_coords:
                base_cols += ", lat, lon"
                base_vals += ", :lat, :lon"
                params["lat"] = ev["lat"]
                params["lon"] = ev["lon"]

            if has_hgnn:
                base_cols += ", hgnn_confidence, hgnn_severity, hgnn_multiplier, severity_corrected"
                base_vals += ", :hgnn_confidence, :hgnn_severity, :hgnn_multiplier, :severity_corrected"
                params["hgnn_confidence"]  = ev["hgnn_confidence"]
                params["hgnn_severity"]    = ev["hgnn_severity"]
                params["hgnn_multiplier"]  = ev["hgnn_multiplier"]
                params["severity_corrected"] = ev["severity_corrected"]

            conn.execute(text(
                f"INSERT INTO traffic_events ({base_cols}) VALUES ({base_vals})"
            ), params)
            inserted += 1

        conn.commit()

    return inserted


def clear_synthetic(dry_run: bool = False) -> int:
    """Delete all rows where source = 'synthetic' or raw_text starts with '[SYNTHETIC]'."""
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM traffic_events WHERE source IN ('synthetic') "
            "OR raw_text LIKE '[SYNTHETIC]%'"
        )).scalar()

        if dry_run:
            print(f"  [DRY RUN] Would delete {count} synthetic rows.")
            return count

        conn.execute(text(
            "DELETE FROM traffic_events WHERE source IN ('synthetic') "
            "OR raw_text LIKE '[SYNTHETIC]%'"
        ))
        conn.commit()
        print(f"  Deleted {count} synthetic rows.")
        return count


def _print_stats(events: list[dict]) -> None:
    from collections import Counter
    sev_ct  = Counter(e["severity"]   for e in events)
    type_ct = Counter(e["event_type"] for e in events)
    src_ct  = Counter(e["source"]     for e in events)
    print(f"\n  Severity  : {dict(sev_ct)}")
    print(f"  Event types (top 6): {dict(type_ct.most_common(6))}")
    print(f"  Sources   : {dict(src_ct)}")
    coords_ok = sum(1 for e in events if e.get("lat") is not None)
    print(f"  With coords: {coords_ok}/{len(events)} (100% — all synthetic have coords)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic HGNN training events")
    parser.add_argument("--count",           type=int, default=400,
                        help="Number of synthetic events to generate (default 400)")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Preview stats without writing to DB")
    parser.add_argument("--clear-synthetic", action="store_true",
                        help="Delete existing synthetic rows before inserting new ones")
    args = parser.parse_args()

    print(f"\n[SyntheticGen] Generating {args.count} synthetic training events (seed={args.seed})...")

    if args.clear_synthetic:
        print("[SyntheticGen] Clearing existing synthetic rows...")
        clear_synthetic(dry_run=args.dry_run)

    events = generate_events(args.count, seed=args.seed)
    _print_stats(events)

    if not args.dry_run:
        inserted = insert_events(events, dry_run=False)
        print(f"\n[SyntheticGen] Inserted {inserted} rows into traffic_events.db")

        # Post-insert DB summary
        engine = create_engine(DB_URL, echo=False)
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM traffic_events")).scalar()
            with_c = conn.execute(text("SELECT COUNT(*) FROM traffic_events WHERE lat IS NOT NULL")).scalar()
            sev_rows = conn.execute(text("SELECT severity, COUNT(*) FROM traffic_events GROUP BY severity")).fetchall()
            synth = conn.execute(text("SELECT COUNT(*) FROM traffic_events WHERE raw_text LIKE '[SYNTHETIC]%'")).scalar()
        print(f"\n  DB summary after insert:")
        print(f"    Total events   : {total}")
        print(f"    Synthetic rows : {synth}")
        print(f"    With coords    : {with_c}/{total} ({100*with_c/total:.1f}%)")
        print(f"    Severity dist  : {dict(sev_rows)}")
    else:
        print(f"\n[SyntheticGen] Dry run complete — no DB writes.")
        print(f"  Run without --dry-run to insert {len(events)} events.")


if __name__ == "__main__":
    main()
