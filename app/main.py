"""
Layer 1 Pipeline — Traffic Disruption Intelligence (Kolkata)
=============================================================
Data sources (in priority order):
  1. TomTom Traffic API     — real-time incidents (accidents, closures, road works)
  2. OpenWeatherMap         — real-time weather + government flood/cyclone alerts
  3. Web scrapers           — Kolkata Police advisories, KMRC metro, WB Disaster Mgmt, Railways
  4. Google News RSS        — free, no key, road-specific + Kolkata city feeds
  5. NewsAPI                — supplementary news source (optional)

LLM stack (LangChain LCEL chain):
  prompt (ChatPromptTemplate — Kolkata-tuned)
    | model (ChatOpenAI → OpenRouter, or ChatGoogleGenerativeAI → Gemini)
    | structured_output (TrafficEventSchema)
    + exponential-backoff retry on 429 rate limits
    + fallback to PydanticOutputParser on empty structured-output responses

Flow:
  1. Ask user for source and destination (Kolkata localities)
  2. Compute driving route on Kolkata OSM graph (cached)
  3. Extract named road segments from route
  4. Fetch city-wide real-time data: TomTom + Weather + Scrapers
  5. For each road: fetch road-specific TomTom incidents + RSS + NewsAPI
  6. Deduplicate; tag each article with age and recency flag
  7. LangChain chain → validated TrafficEventSchema → score → persist
  8. Print summary — recent events separated from historical context
"""

import sys

from routing.route_engine import get_k_routes, extract_road_names
from ingestion.news_fetcher import fetch_news
from ingestion.rss_fetcher import fetch_rss_for_query, fetch_kolkata_city_feeds
from ingestion.tomtom_fetcher import fetch_tomtom_incidents, fetch_tomtom_for_road
from ingestion.weather_fetcher import fetch_weather_conditions
from ingestion.web_scraper import fetch_all_scraped_sources
from llm.extractor import extract_event
from llm.schema import TrafficEventSchema
from llm.filter import (
    filter_articles, is_valid_extracted_event, enforce_location_or_drop,
    format_date_ddmmyy, format_today_ddmmyy,
    semantic_deduplicate_events, filter_by_route_relevance,
)
from llm.location_resolver import resolve_location
from scoring.congestion_score import (
    compute_score, compute_weighted_score, compute_route_impact,
    compare_routes,
)
from scoring.confidence import compute_enhanced_confidence, get_source_reliability, compute_multi_source_confirmation
from scoring.impact_duration import estimate_impact
from database.models import TrafficEvent, init_db, get_session
from utils.helpers import build_article_text, now_iso, deduplicate
from config import (
    NEWS_API_KEY, TOMTOM_API_KEY, OPENWEATHER_API_KEY,
    ENABLE_TOMTOM, ENABLE_WEATHER,
    ENABLE_SCRAPER, ENABLE_NEWSAPI, ENABLE_RSS,
)

# ── Pipeline constants ────────────────────────────────────────────────────────

MAX_ROADS               = 6   # top N road segments to query
MAX_ARTICLES_PER_SOURCE = 3   # per road per source
MAX_CITY_FEED_ARTICLES  = 4   # from Kolkata city-wide feeds
RECENT_DAYS             = 7   # articles older than this are flagged as historical


# ── Kolkata locality menu ─────────────────────────────────────────────────────

# 15 well-known Kolkata localities spread across the map bounding box
# (22.40–22.70 N, 88.20–88.50 E) — all geocode reliably with OSMnx
KOLKATA_LOCALITIES = [
    ("Howrah Station",        "Major railway terminus, west bank of Hooghly"),
    ("Sealdah",               "Major railway terminus, central Kolkata"),
    ("Esplanade",             "City centre, Maidan area"),
    ("Park Street",           "Commercial & dining hub, central Kolkata"),
    ("Salt Lake Sector V",    "IT hub, eastern Kolkata"),
    ("New Town Action Area 1","New township, Rajarhat"),
    ("Jadavpur",              "South Kolkata, university area"),
    ("Dum Dum",               "North Kolkata, near airport"),
    ("Kalighat",              "South Kolkata, temple area"),
    ("Gariahat",              "South Kolkata, shopping hub"),
    ("Ultadanga",             "North-central Kolkata, connector hub"),
    ("Tollygunge",            "South Kolkata, metro terminus"),
    ("Shyambazar",            "North Kolkata, five-point crossing"),
    ("Behala",                "South-west Kolkata"),
    ("Barasat",               "North suburban, NH-12 corridor"),
]


def _print_locality_menu() -> None:
    """Print the numbered locality selection menu."""
    print("\n" + "─" * 62)
    print("  Kolkata Traffic AI — Select your journey localities")
    print("─" * 62)
    print("  No.  Locality                    Area")
    print("  " + "─" * 58)
    for i, (name, area) in enumerate(KOLKATA_LOCALITIES, 1):
        print(f"  {i:>2}.  {name:<28}  {area}")
    print("  " + "─" * 58)
    print("  (Or type any locality name / landmark directly)")
    print("─" * 62)


def _is_valid_locality_input(raw: str) -> tuple[bool, str]:
    """
    Validate that the user's input looks like a real locality name or number.

    Returns (is_valid, reason) where reason explains why it was rejected.

    Rejects:
      - Inputs longer than 80 characters (error messages are much longer)
      - Inputs containing error-message patterns (Request ID, HTTP codes, URLs)
      - Inputs with too many special characters
    """
    # Too long — real locality names are short
    if len(raw) > 80:
        return False, f"Input too long ({len(raw)} chars). Please enter a locality name or number 1–{len(KOLKATA_LOCALITIES)}."

    # Contains error message patterns
    import re as _re
    error_patterns = [
        r"request id:",           # OpenRouter/API error IDs
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",  # UUID
        r"too many requests",
        r"please wait before",
        r"http[s]?://",           # URLs
        r"error:",
        r"traceback",
        r"exception",
    ]
    raw_lower = raw.lower()
    for pattern in error_patterns:
        if _re.search(pattern, raw_lower):
            return False, "That looks like an error message, not a locality name. Please type a number (1–15) or a place name."

    # Too many non-alphanumeric characters (error messages have lots of punctuation)
    non_alnum = sum(1 for c in raw if not c.isalnum() and c not in " ,-.()/")
    if non_alnum > 5:
        return False, "Input contains too many special characters. Please enter a locality name or number."

    return True, ""


def _pick_locality(prompt: str) -> str:
    """
    Ask the user to pick a locality by number (1–15) or type a custom name.
    Validates input to reject accidental pastes of error messages.

    Returns the locality name string (ready for geocoding).
    """
    while True:
        try:
            raw = input(f"\n  {prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted. Exiting.")
            sys.exit(0)

        if not raw:
            print("  ⚠  Cannot be empty. Please enter a number (1–15) or a locality name.")
            continue

        # Validate input before processing
        is_valid, reason = _is_valid_locality_input(raw)
        if not is_valid:
            print(f"  ⚠  {reason}")
            continue

        # Number input → resolve to locality name
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(KOLKATA_LOCALITIES):
                name = KOLKATA_LOCALITIES[idx - 1][0]
                print(f"  ✓  Selected: {name}")
                return name
            else:
                print(f"  ⚠  Please enter a number between 1 and {len(KOLKATA_LOCALITIES)}, "
                      f"or type a locality name.")
                continue

        # Free-text input — use as-is
        return raw


def _prompt_route() -> tuple[str, str]:
    """
    Show the 15-locality menu and ask the user to pick source and destination
    by number or by typing a custom locality name.
    """
    _print_locality_menu()

    source      = _pick_locality("Source      (number or name)")
    destination = _pick_locality("Destination (number or name)")

    if source.lower() == destination.lower():
        print("  ⚠  Source and destination are the same. Please choose different localities.")
        return _prompt_route()

    return source, destination


# ── Source summary helper ─────────────────────────────────────────────────────

def _active_sources() -> list[str]:
    """Return a list of active source labels for the header."""
    sources = []
    if ENABLE_TOMTOM and TOMTOM_API_KEY:
        sources.append("TomTom Traffic")
    if ENABLE_WEATHER and OPENWEATHER_API_KEY:
        sources.append("OpenWeatherMap")
    if ENABLE_SCRAPER:
        sources.append("Web Scrapers")
    if ENABLE_RSS:
        sources.append("Google News RSS")
    if ENABLE_NEWSAPI and NEWS_API_KEY:
        sources.append("NewsAPI")
    return sources


# ── City-wide real-time fetchers ──────────────────────────────────────────────

def _fetch_tomtom_city() -> list[dict]:
    """Fetch TomTom real-time incidents for the full Kolkata area."""
    if not ENABLE_TOMTOM or not TOMTOM_API_KEY:
        return []
    items = fetch_tomtom_incidents(max_items=15)
    return [
        {
            "title":          i["title"],
            "description":    i["description"],
            "url":            i["url"],
            "tomtom_url":     i.get("tomtom_url", ""),
            "source":         "tomtom_traffic",
            "age_label":      i["age_label"],
            "is_recent":      i["is_recent"],
            # Pass coordinates and road metadata for location resolution
            "lat":            i.get("lat"),
            "lon":            i.get("lon"),
            "_tomtom_road":   i.get("_tomtom_road", ""),
            "_api_end_time":  None,   # TomTom endTime is embedded in description
            "_road":          i.get("_tomtom_road") or "Kolkata (TomTom incident)",
        }
        for i in items
    ]


def _fetch_weather_city() -> list[dict]:
    """Fetch current weather conditions and alerts for Kolkata."""
    if not ENABLE_WEATHER or not OPENWEATHER_API_KEY:
        return []
    items = fetch_weather_conditions(max_items=3)
    return [
        {
            "title":       i["title"],
            "description": i["description"],
            "url":         i["url"],
            "source":      i["source"],
            "age_label":   i["age_label"],
            "is_recent":   i["is_recent"],
            "_road":       "Kolkata (weather)",
        }
        for i in items
    ]


def _fetch_scraped_city() -> list[dict]:
    """Fetch scraped advisories from official Kolkata government sites."""
    if not ENABLE_SCRAPER:
        return []
    items = fetch_all_scraped_sources(max_items_each=4)
    return [
        {
            "title":       i["title"],
            "description": i["description"],
            "url":         i["url"],
            "source":      i["source"],
            "age_label":   i["age_label"],
            "is_recent":   i["is_recent"],
            "_road":       "Kolkata (official advisory)",
        }
        for i in items
    ]


# ── Per-road fetchers ─────────────────────────────────────────────────────────

def _articles_from_rss(road: str) -> list[dict]:
    """Fetch Google News RSS articles for a specific road in Kolkata."""
    if not ENABLE_RSS:
        return []
    items = fetch_rss_for_query(
        f"{road} traffic",
        max_items=MAX_ARTICLES_PER_SOURCE,
    )
    return [
        {
            "title":       i.get("title", ""),
            "description": i.get("description", ""),
            "url":         i.get("link", ""),
            "source":      "rss",
            "age_label":   i.get("age_label", "unknown date"),
            "is_recent":   i.get("is_recent", True),
        }
        for i in items
    ]


def _articles_from_newsapi(road: str) -> list[dict]:
    """Fetch NewsAPI articles for a road, scoped to Kolkata."""
    if not ENABLE_NEWSAPI or not NEWS_API_KEY:
        return []
    data = fetch_news(f"{road} Kolkata")
    articles = []
    for a in data.get("articles", [])[:MAX_ARTICLES_PER_SOURCE]:
        published_at = a.get("publishedAt", "")   # ISO-8601 from NewsAPI
        age_label    = format_date_ddmmyy(published_at) or "unknown date"
        articles.append({
            "title":       a.get("title", ""),
            "description": a.get("description", ""),
            "url":         a.get("url", ""),
            "source":      "newsapi",
            "age_label":   published_at or "unknown date",  # keep ISO for age filter
            "is_recent":   True,
        })
    return articles


def _articles_from_tomtom_road(road: str) -> list[dict]:
    """Fetch TomTom incidents filtered to a specific road."""
    if not ENABLE_TOMTOM or not TOMTOM_API_KEY:
        return []
    items = fetch_tomtom_for_road(road, max_items=MAX_ARTICLES_PER_SOURCE)
    return [
        {
            "title":         i["title"],
            "description":   i["description"],
            "url":           i["url"],
            "tomtom_url":    i.get("tomtom_url", ""),
            "source":        "tomtom_traffic",
            "age_label":     i["age_label"],
            "is_recent":     i["is_recent"],
            "lat":           i.get("lat"),
            "lon":           i.get("lon"),
            "_tomtom_road":  i.get("_tomtom_road", ""),
            "_api_end_time": None,
        }
        for i in items
    ]


def _articles_from_city_feeds() -> list[dict]:
    """Pull from Kolkata city-wide RSS feeds."""
    if not ENABLE_RSS:
        return []
    items = fetch_kolkata_city_feeds(max_items=MAX_CITY_FEED_ARTICLES)
    return [
        {
            "title":       i.get("title", ""),
            "description": i.get("description", ""),
            "url":         i.get("link", ""),
            "source":      "rss_city",
            "age_label":   i.get("age_label", "unknown date"),
            "is_recent":   i.get("is_recent", True),
            "_road":       "Kolkata (city-wide)",
        }
        for i in items
    ]


# ── Per-article processing ────────────────────────────────────────────────────

def _process_article(article: dict, road: str, session) -> dict | None:
    """
    Full enhanced pipeline for one article:
      1. Build text
      2. LLM extraction
      3. Transport relevance check
      4. Location resolution (direct → coordinates → text inference → road context)
      5. Multi-factor confidence scoring
      6. Impact duration estimation
      7. Date formatting
      8. Persist to DB
      9. Return enriched result dict
    """
    text = build_article_text(article["title"], article["description"])
    if not text.strip():
        return None

    # ── LLM extraction ────────────────────────────────────────────────────────
    source = article.get("source", "")
    if source == "tomtom_traffic":
        print(f"  [Direct] Extracting from structured API data (no LLM call)")
    event: TrafficEventSchema | None = extract_event(text, article=article)
    if event is None:
        return None

    # ── Post-extraction relevance filter ─────────────────────────────────────
    if not is_valid_extracted_event(event):
        return None

    # ── Location resolution ───────────────────────────────────────────────────
    resolved_location, loc_source, loc_inferred = resolve_location(
        event, article, road_context=road
    )

    # ── Location enforcement ──────────────────────────────────────────────────
    # Drop events with no location unless confidence is high enough
    keep, loc_reason = enforce_location_or_drop(
        resolved_location = resolved_location,
        confidence        = event.confidence,
        event_type        = event.event_type.value,
        source            = article.get("source", "unknown"),
    )
    if not keep:
        print(f"  [Filter] Dropped — {loc_reason}")
        return None

    # ── Road name: prefer event's road_name, fall back to resolved location ───
    road_name = event.road_name or road

    # ── Severity scoring ──────────────────────────────────────────────────────
    sev_score = compute_score(event.severity.value)

    # ── Multi-factor confidence ───────────────────────────────────────────────
    enhanced_conf = compute_enhanced_confidence(
        llm_confidence   = event.confidence,
        source           = article.get("source", "unknown"),
        severity         = event.severity.value,
        location         = resolved_location,
        location_inferred = loc_inferred,
        age_label        = article.get("age_label", "unknown date"),
        is_recent        = article.get("is_recent", True),
    )
    weighted = compute_weighted_score(event.severity.value, enhanced_conf)

    # ── Impact duration estimation ────────────────────────────────────────────
    # Extract official API end time if present in article metadata
    api_end_time = article.get("_api_end_time")   # set by TomTom fetcher if available

    duration = estimate_impact(
        event_type       = event.event_type.value,
        severity         = event.severity.value,
        llm_estimated_end = event.estimated_end_time,
        llm_duration_mins = event.impact_duration_mins,
        api_end_time     = api_end_time,
        age_label        = article.get("age_label", "unknown date"),
    )

    # ── Date formatting ───────────────────────────────────────────────────────
    published_date = (
        format_date_ddmmyy(article.get("age_label", ""))
        or format_today_ddmmyy()
    )

    # ── Persist to database ───────────────────────────────────────────────────
    record = TrafficEvent(
        source              = article["source"],
        source_url          = article.get("url"),
        tomtom_url          = article.get("tomtom_url"),
        raw_text            = text,
        published_date      = published_date,
        event_type          = event.event_type.value,
        transport_relevant  = getattr(event, "transport_relevant", True),
        location            = resolved_location,
        location_inferred   = loc_inferred,
        location_source     = loc_source,
        road_name           = road_name,
        reason              = event.reason,
        time_mentioned      = event.time_mentioned,
        is_future_event     = event.is_future_event,
        severity            = event.severity.value,
        severity_score      = sev_score,
        confidence          = enhanced_conf,
        llm_confidence      = event.confidence,
        source_reliability  = get_source_reliability(article.get("source", "unknown")),
        start_time_display  = duration["start_time"],
        estimated_end_time  = duration["estimated_end_time"],
        impact_duration_mins  = duration["impact_duration_mins"],
        impact_duration_label = duration["impact_duration_label"],
        duration_source     = duration["duration_source"],
    )
    session.add(record)
    session.commit()

    return {
        "event_type":           event.event_type.value,
        "transport_relevant":   getattr(event, "transport_relevant", True),
        "location":             resolved_location,
        "location_inferred":    loc_inferred,
        "location_source":      loc_source,
        "road_name":            road_name,
        "severity":             event.severity.value,
        "confidence":           enhanced_conf,
        "llm_confidence":       event.confidence,
        "reason":               event.reason,
        "time_mentioned":       event.time_mentioned,
        "is_future_event":      event.is_future_event,
        "severity_score":       sev_score,
        "weighted_score":       weighted,
        "age_label":            article.get("age_label", "unknown date"),
        "is_recent":            article.get("is_recent", True),
        "published_date":       published_date,
        "source":               article.get("source", "unknown"),
        # Impact duration
        "start_time":           duration["start_time"],
        "estimated_end_time":   duration["estimated_end_time"],
        "impact_duration_mins": duration["impact_duration_mins"],
        "impact_duration_label": duration["impact_duration_label"],
        "duration_source":      duration["duration_source"],
        # URLs
        "source_url":           article.get("url", ""),
        "tomtom_url":           article.get("tomtom_url", ""),
        # Coordinates (from TomTom API — used for spatial route filtering)
        "lat":                  article.get("lat"),
        "lon":                  article.get("lon"),
    }


# ── Summary helpers ───────────────────────────────────────────────────────────

def _print_event_block(r: dict, indent: str = "    ") -> None:
    age_tag  = f"  [{r['published_date']}]" if r.get("published_date") else ""
    stale    = "  ⏳ HISTORICAL" if not r["is_recent"] else ""
    loc_tag  = "  (inferred)" if r.get("location_inferred") else ""
    print(f"{indent}• [{r['road_name']}] {r['event_type'].upper()}{age_tag}{stale}")
    print(f"{indent}  Reason   : {r['reason']}")
    if r.get("location"):
        print(f"{indent}  Location : {r['location']}{loc_tag}")
    if r.get("time_mentioned"):
        print(f"{indent}  Time     : {r['time_mentioned']}")
    print(f"{indent}  Duration : {r.get('impact_duration_label', 'unknown')} "
          f"(est. end: {r.get('estimated_end_time', 'unknown')})")
    print(f"{indent}  Confidence: κ={r['confidence']:.2f} "
          f"[LLM={r.get('llm_confidence', 0):.2f}, "
          f"src={r.get('location_source', '?')}]")
    if r.get("source_url") and not r["source_url"].startswith(("tomtom://", "here://", "owm://")):
        print(f"{indent}  Map      : {r['source_url']}")
    if r.get("tomtom_url"):
        print(f"{indent}  TomTom   : {r['tomtom_url']}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    # ── Step 0: Get route from user ───────────────────────────────────────────
    source, destination = _prompt_route()

    active_sources = _active_sources()
    print("\n" + "=" * 60)
    print("  Kolkata Traffic AI — Layer 1 Disruption Intelligence")
    print(f"  Route  : {source}  →  {destination}")
    print(f"  Time   : {now_iso()}")
    print(f"  Sources: {' | '.join(active_sources) if active_sources else 'RSS only'}")
    print("=" * 60)

    init_db()
    session = get_session()

    # ── Step 1: Compute alternative routes ───────────────────────────────────
    print("\n[1/5] Computing alternative routes on Kolkata road network...")
    try:
        graph, routes = get_k_routes(source, destination)
    except ValueError as e:
        print(f"\n  ✗ Route error: {e}")
        session.close()
        sys.exit(1)

    print(f"      {len(routes)} route option(s) found")
    for route in routes:
        print(f"        • {route['route_label']}: {route['distance_km']} km, "
              f"~{route['travel_time_min']} min, {len(route['roads'])} named roads")

    # ── Step 2: Fetch city-wide sources once ──────────────────────────────────
    print("\n[2/5] Fetching city-wide real-time data ...")
    citywide_articles: list[dict] = []

    print("\n  ── City-wide real-time sources ──────────────────────────")

    tomtom_city = _fetch_tomtom_city()
    print(f"    TomTom Traffic   : {len(tomtom_city)} incidents")
    citywide_articles.extend(tomtom_city)

    weather_articles = _fetch_weather_city()
    print(f"    OpenWeatherMap   : {len(weather_articles)} weather alerts")
    citywide_articles.extend(weather_articles)

    scraped_articles = _fetch_scraped_city()
    print(f"    Web Scrapers     : {len(scraped_articles)} advisories")
    citywide_articles.extend(scraped_articles)

    city_rss = _articles_from_city_feeds()
    city_rss = deduplicate(city_rss, key="url")
    print(f"    Google News RSS  : {len(city_rss)} articles")
    citywide_articles.extend(city_rss)
    citywide_articles = deduplicate(citywide_articles, key="url")

    # ── Step 3: Evaluate each route option separately ────────────────────────
    print("\n[3/5] Evaluating each route option ...")
    route_cache: dict[str, list[dict]] = {}
    route_profiles: list[dict] = []

    for route in routes:
        print(f"\n  ── Route option: {route['route_label']} ───────────────────────")
        print(f"      {route['distance_km']} km, ~{route['travel_time_min']} min")

        route_articles: list[dict] = []
        route_roads = route.get('roads', [])[:MAX_ROADS]

        for road in route_roads:
            if road not in route_cache:
                road_articles: list[dict] = []
                road_articles.extend(_articles_from_tomtom_road(road))
                road_articles.extend(_articles_from_rss(road))
                road_articles.extend(_articles_from_newsapi(road))
                road_articles = deduplicate(road_articles, key="url")
                for a in road_articles:
                    a["_road"] = road
                route_cache[road] = road_articles

            route_articles.extend(route_cache[road])

        route_articles = deduplicate(route_articles, key="url")
        route_articles.extend(citywide_articles)
        route_articles = deduplicate(route_articles, key="url")

        route_articles, filter_stats = filter_articles(route_articles, max_age_days=2)
        recent_count = sum(1 for a in route_articles if a.get("is_recent", True))

        print(f"    Articles collected : {len(route_articles)}")
        print(f"    Kept after filters: {filter_stats['kept']} "
              f"(age: {filter_stats['dropped_age']}, "
              f"geo: {filter_stats['dropped_geography']}, "
              f"irrelevant: {filter_stats['dropped_relevance']})")
        print(f"    Recent articles    : {recent_count}")

        raw_results: list[dict] = []
        if route_articles:
            print("\n  [3/5] Running LangChain extraction chain (Kolkata-tuned)...")
        for i, raw_article in enumerate(route_articles, 1):
            article = dict(raw_article)
            road = article.get("_road", "unknown")
            src_label = article["source"].upper()
            age_tag = f"  [{article.get('age_label', '?')}]"

            print(f"\n    [{i}/{len(route_articles)}] [{src_label}]{age_tag} {road}")
            print(f"    Title: {article['title'][:80]}")

            result = _process_article(article, road, session)
            if result:
                raw_results.append(result)
                future_flag = "  ⚠ FUTURE" if result["is_future_event"] else ""
                print(f"    → {result['event_type']} | {result['severity'].upper()} "
                      f"(σ={result['severity_score']}, "
                      f"κ={result['confidence']:.2f}, "
                      f"weighted={result['weighted_score']})"
                      f"{future_flag}")
            else:
                print("    → No disruption event extracted")

        results = semantic_deduplicate_events(raw_results)
        dedup_removed = len(raw_results) - len(results)
        if dedup_removed > 0:
            print(f"\n    [Dedup] Removed {dedup_removed} semantically duplicate events "
                  f"({len(raw_results)} → {len(results)})")

        # Rule-based severity correction
        from scoring.severity_rules import correct_severities
        results, n_sev_corrected = correct_severities(results)
        if n_sev_corrected:
            print(f"    [SevRule] Corrected {n_sev_corrected} event severities")

        confirmation_map = compute_multi_source_confirmation(results)
        confirmed_count = sum(1 for v in confirmation_map.values() if v)
        if confirmed_count > 0:
            print(f"    [Confidence] {confirmed_count} events confirmed by multiple sources")
            for idx, r in enumerate(results):
                if confirmation_map.get(str(idx)):
                    old_conf = r["confidence"]
                    r["confidence"] = min(1.0, round(old_conf + 0.05, 4))
                    r["weighted_score"] = compute_weighted_score(r["severity"], r["confidence"])

        route_results, citywide_results = filter_by_route_relevance(
            results, route_roads,
            source=source, destination=destination,
            route_coords=route.get("coords"),
            corridor_km=2.0,
        )
        print(f"    Route-relevant     : {len(route_results)}")
        print(f"    City-wide context  : {len(citywide_results)}")

        impact = compute_route_impact(route_results)
        route_profiles.append({
            "route_id": route["route_id"],
            "route_label": route["route_label"],
            "distance_km": route["distance_km"],
            "travel_time_min": route["travel_time_min"],
            "roads": route_roads,
            "articles": len(route_articles),
            "events_extracted": len(results),
            "route_results": route_results,
            "citywide_results": citywide_results,
            "impact": impact,
        })

    # ── Step 4: Compare route options and choose best route ───────────────────
    sorted_routes = compare_routes(route_profiles)
    best_route = sorted_routes[0]

    print("\n" + "=" * 60)
    print("[4/5] Route Comparison")
    print("=" * 60)
    for r in sorted_routes:
        print(f"  {r['route_label']}: {r['distance_km']} km, ~{r['travel_time_min']} min, "
              f"active={r['impact']['active_score']:.2f}, "
              f"future={r['impact']['future_score']:.2f}, "
              f"risk={r['impact']['risk_level']}, "
              f"route events={len(r['route_results'])}")

    print(f"\n  Recommended best route: {best_route['route_label']} "
          f"(lowest active impact score)")

    print("\n" + "=" * 60)
    print("[5/5] Best Route Disruption Summary")
    print("=" * 60)
    print(f"  Route              : {source}  →  {destination}")
    print(f"  Selected route     : {best_route['route_label']}")
    print(f"  Distance           : {best_route['distance_km']} km")
    print(f"  Travel time        : ~{best_route['travel_time_min']} min")
    print(f"  Articles processed : {best_route['articles']}")
    print(f"  Events extracted   : {best_route['events_extracted']}")
    print(f"  Route-relevant     : {len(best_route['route_results'])}")
    print(f"  City-wide context  : {len(best_route['citywide_results'])}")

    if not best_route['route_results']:
        print("\n  ✓  No disruption events found for the selected route.")
    else:
        active_results = [r for r in best_route['route_results']
                           if r.get('is_recent', True) and not r.get('is_future_event', False)]
        future_results = [r for r in best_route['route_results'] if r.get('is_future_event', False)]
        hist_results = [r for r in best_route['route_results'] if not r.get('is_recent', True)]

        _sev_order = {'high': 0, 'medium': 1, 'low': 2}
        active_results.sort(key=lambda r: (_sev_order.get(r['severity'], 3),
                                           -r.get('confidence', 0)))

        high_r = [r for r in active_results if r['severity'] == 'high']
        medium_r = [r for r in active_results if r['severity'] == 'medium']
        low_r = [r for r in active_results if r['severity'] == 'low']

        impact = best_route['impact']
        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  ROUTE IMPACT SCORE                                 │")
        print(f"  │  Active score  : {impact['active_score']:<6.2f}  "
              f"Future score: {impact['future_score']:<6.2f}          │")
        print(f"  │  Risk level    : {impact['risk_level']:<10}                        │")
        print(f"  │  {impact['risk_label']:<51} │")
        print(f"  └─────────────────────────────────────────────────────┘")
        print(f"\n  {impact['recommendation']}")

        print(f"\n  ── Active disruptions on route ({len(active_results)}) ──────────────")
        print(f"     High: {len(high_r)}  Medium: {len(medium_r)}  Low: {len(low_r)}")

        if high_r:
            print(f"\n  ⚠  HIGH SEVERITY:")
            for r in high_r:
                _print_event_block(r)

        if medium_r:
            print(f"\n  ⚡  MEDIUM SEVERITY:")
            for r in medium_r:
                _print_event_block(r)

        if low_r:
            print(f"\n  ℹ  LOW SEVERITY:")
            for r in low_r[:3]:
                _print_event_block(r)
            if len(low_r) > 3:
                print(f"    ... and {len(low_r) - 3} more low-severity events")

        if future_results:
            print(f"\n  📅  UPCOMING / PLANNED DISRUPTIONS ({len(future_results)}):")
            for r in future_results:
                _print_event_block(r)

        if hist_results:
            old_high = [r for r in hist_results if r['severity'] == 'high']
            print(f"\n  ⏳  HISTORICAL CONTEXT ({len(hist_results)} events, "
                  f"{len(old_high)} high-severity) — may not reflect current conditions:")
            for r in hist_results[:3]:
                _print_event_block(r, indent='      ')
            if len(hist_results) > 3:
                print(f"      ... and {len(hist_results) - 3} more (see traffic_events.db)")

        cw_high = [r for r in best_route['citywide_results']
                   if r.get('is_recent', True) and r['severity'] == 'high']
        if cw_high:
            print(f"\n  🌆  CITY-WIDE HIGH-SEVERITY EVENTS (not on your route):")
            for r in cw_high[:3]:
                _print_event_block(r, indent='      ')

    print(f"\n  Events saved to: traffic_events.db")
    print("=" * 60)

    session.close()
    return best_route['route_results']


if __name__ == "__main__":
    run()