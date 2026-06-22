"""
api.py — FastAPI backend for Kolkata Traffic AI map visualisation
=================================================================
Endpoints:
  GET  /                    → serves the Leaflet.js frontend
  GET  /api/locations       → list of valid Kolkata localities
  POST /api/route           → Step 1: compute routes only (~2s, no LLM)
  POST /api/disruptions     → Step 2: fetch data + LLM extraction (~30-60s)

Two-step design: map draws instantly after /api/route, then
/api/disruptions fills in risk scores and event markers.

Run with:
  cd app_rucha
  uvicorn api:app --reload --port 8000
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from routing.route_engine import get_multiple_routes
from routing.multimodal import get_routes_for_mode, get_routes_for_modes, get_metro_geojson_overlay
from ingestion.rss_fetcher import fetch_rss_for_query, fetch_kolkata_city_feeds
from ingestion.news_fetcher import fetch_news
from ingestion.tomtom_fetcher import fetch_tomtom_incidents
from ingestion.weather_fetcher import fetch_weather_conditions
from ingestion.web_scraper import fetch_all_scraped_sources
from llm.extractor import extract_event
from llm.filter import (
    filter_articles, semantic_deduplicate_events,
    filter_by_route_relevance, format_date_ddmmyy, format_today_ddmmyy,
)
from llm.location_resolver import resolve_location
from llm.filter import is_valid_extracted_event, enforce_location_or_drop
from scoring.congestion_score import compute_score, compute_weighted_score
from scoring.confidence import compute_enhanced_confidence, get_source_reliability
from scoring.impact_duration import estimate_impact
from database.models import TrafficEvent, init_db, get_session
from utils.helpers import build_article_text, deduplicate, now_iso
from config import (
    NEWS_API_KEY, TOMTOM_API_KEY, OPENWEATHER_API_KEY,
    ENABLE_TOMTOM, ENABLE_WEATHER, ENABLE_SCRAPER,
    ENABLE_NEWSAPI, ENABLE_RSS,
)
from datetime import datetime, timezone, timedelta
from weather.route_weather import fetch_weather as _fetch_weather_point
from weather.weather_risk import compute_weather_risk
from hgnn.integration import enhance_event_confidences, score_route_with_hgnn, get_cascade_road_names

app = FastAPI(title="Kolkata Traffic AI", version="2.0")

# ── Forward-geocode cache — used to fill lat/lon for non-TomTom events ────────
# Nominatim calls are expensive (~1s each). We cache by location string so the
# same road name resolved from multiple articles only triggers one HTTP request
# per process lifetime. This also avoids the hot-path latency problem (#3) by
# ensuring each unique location string hits Nominatim at most once per session.
_fwd_geocode_cache: dict[str, tuple[float, float] | None] = {}

# Generic city-level strings that are not worth geocoding — they'd return the
# Kolkata city centroid, which is misleading for spatial adjacency edges.
_GENERIC_LOCATION_STRINGS = {
    "kolkata", "kolkata (city-wide)", "kolkata (tomtom)", "kolkata (here incident)",
    "kolkata (tomtom incident)", "kolkata (weather)", "kolkata (official advisory)",
    "kolkata (twitter)", "west bengal", "unknown",
}


def _forward_geocode(location: str) -> tuple[float, float] | None:
    """
    Forward-geocode a location name → (lat, lon) using Nominatim.

    Results are cached for the lifetime of the process. Generic city-wide
    strings are skipped to avoid returning a misleading city centroid.
    Always appends ', Kolkata, West Bengal' to bias results to the city.
    """
    from config import ENABLE_NOMINATIM
    if not ENABLE_NOMINATIM:
        return None

    loc_lower = location.strip().lower()
    if not loc_lower or loc_lower in _GENERIC_LOCATION_STRINGS:
        return None

    if loc_lower in _fwd_geocode_cache:
        return _fwd_geocode_cache[loc_lower]

    result = None
    try:
        import requests as _req
        query = f"{location.strip()}, Kolkata, West Bengal, India"
        resp = _req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
            headers={"User-Agent": "KolkataTrafficAI/1.0"},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            result = (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception:
        pass   # non-fatal — event still persisted, just without coordinates

    _fwd_geocode_cache[loc_lower] = result
    return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve CSS/JS from app/static/ at /static/*
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_ARTICLES_PER_SOURCE = 3
MAX_CITY_FEED_ARTICLES  = 4

SEVERITY_COLOR = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}
RISK_COLOR     = {"HIGH": "#e74c3c", "MODERATE": "#f39c12", "LOW": "#27ae60",
                  "CLEAR": "#27ae60", "CRITICAL": "#8e44ad"}

KOLKATA_LOCATIONS = [
    {"id": 1,  "name": "Howrah Station",        "desc": "Major railway terminus, west bank of Hooghly"},
    {"id": 2,  "name": "Sealdah",               "desc": "Major railway terminus, central Kolkata"},
    {"id": 3,  "name": "Esplanade",             "desc": "City centre, Maidan area"},
    {"id": 4,  "name": "Park Street",           "desc": "Commercial & dining hub, central Kolkata"},
    {"id": 5,  "name": "Salt Lake Sector V",    "desc": "IT hub, eastern Kolkata"},
    {"id": 6,  "name": "New Town Action Area 1","desc": "New township, Rajarhat"},
    {"id": 7,  "name": "Jadavpur",              "desc": "South Kolkata, university area"},
    {"id": 8,  "name": "Dum Dum",               "desc": "North Kolkata, near airport"},
    {"id": 9,  "name": "Kalighat",              "desc": "South Kolkata, temple area"},
    {"id": 10, "name": "Gariahat",              "desc": "South Kolkata, shopping hub"},
    {"id": 11, "name": "Ultadanga",             "desc": "North-central Kolkata, connector hub"},
    {"id": 12, "name": "Tollygunge",            "desc": "South Kolkata, metro terminus"},
    {"id": 13, "name": "Shyambazar",            "desc": "North Kolkata, five-point crossing"},
    {"id": 14, "name": "Behala",                "desc": "South-west Kolkata"},
    {"id": 15, "name": "Barasat",               "desc": "North suburban, NH-12 corridor"},
]


# ── Pydantic request/response models ──────────────────────────────────────────

class RouteRequest(BaseModel):
    source:      str
    destination: str
    mode:        str = "drive"   # drive | walk | bike | metro | bus
    modes:       list[str] | None = None  # ['metro','walk'] or ['metro','bike']


class RouteSummary(BaseModel):
    id:              int
    label:           str
    distance_km:     float
    travel_time_min: float
    road_names:      list[str]
    geojson:         dict
    coords:          list[list[float]] | None = None
    risk_score:      float | None = None
    risk_level:      str | None = None
    risk_color:      str | None = None
    matched_events:  list[dict] | None = None
    event_count:     int | None = None
    is_best:         bool | None = None
    mode:            str | None = None
    segments:        list[dict] | None = None   # metro legs


class DisruptionRequest(BaseModel):
    source:      str
    destination: str
    road_names:  list[str]
    routes:      list[RouteSummary] | None = None
    mode:        str = "drive"


# ── City-wide article fetchers ─────────────────────────────────────────────────

def _fetch_citywide(skip_tomtom: bool = False) -> list[dict]:
    """Fetch Weather + Scrapers + city RSS — city-wide sources.
    Pass skip_tomtom=True when TomTom has already been fetched separately."""
    articles: list[dict] = []

    if not skip_tomtom and ENABLE_TOMTOM and TOMTOM_API_KEY:
        for i in fetch_tomtom_incidents(max_items=15):
            articles.append({
                "title":        i["title"],
                "description":  i["description"],
                "url":          i.get("url", ""),
                "tomtom_url":   i.get("tomtom_url", ""),
                "source":       "tomtom_traffic",
                "age_label":    i["age_label"],
                "is_recent":    i["is_recent"],
                "lat":          i.get("lat"),
                "lon":          i.get("lon"),
                "_tomtom_road": i.get("_tomtom_road", ""),
                "_tomtom_category": i.get("_tomtom_category", ""),
                "_tomtom_severity": i.get("_tomtom_severity", "low"),
                "_road":        i.get("_tomtom_road") or "Kolkata (TomTom)",
            })

    if ENABLE_WEATHER and OPENWEATHER_API_KEY:
        for i in fetch_weather_conditions(max_items=3):
            articles.append({
                "title":       i["title"],
                "description": i["description"],
                "url":         i.get("url", ""),
                "source":      i["source"],
                "age_label":   i["age_label"],
                "is_recent":   i["is_recent"],
                "_road":       "Kolkata (weather)",
            })

    if ENABLE_SCRAPER:
        for i in fetch_all_scraped_sources(max_items_each=4):
            articles.append({
                "title":       i["title"],
                "description": i["description"],
                "url":         i.get("url", ""),
                "source":      i["source"],
                "age_label":   i["age_label"],
                "is_recent":   i["is_recent"],
                "_road":       "Kolkata (advisory)",
            })

    if ENABLE_RSS:
        for i in fetch_kolkata_city_feeds(max_items=MAX_CITY_FEED_ARTICLES):
            articles.append({
                "title":       i.get("title", ""),
                "description": i.get("description", ""),
                "url":         i.get("link", ""),
                "source":      "rss_city",
                "age_label":   i.get("age_label", "unknown date"),
                "is_recent":   i.get("is_recent", True),
                "_road":       "Kolkata (city-wide)",
            })

    return deduplicate(articles, key="url")


def _fetch_road_articles(roads: list[str], tomtom_cache: list[dict]) -> list[dict]:
    """
    Fetch per-road articles from TomTom (cached) + RSS + NewsAPI.

    tomtom_cache: pre-fetched full TomTom incident list (passed in to avoid
                  re-calling the API once per road).
    """
    articles: list[dict] = []

    # Build a lowercased set for fast TomTom filtering
    for road in roads:
        road_lower = road.lower()

        # ── TomTom: filter from cache, no new API call ────────────────────────
        if ENABLE_TOMTOM and TOMTOM_API_KEY:
            matched = [
                i for i in tomtom_cache
                if road_lower in i["title"].lower()
                or road_lower in i["description"].lower()
                or road_lower in i.get("_tomtom_road", "").lower()
            ]
            # Fall back to first few city-wide incidents if no road match
            road_incidents = matched[:MAX_ARTICLES_PER_SOURCE] or tomtom_cache[:MAX_ARTICLES_PER_SOURCE]
            for i in road_incidents:
                articles.append({
                    "title":        i["title"],
                    "description":  i["description"],
                    "url":          i.get("url", ""),
                    "tomtom_url":   i.get("tomtom_url", ""),
                    "source":       "tomtom_traffic",
                    "age_label":    i["age_label"],
                    "is_recent":    i["is_recent"],
                    "lat":          i.get("lat"),
                    "lon":          i.get("lon"),
                    "_tomtom_road": i.get("_tomtom_road", ""),
                    "_tomtom_category": i.get("_tomtom_category", ""),
                    "_tomtom_severity": i.get("_tomtom_severity", "low"),
                    "_road":        road,
                })

        if ENABLE_RSS:
            for i in fetch_rss_for_query(f"{road} traffic", max_items=MAX_ARTICLES_PER_SOURCE):
                articles.append({
                    "title":       i.get("title", ""),
                    "description": i.get("description", ""),
                    "url":         i.get("link", ""),
                    "source":      "rss",
                    "age_label":   i.get("age_label", "unknown date"),
                    "is_recent":   i.get("is_recent", True),
                    "_road":       road,
                })

        if ENABLE_NEWSAPI and NEWS_API_KEY:
            data = fetch_news(f"{road} Kolkata")
            for a in data.get("articles", [])[:MAX_ARTICLES_PER_SOURCE]:
                articles.append({
                    "title":       a.get("title", ""),
                    "description": a.get("description", ""),
                    "url":         a.get("url", ""),
                    "source":      "newsapi",
                    "age_label":   a.get("publishedAt", "unknown date"),
                    "is_recent":   True,
                    "_road":       road,
                })

    return deduplicate(articles, key="url")


# ── Event extraction pipeline ──────────────────────────────────────────────────

def _process_articles(articles: list[dict], session) -> list[dict]:
    """
    Run the full extraction pipeline on a list of articles.
    Uses your enhanced pipeline: location resolver, multi-factor confidence,
    impact duration, DB persist — same as main.py.
    """
    results = []

    for raw_article in articles:
        article = dict(raw_article)
        road    = article.get("_road", "unknown")
        text    = build_article_text(article.get("title", ""), article.get("description", ""))
        if not text.strip():
            continue

        source = article.get("source", "")
        event  = extract_event(text, article=article)
        if event is None:
            continue

        if not is_valid_extracted_event(event):
            continue

        resolved_location, loc_source, loc_inferred = resolve_location(
            event, article, road_context=road
        )

        keep, _ = enforce_location_or_drop(
            resolved_location=resolved_location,
            confidence=event.confidence,
            event_type=event.event_type.value,
            source=source,
        )
        if not keep:
            continue

        road_name  = event.road_name or road

        # ── Derive fetched_at from age_label ──────────────────────────────────
        # age_label is the only reliable timestamp for RSS/NewsAPI articles.
        # Without this, all events fall back to datetime.now() in graph_builder,
        # making stale events look like they just happened — corrupting temporal
        # features (hour_sin/cos, is_rush_hour, is_weekend) used by the HGNN.
        from llm.filter import parse_age_label_to_hours
        _age_hours = parse_age_label_to_hours(article.get("age_label", ""))
        article["fetched_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=_age_hours)
            if _age_hours is not None
            else datetime.now(timezone.utc)
        )

        sev_score  = compute_score(event.severity.value)
        enh_conf   = compute_enhanced_confidence(
            llm_confidence    = event.confidence,
            source            = source,
            severity          = event.severity.value,
            location          = resolved_location,
            location_inferred = loc_inferred,
            age_label         = article.get("age_label", "unknown date"),
            is_recent         = article.get("is_recent", True),
        )
        weighted   = compute_weighted_score(event.severity.value, enh_conf)
        api_end    = article.get("_api_end_time")
        duration   = estimate_impact(
            event_type        = event.event_type.value,
            severity          = event.severity.value,
            llm_estimated_end = event.estimated_end_time,
            llm_duration_mins = event.impact_duration_mins,
            api_end_time      = api_end,
            age_label         = article.get("age_label", "unknown date"),
        )
        pub_date   = format_date_ddmmyy(article.get("age_label", "")) or format_today_ddmmyy()

        # ── Forward-geocode for non-TomTom events that lack coordinates ───────
        # TomTom/HERE articles carry lat/lon directly from the API response.
        # RSS/NewsAPI/scraper articles never carry coordinates — the LLM only
        # extracts a location *name*, leaving lat/lon as None. Without these,
        # build_graph_from_db() produces no road-road spatial adjacency edges
        # for the ~60-70% of events that come from non-TomTom sources, so the
        # HGNN trains on a graph that is mostly disconnected.
        #
        # We forward-geocode the resolved location name ONCE per unique string
        # (cached in _fwd_geocode_cache for the process lifetime) and write the
        # result back into the article dict so the same coordinates flow into
        # both the DB record below AND the results list appended after it.
        if article.get("lat") is None and resolved_location:
            geo = _forward_geocode(resolved_location)
            if geo:
                article["lat"], article["lon"] = geo

        # Persist
        record = TrafficEvent(
            source               = source,
            source_url           = article.get("url"),
            tomtom_url           = article.get("tomtom_url"),
            raw_text             = text,
            published_date       = pub_date,
            event_type           = event.event_type.value,
            transport_relevant   = getattr(event, "transport_relevant", True),
            location             = resolved_location,
            location_inferred    = loc_inferred,
            location_source      = loc_source,
            road_name            = road_name,
            reason               = event.reason,
            time_mentioned       = event.time_mentioned,
            is_future_event      = event.is_future_event,
            severity             = event.severity.value,
            severity_score       = sev_score,
            confidence           = enh_conf,
            llm_confidence       = event.confidence,
            source_reliability   = get_source_reliability(source),
            # Coordinates — persisted so HGNN trainer can build road-road
            # spatial adjacency edges from the DB (was always NULL before)
            lat                  = float(article["lat"])  if article.get("lat")  is not None else None,
            lon                  = float(article["lon"])  if article.get("lon")  is not None else None,
            start_time_display   = duration["start_time"],
            estimated_end_time   = duration["estimated_end_time"],
            impact_duration_mins = duration["impact_duration_mins"],
            impact_duration_label= duration["impact_duration_label"],
            duration_source      = duration["duration_source"],
        )
        session.add(record)
        session.commit()

        results.append({
            "event_type":           event.event_type.value,
            "location":             resolved_location,
            "location_inferred":    loc_inferred,
            "road_name":            road_name,
            "severity":             event.severity.value,
            "confidence":           enh_conf,
            "reason":               event.reason,
            "time_mentioned":       event.time_mentioned,
            "is_future_event":      event.is_future_event,
            "severity_score":       sev_score,
            "weighted_score":       weighted,
            "age_label":            article.get("age_label", "unknown date"),
            "is_recent":            article.get("is_recent", True),
            "published_date":       pub_date,
            "source":               source,
            "source_url":           article.get("url", ""),
            "tomtom_url":           article.get("tomtom_url", ""),
            "lat":                  article.get("lat"),
            "lon":                  article.get("lon"),
            "fetched_at":           article["fetched_at"],   # for HGNN temporal features
            "impact_duration_label": duration["impact_duration_label"],
            "estimated_end_time":   duration["estimated_end_time"],
            "color":                SEVERITY_COLOR.get(event.severity.value, "#95a5a6"),
        })

    return results


def _persist_hgnn_outputs(events: list[dict], session) -> None:
    """
    Write HGNN-adjusted confidence, severity, and multiplier back to the DB
    for events that were processed this run.

    This closes the training feedback loop:
      1. Event enters DB with LLM confidence/severity (initial labels)
      2. HGNN adjusts confidence and optionally corrects severity
      3. We write hgnn_confidence, hgnn_severity, hgnn_multiplier back
      4. Next trainer run can use these as improved training targets

    Matches by source_url (unique per article). Falls back to raw_text
    prefix match for articles without a URL. Non-fatal if DB update fails.
    """
    if not events:
        return

    try:
        from sqlalchemy import text as sql_text
        updated = 0
        for ev in events:
            # Only update events that HGNN actually touched
            hgnn_conf = ev.get("hgnn_confidence")
            if hgnn_conf is None:
                continue

            source_url   = ev.get("source_url", "")
            hgnn_sev     = ev.get("hgnn_severity")
            hgnn_mult    = ev.get("hgnn_multiplier")
            sev_corrected = ev.get("severity_corrected", False)

            if source_url:
                session.execute(sql_text("""
                    UPDATE traffic_events
                    SET    hgnn_confidence    = :hc,
                           hgnn_severity      = :hs,
                           hgnn_multiplier    = :hm,
                           severity_corrected = :sc
                    WHERE  source_url = :url
                """), {
                    "hc":  round(float(hgnn_conf), 4),
                    "hs":  hgnn_sev,
                    "hm":  round(float(hgnn_mult), 4) if hgnn_mult is not None else None,
                    "sc":  bool(sev_corrected),
                    "url": source_url,
                })
                updated += 1

        session.commit()
        if updated:
            print(f"  [DB] HGNN outputs written back for {updated} events.")

    except Exception as e:
        print(f"  [DB] HGNN write-back warning (non-fatal): {e}")


def _fetch_city_weather(all_routes: list[dict]) -> dict:
    """
    Fetch weather ONCE for the city using the midpoint of the first route.

    Weather is city-wide (heavy rain / high winds affect all routes equally)
    so we sample a single representative location rather than per-route points.
    Falls back to the first route's midpoint coordinate, or Kolkata city centre.

    Returns a weather profile dict:
      {avg_wsi, max_wsi, severity, sample_points, score, success, coords_used}
    """
    if not OPENWEATHER_API_KEY:
        return {"avg_wsi": None, "max_wsi": None, "severity": "unknown",
                "sample_points": 0, "score": 0.0, "success": False,
                "coords_used": None}

    # Pick a representative point — midpoint of the first route's coord list,
    # or Kolkata city centre as fallback (22.5726°N, 88.3639°E).
    KOLKATA_CENTRE = (22.5726, 88.3639)
    sample_coord = KOLKATA_CENTRE

    for route in all_routes:
        coords = route.get("coords") or []
        if coords:
            mid_idx = len(coords) // 2
            sample_coord = coords[mid_idx]
            break

    try:
        lat, lon = float(sample_coord[0]), float(sample_coord[1])
        weather_pt = _fetch_weather_point(lat, lon)

        if not weather_pt:
            return {"avg_wsi": None, "max_wsi": None, "severity": "unknown",
                    "sample_points": 0, "score": 0.0, "success": False,
                    "coords_used": sample_coord}

        risk = compute_weather_risk(weather_pt)
        wsi  = risk["wsi"]
        # Scale WSI [0–1] → disruption-score scale (max ~10 pts)
        # WSI=0.3 (medium rain) → +3 pts,  WSI=0.7 (heavy monsoon) → +7 pts
        weather_score = round(wsi * 10, 2)

        print(f"  [Weather] City-wide sample at ({lat:.4f},{lon:.4f}) "
              f"WSI={wsi:.2f} → city_weather_score={weather_score}")

        return {
            "avg_wsi":     wsi,
            "max_wsi":     wsi,         # single point, avg == max
            "severity":    risk["severity"],
            "sample_points": 1,
            "score":       weather_score,
            "success":     True,
            "coords_used": sample_coord,
            "raw": weather_pt,          # full OWM fields for frontend tooltip
        }

    except Exception as e:
        print(f"  [Weather] City-wide fetch failed: {e}")
        return {"avg_wsi": None, "max_wsi": None, "severity": "unknown",
                "sample_points": 0, "score": 0.0, "success": False,
                "coords_used": sample_coord}


def _score_route(route_data: dict, events: list[dict],
                 city_weather: dict | None = None,
                 cascade_cache: dict | None = None) -> dict:
    """
    Score a route using HGNN road-probability as a continuous per-event
    weight multiplier rather than a binary route-specific/area-wide split.

    Core idea (HGNN integration):
      For every event that matches this route, its score contribution is:

          contribution = weighted_score × hgnn_road_multiplier

      where hgnn_road_multiplier is derived from the HGNN's predicted
      disruption probability for the event's road ON THIS ROUTE'S graph.

      Since each route has a different set of road nodes and edges, the HGNN
      produces different road disruption probabilities per route. Two routes
      that textually match the same 4 events will now score differently
      because the HGNN topology is different — one route's roads may cluster
      around a disrupted zone, another's may not.

    Multiplier mapping:
      hgnn_prob ≥ 0.7  → multiplier = 1.0   (confirmed disruption, full weight)
      hgnn_prob ≥ 0.5  → multiplier = 0.75  (probable, slight discount)
      hgnn_prob ≥ 0.3  → multiplier = 0.45  (possible, significant discount)
      hgnn_prob < 0.3  → multiplier = 0.20  (unlikely on this route)
      no hgnn data     → multiplier = 0.50  (neutral fallback)

    This directly solves the "identical score" problem: routes sharing the
    same text-matched events get different scores because their HGNN graphs
    assign different road probabilities to those events' roads.
    """
    coords     = route_data.get("coords")
    road_names = route_data.get("road_names", [])

    # ── Step 1: Run HGNN on THIS route's graph ────────────────────────────────
    # Build the graph with only this route's road names marked as is_on_route.
    # This means each route gets a unique graph topology → unique road probs.
    hgnn_road_probs: dict[str, float] = {}
    hgnn_available = False
    try:
        from hgnn.inference import get_inference
        hgnn   = get_inference()
        result = hgnn.predict(events, road_names)
        if result:
            hgnn_road_probs = result.get("road_disruption_probs", {})
            hgnn_available  = bool(hgnn_road_probs)
    except Exception:
        pass   # HGNN unavailable — use neutral multiplier for all events

    def _hgnn_multiplier(ev: dict) -> float:
        """
        Convert HGNN road probability → a score multiplier for this event.

        Looks up the event's road in THIS route's hgnn_road_probs dict.
        Falls back to 0.50 (neutral) if HGNN is unavailable or road unknown.
        """
        if not hgnn_available:
            return 0.50
        ev_road = (ev.get("road_name") or "").lower().strip()
        prob = hgnn_road_probs.get(ev_road)
        if prob is None:
            # Try partial match — event road may be a substring of OSM road name
            for osm_road, p in hgnn_road_probs.items():
                if ev_road and (ev_road in osm_road or osm_road in ev_road):
                    prob = p
                    break
        if prob is None:
            return 0.50   # unknown road — neutral
        if prob >= 0.70:  return 1.00
        if prob >= 0.50:  return 0.75
        if prob >= 0.30:  return 0.45
        return 0.20

    # ── Step 2: Cascade expansion (cached) ───────────────────────────────────
    cache_key = tuple(sorted(road_names))
    if cascade_cache is not None and cache_key in cascade_cache:
        cascade_roads = cascade_cache[cache_key]
    else:
        cascade_roads = get_cascade_road_names(road_names, events)
        if cascade_cache is not None:
            cascade_cache[cache_key] = cascade_roads

    expanded_road_names = road_names + cascade_roads

    # ── Step 3: Spatial + road-name event matching ────────────────────────────
    route_events, _ = filter_by_route_relevance(
        results      = events,
        route_roads  = expanded_road_names,
        route_coords = [tuple(c) for c in coords] if coords else None,
        corridor_km  = 1.0,
    )

    # ── Step 4: Tag route-specific vs area-wide (for UI display only) ─────────
    route_road_set = {r.lower() for r in road_names}
    for ev in route_events:
        ev_road = (ev.get("road_name") or "").lower()
        ev_loc  = (ev.get("location") or "").lower()
        is_specific = any(
            r in ev_road or r in ev_loc
            for r in route_road_set if len(r) >= 5
        )
        ev["route_specific"]     = is_specific
        ev["hgnn_multiplier"]    = round(_hgnn_multiplier(ev), 3)

    recent = [
        e for e in route_events
        if e.get("is_recent", True) and not e.get("is_future_event", False)
    ]

    route_specific = [e for e in recent if e.get("route_specific")]
    area_wide      = [e for e in recent if not e.get("route_specific")]

    # ── Step 5: HGNN-weighted disruption score ────────────────────────────────
    # Every event is weighted by its HGNN road disruption probability on THIS
    # route. Same event → different weight on different routes → different score.
    #
    # Area-wide events use a spatially-graduated discount instead of a flat 0.40×.
    # If the event has lat/lon (TomTom or forward-geocoded), we compute its
    # Haversine distance to the nearest route node and map that to a [0.15–0.55]
    # discount. Events close to the route corridor score higher; truly city-wide
    # events (no coords or very far) fall back to 0.20. This breaks the
    # "identical score" tie when HGNN is untrained (all multipliers = 0.50) by
    # giving each route a different proximity profile for the same event pool.
    import math as _math

    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = _math.radians(lat2 - lat1)
        dlon = _math.radians(lon2 - lon1)
        a = (_math.sin(dlat / 2) ** 2 +
             _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2)) *
             _math.sin(dlon / 2) ** 2)
        return R * 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1 - a))

    def _area_wide_discount(ev: dict) -> float:
        """
        Proximity-based discount for area-wide events.

        If the event has coordinates and the route has coordinates, compute
        the minimum Haversine distance from the event to any route node and
        map it to a discount factor:
          ≤ 0.5 km  → 0.55  (practically on-route, near-specific weight)
          ≤ 1.5 km  → 0.40  (within corridor, standard city discount)
          ≤ 3.0 km  → 0.28  (nearby area)
          > 3.0 km  → 0.15  (genuinely city-wide, far away)
          no coords → 0.20  (unknown proximity, conservative discount)

        Two routes receiving the same area-wide event will get different
        discounts when one's corridor is closer to the event's location.
        """
        ev_lat = ev.get("lat")
        ev_lon = ev.get("lon")
        if ev_lat is None or ev_lon is None or not coords:
            return 0.20   # no spatial data — conservative city-wide discount

        # Sample up to 20 evenly-spaced route nodes for efficiency
        # (routes can have hundreds of OSM nodes; checking all is wasteful)
        step = max(1, len(coords) // 20)
        sample = coords[::step]

        min_dist = min(
            _haversine_km(float(ev_lat), float(ev_lon), float(c[0]), float(c[1]))
            for c in sample
        )

        if min_dist <= 0.5:  return 0.55
        if min_dist <= 1.5:  return 0.40
        if min_dist <= 3.0:  return 0.28
        return 0.15

    disruption_score = sum(
        ev["weighted_score"] * ev["hgnn_multiplier"]
        for ev in recent if ev.get("route_specific")
    ) + sum(
        ev["weighted_score"] * ev["hgnn_multiplier"] * _area_wide_discount(ev)
        for ev in recent if not ev.get("route_specific")
    )

    # ── Step 6: Per-route HGNN topology score ────────────────────────────────
    # score_route_with_hgnn() runs a separate HGNN pass on route-only events
    # and returns max/avg road disruption probability scaled to [0,10].
    # This captures the structural disruption of the route's road network
    # independent of individual event text matching.
    hgnn_score = score_route_with_hgnn(route_data, route_events)

    # ── Step 7: Weather — city-wide ───────────────────────────────────────────
    weather_score = 0.0
    weather_out   = {"avg_wsi": None, "max_wsi": None, "severity": "unknown",
                     "sample_points": 0, "score": 0.0, "city_wide": True}

    if city_weather and city_weather.get("success"):
        weather_score = city_weather.get("score", 0.0)
        weather_out   = {
            "avg_wsi":       city_weather.get("avg_wsi"),
            "max_wsi":       city_weather.get("max_wsi"),
            "severity":      city_weather.get("severity", "unknown"),
            "sample_points": city_weather.get("sample_points", 0),
            "score":         weather_score,
            "city_wide":     True,
            "raw":           city_weather.get("raw", {}),
        }

    # Final composite:
    #   disruption_score — HGNN-weighted event severity × confidence
    #   hgnn_score       — route topology disruption from HGNN graph (0–10)
    #   weather_score    — city-wide weather risk (identical for all routes)
    # HGNN topology score gets 0.5× weight so it influences but doesn't dominate
    # when the model has limited training data.
    total_risk = round(disruption_score + weather_score + 0.50 * hgnn_score, 2)

    if total_risk >= 25:
        risk_level = "CRITICAL"
    elif total_risk >= 12:
        risk_level = "HIGH"
    elif total_risk >= 5:
        risk_level = "MODERATE"
    elif total_risk > 0:
        risk_level = "LOW"
    else:
        risk_level = "CLEAR"

    return {
        **route_data,
        "risk_score":            total_risk,
        "disruption_score":      round(disruption_score, 2),
        "weather_score":         weather_score,
        "hgnn_score":            round(hgnn_score, 3),
        "hgnn_road_probs":       hgnn_road_probs,        # per-road prob for debugging
        "cascade_roads":         cascade_roads,
        "risk_level":            risk_level,
        "risk_color":            RISK_COLOR.get(risk_level, "#27ae60"),
        "matched_events":        route_events,
        "route_specific_events": len(route_specific),
        "area_wide_events":      len(area_wide),
        "event_count":           len(recent),
        "weather":               weather_out,
    }


def _pick_best_route(scored: list[dict]) -> int:
    """
    Best route = lowest composite: effective_time × (1 + risk_score / MAX_RISK).

    effective_time = travel_time_min + metro_wait_min
      Metro wait (time until the next train departs) is extracted from the first
      metro segment's next_train.minutes_away field. A route with a 12-min wait
      should not beat a route with a 1-min wait just because its in-motion time
      happens to look identical after integer rounding.

    Calibration (MAX_RISK = 20):
      risk=0  (CLEAR)    → 1.0×  — pure travel time, no penalty
      risk=5  (MODERATE) → 1.25× — 25% time penalty; still beats a 30% longer clear route
      risk=12 (HIGH)     → 1.6×  — a 10-min route now costs 16 min equivalent
      risk=25 (CRITICAL) → 2.25× — overrides up to ~55% time savings on safer route
    """
    MAX_RISK = 20.0

    def _metro_wait_min(r: dict) -> float:
        """Return the wait time (minutes) for the first metro segment, or 0."""
        for seg in r.get("segments") or []:
            if seg.get("type") == "metro":
                nt = seg.get("next_train")
                if nt and isinstance(nt.get("minutes_away"), (int, float)):
                    return float(nt["minutes_away"])
        return 0.0

    def composite(r: dict) -> float:
        effective_time = r["travel_time_min"] + _metro_wait_min(r)
        return effective_time * (1.0 + r["risk_score"] / MAX_RISK)

    return min(range(len(scored)), key=lambda i: composite(scored[i]))

# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/locations")
def get_locations():
    from routing.metro_timetable import ALL_LINES
    from transit.bus_graph import load_stops as _load_bus_stops

    # Collect all unique metro station names, preserving display order:
    # Blue → Green → Purple → Orange → Yellow
    seen_metro: set[str] = set()
    metro_locations = []
    _id = len(KOLKATA_LOCATIONS) + 1
    for line in ALL_LINES:
        for stn in line.stations:
            if stn.lower() not in seen_metro:
                seen_metro.add(stn.lower())
                metro_locations.append({
                    "id":   _id,
                    "name": stn,
                    "desc": f"{line.display_name} station",
                    "type": "metro",
                    "line": line.line,
                    "color": line.color_hex,
                })
                _id += 1

    # Bus stops — flat list sorted by name for the dropdown
    bus_stops = []
    try:
        raw_stops = _load_bus_stops()
        for stop_id, info in sorted(raw_stops.items(), key=lambda x: x[1]["name"]):
            bus_stops.append({
                "id":      stop_id,
                "name":    info["name"],
                "desc":    "Bus stop",
                "type":    "bus",
                "lat":     info["lat"],
                "lon":     info["lon"],
            })
    except Exception as _e:
        print(f"[API] Could not load bus stops: {_e}")

    return {
        "locations":      KOLKATA_LOCATIONS,
        "metro_stations": metro_locations,
        "bus_stops":      bus_stops,
    }


@app.get("/api/metro-overlay")
def metro_overlay():
    """GeoJSON overlay of all Kolkata Metro lines + stations for map display."""
    return get_metro_geojson_overlay()


@app.get("/api/bus-overlay")
def bus_overlay():
    """
    GeoJSON FeatureCollection of all Kolkata bus stops for optional map display.
    Each stop is a Point feature with {stop_id, name} properties.
    """
    from transit.bus_graph import load_stops as _load_bus_stops
    try:
        stops = _load_bus_stops()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bus stop data unavailable: {e}")

    features = []
    for stop_id, info in stops.items():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [info["lon"], info["lat"]],
            },
            "properties": {
                "stop_id": stop_id,
                "name":    info["name"],
            },
        })
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/bus-network")
def bus_network():
    """
    Returns the full Kolkata bus network for map overlay and panel display.

    Response:
      routes: [{route_id, route_name, route_type, stops: [{stop_id, name, lat, lon}], coords: [[lon,lat],...]}]
      stops:  [{stop_id, name, lat, lon}]   -- deduplicated flat list of all stops
    """
    from transit.bus_graph import (
        load_stops as _load_bus_stops,
        load_routes as _load_bus_routes,
        load_route_sequences as _load_seqs,
    )
    try:
        stops_map = _load_bus_stops()
        routes_map = _load_bus_routes()
        sequences = _load_seqs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bus data unavailable: {e}")

    routes_out = []
    for route_id, stop_seq in sequences.items():
        stop_seq_sorted = sorted(stop_seq, key=lambda x: x[0])
        stop_list = []
        coords    = []
        for _, sid in stop_seq_sorted:
            if sid not in stops_map:
                continue
            info = stops_map[sid]
            stop_list.append({"stop_id": sid, "name": info["name"],
                               "lat": info["lat"], "lon": info["lon"]})
            coords.append([info["lon"], info["lat"]])
        if not stop_list:
            continue
        meta = routes_map.get(route_id, {})
        routes_out.append({
            "route_id":   route_id,
            "route_name": meta.get("route_name", route_id),
            "route_type": meta.get("route_type", ""),
            "stops":      stop_list,
            "coords":     coords,
            "num_stops":  len(stop_list),
        })

    # Sort: AC first, then Government, then others
    TYPE_ORDER = {"AC": 0, "Government": 1, "Non-AC": 2, "Mini": 3}
    routes_out.sort(key=lambda r: (TYPE_ORDER.get(r["route_type"], 9), r["route_name"]))

    # Flat deduplicated stop list
    seen_stops: set[str] = set()
    flat_stops = []
    for sid, info in stops_map.items():
        if sid not in seen_stops:
            seen_stops.add(sid)
            flat_stops.append({"stop_id": sid, "name": info["name"],
                                "lat": info["lat"], "lon": info["lon"]})
    flat_stops.sort(key=lambda s: s["name"])

    return {"routes": routes_out, "stops": flat_stops}


@app.get("/api/hgnn-status")
def hgnn_status():
    """
    Returns HGNN model readiness status.
    - available: torch is installed
    - ready:     model weights loaded and inference active
    - message:   human-readable status / next step
    """
    from hgnn.integration import get_hgnn_status
    return get_hgnn_status()


@app.post("/api/explain-route")
def explain_route(req: DisruptionRequest):
    """
    Returns HGNN attention-based explanation for why a route has its risk score.
    Exposes which events had highest attention weight → explainable AI.
    """
    from hgnn.integration import explain_route_risk

    # Use first route's road names if no specific route provided
    road_names = req.road_names or []
    route_data = {"road_names": road_names}

    # Quick event extraction (reuse disruptions endpoint logic)
    events: list[dict] = []
    session = get_session()
    try:
        from sqlalchemy import text as sql_text
        from config import DATABASE_URL
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL, echo=False)
        with engine.connect() as conn:
            rows = conn.execute(sql_text("""
                SELECT event_type, severity, confidence, road_name, location,
                       source, 0 as is_future_event, impact_duration_mins
                FROM traffic_events
                ORDER BY fetched_at DESC LIMIT 100
            """)).fetchall()
        for row in rows:
            events.append({
                "event_type": row[0] or "unknown",
                "severity":   row[1] or "low",
                "confidence": float(row[2] or 0.5),
                "road_name":  row[3],
                "location":   row[4],
                "source":     row[5] or "unknown",
                "is_future_event": bool(row[6]),
                "impact_duration_mins": int(row[7]) if row[7] else 60,
                "is_recent": True,
                "weighted_score": float(row[2] or 0.5),
            })
    except Exception as e:
        print(f"  [Explain] DB read failed: {e}")
    finally:
        session.close()

    explanation = explain_route_risk(route_data, events)
    return {
        "road_names":  road_names,
        "explanation": explanation,
        "generated_at": now_iso(),
    }


@app.get("/api/metro-lines")
def metro_lines_summary():
    """Summary of all metro lines — first/last trains, frequency, stations."""
    from routing.metro_timetable import get_all_lines_summary
    return {"lines": get_all_lines_summary()}


@app.get("/api/next-metro/{station_name}")
def next_metro(station_name: str, n: int = 4):
    """
    Return the next n trains at a station across all lines and directions.
    Uses current IST time on the server.

    Returns 404 if no trains are found (no service / outside operating hours).
    Example: GET /api/next-metro/Esplanade?n=4
    """
    from routing.metro_timetable import next_trains_at_station, IST
    now    = datetime.now(tz=IST)
    result = next_trains_at_station(station_name, now=now, n=n)
    if not result["trains"]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No upcoming trains at '{station_name}' right now. "
                "Service may have ended for the day, or the station name may not match."
            )
        )
    return result


@app.post("/api/route")
def compute_routes_only(req: RouteRequest):
    """
    Step 1 — FAST (~2s for drive; ~5s first-time for walk/bike).
    Computes routes for the requested mode. Metro uses static station data.
    All routes get placeholder risk=LOW/score=0 until /api/disruptions runs.
    """
    modes = req.modes if req.modes is not None else [req.mode or "drive"]
    if not modes:
        raise HTTPException(status_code=400, detail="At least one transport mode must be provided.")
    try:
        if len(modes) == 1:
            result = get_routes_for_mode(req.source, req.destination, modes[0])  # type: ignore
        else:
            result = get_routes_for_modes(req.source, req.destination, modes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {e}")

    mode = result.get("mode") or (modes[0] if len(modes) == 1 else "+".join(modes))

    # Mode-specific colours for route lines
    MODE_COLOR = {
        "drive": "#1a73e8",
        "walk":  "#34a853",
        "bike":  "#fbbc04",
        "metro": "#9c27b0",
        "metro+walk":       "#9c27b0",
        "metro+bike":       "#9c27b0",
        "metro+drive":      "#9c27b0",
        "metro+walk+drive": "#9c27b0",
        "bus":   "#FF5722",
    }
    base_color = MODE_COLOR.get(mode, "#1a73e8")

    for r in result["routes"]:
        r["risk_score"]     = 0.0
        r["risk_level"]     = "LOW"
        r["risk_color"]     = base_color
        r["matched_events"] = []
        r["event_count"]    = 0
        r["is_best"]        = False
        r.setdefault("mode", mode)
        # Strip any segments that have no metro leg in a metro+walk result
        # (guards against plain-walk routes leaking into the metro+walk section)
        if "metro" in mode and r.get("segments"):
            has_metro_seg = any(s.get("type") == "metro" for s in r["segments"])
            if not has_metro_seg:
                r["_no_metro_leg"] = True

    # Filter out walk-only routes that leaked into a metro+walk result
    if "metro" in mode:
        result["routes"] = [
            r for r in result["routes"] if not r.get("_no_metro_leg")
        ]

    if result["routes"]:
        result["routes"][0]["is_best"] = True

    all_roads = list({road for route in result["routes"] for road in route.get("road_names", [])})

    return {
        **result,
        "all_road_names":     all_roads,
        "disruptions_loaded": False,
        "markers":            [],
        "recent_events":      0,
        "best_route_id":      result["routes"][0]["id"] if result["routes"] else None,
        "generated_at":       now_iso(),
        "mode":               mode,
        "metro_note":         result.get("metro_note"),   # surfaced when no metro routes found
        "bus_note":           result.get("bus_note"),     # surfaced when no direct bus route found
    }


@app.post("/api/disruptions")
def fetch_disruptions(req: DisruptionRequest):
    """
    Step 2 — SLOW (~30-60s). Fetches all sources, runs LLM extraction,
    scores each route spatially, returns updated risk data + event markers.
    """
    events: list[dict] = []
    session = get_session()
    try:
        # ── Fetch TomTom ONCE for the whole city, then reuse ──────────────────
        tomtom_cache: list[dict] = []
        if ENABLE_TOMTOM and TOMTOM_API_KEY:
            raw_tomtom = fetch_tomtom_incidents(max_items=50)
            for i in raw_tomtom:
                tomtom_cache.append({
                    "title":            i["title"],
                    "description":      i["description"],
                    "url":              i.get("url", ""),
                    "tomtom_url":       i.get("tomtom_url", ""),
                    "source":           "tomtom_traffic",
                    "age_label":        i["age_label"],
                    "is_recent":        i["is_recent"],
                    "lat":              i.get("lat"),
                    "lon":              i.get("lon"),
                    "_tomtom_road":     i.get("_tomtom_road", ""),
                    "_tomtom_category": i.get("_tomtom_category", ""),
                    "_tomtom_severity": i.get("_tomtom_severity", "low"),
                    "_road":            i.get("_tomtom_road") or "Kolkata (TomTom)",
                })

        # ── Per-road articles (uses cached TomTom, no extra API calls) ────────
        road_articles = _fetch_road_articles(req.road_names, tomtom_cache)

        # ── City-wide sources (TomTom already fetched, skip re-fetch) ─────────
        city_articles = _fetch_citywide(skip_tomtom=True)
        city_articles.extend(tomtom_cache)

        all_articles = deduplicate(road_articles + city_articles, key="url")

        # Pre-LLM filters
        all_articles, fstats = filter_articles(all_articles, max_age_days=2)
        print(f"  [API] Articles after filter: {fstats['kept']} kept "
              f"(age: {fstats['dropped_age']}, geo: {fstats['dropped_geography']}, "
              f"relevance: {fstats['dropped_relevance']})")

        # LLM extraction
        raw_events = _process_articles(all_articles, session)
        print(f"  [API] Events extracted: {len(raw_events)}")

        # Semantic dedup
        events = semantic_deduplicate_events(raw_events)
        print(f"  [API] After dedup: {len(events)} events")

        # ── HGNN confidence enhancement — uses ALL road names for full city
        # graph context (multi-source corroboration, spatial clustering) ───────
        all_route_road_names = list({
            road
            for r in (req.routes or [])
            for road in (r.road_names if hasattr(r, "road_names") else r.get("road_names", []))
        }) or req.road_names
        events = enhance_event_confidences(events, all_route_road_names)

        # ── Write HGNN outputs back to DB for training feedback loop ──────────
        # After HGNN adjusts confidence/severity, persist those values so the
        # trainer can load them as improved targets on the next training run.
        # Only updates rows that exist (matched by source_url or raw_text hash).
        _persist_hgnn_outputs(events, session)

    except Exception as e:
        import traceback
        print(f"  [API] ERROR in disruption fetch: {e}")
        traceback.print_exc()
        # Return partial results (empty events, scored with 0 risk) rather than 500
    finally:
        session.close()

    # Build route list from request
    route_list = []
    if req.routes:
        for r in req.routes:
            # Pydantic v2 uses model_dump(), v1 uses dict()
            if hasattr(r, "model_dump"):
                route_list.append(r.model_dump())
            elif hasattr(r, "dict"):
                route_list.append(r.dict())
            else:
                route_list.append(dict(r))
    else:
        for idx, rn in enumerate(req.road_names):
            route_list.append({
                "id": idx, "label": f"Route {idx+1}",
                "road_names": [rn], "distance_km": 0.0,
                "travel_time_min": 0.0, "geojson": {},
                "coords": [], "risk_score": 0.0,
                "risk_level": "LOW", "risk_color": RISK_COLOR["LOW"],
                "matched_events": [], "event_count": 0, "is_best": False,
            })

    # ── Context-aware re-routing for metro/multimodal modes ───────────────────
    # Now that we have extracted events, re-run the metro routing with full
    # disruption context so blocked stations and path scores use real data.
    req_mode = req.mode or "drive"
    is_metro_mode = req_mode in ("metro", "metro+walk", "metro+bike", "metro+drive", "metro+walk+drive")
    if is_metro_mode and events:
        try:
            print(f"  [API] Re-routing {req_mode} with {len(events)} disruption events for context")
            modes_for_reroute = req_mode.split("+")
            if len(modes_for_reroute) == 1:
                rerouted = get_routes_for_mode(
                    req.source, req.destination, modes_for_reroute[0], events=events
                )
            else:
                rerouted = get_routes_for_modes(
                    req.source, req.destination, modes_for_reroute, events=events
                )
            # Replace route_list with context-aware routes
            route_list = rerouted.get("routes", route_list)
            print(f"  [API] Context-aware re-routing produced {len(route_list)} routes")
        except Exception as e:
            print(f"  [API] Context-aware re-routing failed, using original routes: {e}")

    # ── City-wide weather — fetched ONCE, applied equally to all routes ─────
    city_weather = _fetch_city_weather(route_list)

    cascade_cache: dict = {}
    scored = [_score_route(r, events, city_weather=city_weather,
                           cascade_cache=cascade_cache) for r in route_list]
    best_i = _pick_best_route(scored) if scored else 0
    for i, r in enumerate(scored):
        r["is_best"] = (i == best_i)

    # Build markers — TomTom events use lat/lon, others need geocoding (future work)
    markers = []
    for ev in events:
        if ev.get("is_recent") and ev.get("location"):
            markers.append({
                "location":    ev["location"],
                "event_type":  ev["event_type"],
                "severity":    ev["severity"],
                "reason":      ev["reason"],
                "age_label":   ev.get("age_label", ""),
                "color":       ev.get("color", "#95a5a6"),
                "source":      ev.get("source", ""),
                "source_url":  ev.get("source_url", ""),
                "tomtom_url":  ev.get("tomtom_url", ""),
                "is_future":   ev.get("is_future_event", False),
                "duration":    ev.get("impact_duration_label", ""),
                "lat":         ev.get("lat"),
                "lon":         ev.get("lon"),
            })

    return {
        "routes":        scored,
        "best_route_id": scored[best_i]["id"] if scored else None,
        "events":        events,
        "markers":       markers,
        "total_events":  len(events),
        "recent_events": len([e for e in events if e.get("is_recent")]),
        "generated_at":  now_iso(),
        "source":        req.source,
        "destination":   req.destination,
        "src_coords":    None,
        "dst_coords":    None,
        # City-wide weather context — same value shown for all routes
        "city_weather":  city_weather,
        "mode":          req.mode,
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────

# Suppress browser auto-requests for favicon/apple-touch-icon
_FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    b'<text y=".9em" font-size="90">\xf0\x9f\x9a\xa6</text></svg>'  # 🚦
)

@app.get("/favicon.ico", include_in_schema=False)
@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()