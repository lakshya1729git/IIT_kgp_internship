"""
multimodal.py — Mode-aware routing for Kolkata Traffic AI
==========================================================
Supports: drive | walk | bike | metro (all 5 Kolkata Metro lines)

Multimodal improvements:
  - Multi-interchange BFS path-finding (any number of line changes)
  - Per-segment coloured GeoJSON (feeder road + metro tunnel geometry)
  - Multiple route alternatives (vary boarding/alighting + feeder legs)
  - Disruption context-awareness (metro closure events suppress stations)
"""

from __future__ import annotations

import os
import pickle
import math
import re
from collections import deque
from datetime import datetime
from typing import Literal

import osmnx as ox

from config import GRAPH_CACHE_PATH, MAX_ROUTES, MIN_ROUTE_DIVERGENCE, MAX_CONSECUTIVE_DUPES

_CACHE_DIR  = os.path.dirname(GRAPH_CACHE_PATH)
_GRAPH_WALK = os.path.join(_CACHE_DIR, "graph_walk.pkl")
_GRAPH_BIKE = os.path.join(_CACHE_DIR, "graph_bike.pkl")

TransportMode = Literal["drive", "walk", "bike", "metro", "bus"]

MODE_SPEED_KMH: dict[str, float] = {
    "drive": 30.0, "walk": 5.0, "bike": 15.0, "metro": 35.0,
}

# ── All 5 metro lines — station coords ───────────────────────────────────────

METRO_STATIONS: list[dict] = [
    # ── BLUE LINE (North-South, Line 1) ──────────────────────────────────────
    {"id":"bl_dakshineswar",      "name":"Dakshineswar",         "lat":22.6536796,  "lon":88.36279,    "line":"blue"},
    {"id":"bl_baranagar",         "name":"Baranagar Road",       "lat":22.6536402,  "lon":88.3736444,  "line":"blue"},
    {"id":"bl_noapara",           "name":"Noapara",              "lat":22.6399176,  "lon":88.3940964,  "line":"blue"},
    {"id":"bl_dumdum",            "name":"Dum Dum",              "lat":22.6214620,  "lon":88.3924889,  "line":"blue"},
    {"id":"bl_belgachia",         "name":"Belgachia",            "lat":22.6060188,  "lon":88.3865456,  "line":"blue"},
    {"id":"bl_shyambazar",        "name":"Shyambazar",           "lat":22.6007116,  "lon":88.3702820,  "line":"blue"},
    {"id":"bl_shobhabazar",       "name":"Shobhabazar Sutanuti", "lat":22.5960748,  "lon":88.3652955,  "line":"blue"},
    {"id":"bl_girish_park",       "name":"Girish Park",          "lat":22.5872207,  "lon":88.3630412,  "line":"blue"},
    {"id":"bl_mg_road",           "name":"Mahatma Gandhi Road",  "lat":22.5808049,  "lon":88.3613813,  "line":"blue"},
    {"id":"bl_central",           "name":"Central",              "lat":22.5725323,  "lon":88.3582318,  "line":"blue"},
    {"id":"bl_chandni",           "name":"Chandni Chowk",        "lat":22.5666859,  "lon":88.3536634,  "line":"blue"},
    {"id":"bl_esplanade",         "name":"Esplanade",            "lat":22.5648375,  "lon":88.3516199,  "line":"blue"},
    {"id":"bl_park_street",       "name":"Park Street",          "lat":22.5544697,  "lon":88.3498285,  "line":"blue"},
    {"id":"bl_maidan",            "name":"Maidan",               "lat":22.5494127,  "lon":88.3484939,  "line":"blue"},
    {"id":"bl_rabindra_sadan",    "name":"Rabindra Sadan",       "lat":22.5413538,  "lon":88.3472950,  "line":"blue"},
    {"id":"bl_netaji_bhavan",     "name":"Netaji Bhavan",        "lat":22.5331454,  "lon":88.3456480,  "line":"blue"},
    {"id":"bl_jatin_das_park",    "name":"Jatin Das Park",       "lat":22.5243569,  "lon":88.3464894,  "line":"blue"},
    {"id":"bl_kalighat",          "name":"Kalighat",             "lat":22.5167914,  "lon":88.3459603,  "line":"blue"},
    {"id":"bl_tollygunge",        "name":"Tollygunge",           "lat":22.5078731,  "lon":88.3474928,  "line":"blue"},
    {"id":"bl_mahanayak",         "name":"Mahanayak Uttam Kumar","lat":22.4945906,  "lon":88.3451752,  "line":"blue"},
    {"id":"bl_netaji",            "name":"Netaji",               "lat":22.4810865,  "lon":88.3458921,  "line":"blue"},
    {"id":"bl_masterda",          "name":"Masterda Surya Sen",   "lat":22.4736072,  "lon":88.3607502,  "line":"blue"},
    {"id":"bl_gitanjali",         "name":"Gitanjali",            "lat":22.4695042,  "lon":88.3699437,  "line":"blue"},
    {"id":"bl_kavi_nazrul",       "name":"Kavi Nazrul",          "lat":22.4643036,  "lon":88.3806582,  "line":"blue"},
    {"id":"bl_shahid_khudiram",   "name":"Shahid Khudiram",      "lat":22.4661679,  "lon":88.3916762,  "line":"blue"},

    # ── GREEN LINE (East-West, Line 2) ────────────────────────────────────────
    {"id":"gl_howrah_maidan",     "name":"Howrah Maidan",        "lat":22.5835267,  "lon":88.3338090,  "line":"green"},
    {"id":"gl_howrah",            "name":"Howrah",               "lat":22.5830362,  "lon":88.3410134,  "line":"green"},
    {"id":"gl_mahakaran",         "name":"Mahakaran",            "lat":22.5725301,  "lon":88.3506717,  "line":"green"},
    {"id":"gl_esplanade",         "name":"Esplanade",            "lat":22.5648375,  "lon":88.3516199,  "line":"green"},
    {"id":"gl_sealdah",           "name":"Sealdah",              "lat":22.5665634,  "lon":88.3706984,  "line":"green"},
    {"id":"gl_phoolbagan",        "name":"Phoolbagan",           "lat":22.5721820,  "lon":88.3902921,  "line":"green"},
    {"id":"gl_sl_stadium",        "name":"Salt Lake Stadium",    "lat":22.5733536,  "lon":88.4035940,  "line":"green"},
    {"id":"gl_bengal_chemical",   "name":"Bengal Chemical",      "lat":22.5801737,  "lon":88.4013465,  "line":"green"},
    {"id":"gl_city_centre",       "name":"City Centre",          "lat":22.5910706,  "lon":88.4108739,  "line":"green"},
    {"id":"gl_central_park",      "name":"Central Park",         "lat":22.5903855,  "lon":88.4154789,  "line":"green"},
    {"id":"gl_karunamoyee",       "name":"Karunamoyee",          "lat":22.5864805,  "lon":88.4213914,  "line":"green"},
    {"id":"gl_sl_sector_v",       "name":"Salt Lake Sector V",   "lat":22.5810776,  "lon":88.4290090,  "line":"green"},

    # ── PURPLE LINE (Joka–Majerhat, Line 3) ──────────────────────────────────
    {"id":"pl_joka",              "name":"Joka",                 "lat":22.4520677,  "lon":88.3015804,  "line":"purple"},
    {"id":"pl_thakurpukur",       "name":"Thakurpukur",          "lat":22.4643386,  "lon":88.3074909,  "line":"purple"},
    {"id":"pl_sakherbazar",       "name":"Sakherbazar",          "lat":22.4749579,  "lon":88.3099487,  "line":"purple"},
    {"id":"pl_behala_chowrasta",  "name":"Behala Chowrasta",     "lat":22.4872187,  "lon":88.3132142,  "line":"purple"},
    {"id":"pl_behala_bazar",      "name":"Behala Bazar",         "lat":22.5004968,  "lon":88.3177963,  "line":"purple"},
    {"id":"pl_taratala",          "name":"Taratala",             "lat":22.5078226,  "lon":88.3203432,  "line":"purple"},
    {"id":"pl_majerhat",          "name":"Majerhat",             "lat":22.5191900,  "lon":88.3237635,  "line":"purple"},

    # ── ORANGE LINE (Kavi Subhash–Beleghata, Line 6) ─────────────────────────
    {"id":"ol_kavi_subhash",      "name":"Kavi Subhash",         "lat":22.4722638,  "lon":88.3978346,  "line":"orange"},
    {"id":"ol_satyajit_ray",      "name":"Satyajit Ray",         "lat":22.4839792,  "lon":88.3922501,  "line":"orange"},
    {"id":"ol_jyotirindra_nandi", "name":"Jyotirindra Nandi",    "lat":22.4959849,  "lon":88.3985185,  "line":"orange"},
    {"id":"ol_kavi_sukanta",      "name":"Kavi Sukanta",         "lat":22.5056316,  "lon":88.4010123,  "line":"orange"},
    {"id":"ol_hemanta",           "name":"Hemanta Mukhopadhyay", "lat":22.5148128,  "lon":88.4016147,  "line":"orange"},
    {"id":"ol_vip_bazar",         "name":"VIP Bazar",            "lat":22.5249681,  "lon":88.3961903,  "line":"orange"},
    {"id":"ol_ritwik_ghatak",     "name":"Ritwik Ghatak",        "lat":22.5330417,  "lon":88.3964400,  "line":"orange"},
    {"id":"ol_barun_sengupta",    "name":"Barun Sengupta",       "lat":22.5440086,  "lon":88.3993215,  "line":"orange"},
    {"id":"ol_beleghata",         "name":"Beleghata",            "lat":22.5507592,  "lon":88.4040364,  "line":"orange"},

    # ── YELLOW LINE (Noapara–Jai Hind Bimanbandar, Line 5) ───────────────────
    {"id":"yl_noapara",           "name":"Noapara",              "lat":22.6399953,  "lon":88.3940739,  "line":"yellow"},
    {"id":"yl_jessore_road",      "name":"Jessore Road",         "lat":22.6392320,  "lon":88.4297938,  "line":"yellow"},
    {"id":"yl_nagerbazar",        "name":"Nagerbazar",           "lat":22.6481480,  "lon":88.4279020,  "line":"yellow"},
    {"id":"yl_jai_hind",          "name":"Jai Hind Bimanbandar", "lat":22.6474634,  "lon":88.4373223,  "line":"yellow"},
]

_STATION_BY_ID   = {s["id"]: s for s in METRO_STATIONS}
_STATION_BY_NAME: dict[str, list[dict]] = {}
for _s in METRO_STATIONS:
    _STATION_BY_NAME.setdefault(_s["name"].lower(), []).append(_s)

# Per-line ordered lists for path finding
_LINE_STATIONS: dict[str, list[dict]] = {}
for _s in METRO_STATIONS:
    _LINE_STATIONS.setdefault(_s["line"], []).append(_s)

LINE_COLORS = {
    "blue":   "#2196F3",
    "green":  "#4CAF50",
    "purple": "#9C27B0",
    "orange": "#EA4200",
    "yellow": "#FFCE18",
}

# Physical interchange points: station name (normalised) → list of (line, station_id)
# A station can appear on more than two lines — add more tuples to extend.
_INTERCHANGES: dict[str, list[tuple[str, str]]] = {
    "esplanade":    [("blue", "bl_esplanade"),  ("green",  "gl_esplanade")],
    "noapara":      [("blue", "bl_noapara"),     ("yellow", "yl_noapara")],
    "kavi nazrul":  [("blue", "bl_kavi_nazrul"), ("orange", "ol_kavi_subhash")],
    # Kavi Subhash on Orange IS the same platform as Kavi Nazrul on Blue
    "kavi subhash": [("blue", "bl_kavi_nazrul"), ("orange", "ol_kavi_subhash")],
}

# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_station(lat: float, lon: float, line: str | None = None) -> dict:
    pool = [s for s in METRO_STATIONS if line is None or s["line"] == line]
    return min(pool, key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))


def _on_line(a: dict, b: dict, line_stations: list[dict]) -> list[dict] | None:
    ids = [s["id"] for s in line_stations]
    if a["id"] in ids and b["id"] in ids:
        i, j = ids.index(a["id"]), ids.index(b["id"])
        return line_stations[i:j+1] if i <= j else list(reversed(line_stations[j:i+1]))
    return None


def _station_on_line(stn: dict, line_name: str) -> dict | None:
    target = stn["name"].lower()
    for s in _LINE_STATIONS.get(line_name, []):
        if s["name"].lower() == target:
            return s
    return None


# ── Multi-interchange BFS metro path-finder ───────────────────────────────────

def _build_station_graph() -> dict[str, set[str]]:
    """
    Build an adjacency graph where nodes are station IDs.
    Edges:
      - consecutive stations on the same line (board → next stop)
      - interchange edges between stations with the same physical name
        (e.g. bl_esplanade ↔ gl_esplanade, bl_noapara ↔ yl_noapara)
    """
    graph: dict[str, set[str]] = {s["id"]: set() for s in METRO_STATIONS}

    # Same-line adjacency
    for line_name, stations in _LINE_STATIONS.items():
        for i in range(len(stations) - 1):
            a, b = stations[i]["id"], stations[i+1]["id"]
            graph[a].add(b)
            graph[b].add(a)

    # Interchange adjacency (same physical station, different line IDs)
    for ic_variants in _INTERCHANGES.values():
        for (line_a, id_a) in ic_variants:
            for (line_b, id_b) in ic_variants:
                if id_a != id_b and id_a in graph and id_b in graph:
                    graph[id_a].add(id_b)
                    graph[id_b].add(id_a)

    # Also link any two stations with identical names on different lines
    for name_lower, stns in _STATION_BY_NAME.items():
        if len(stns) > 1:
            for i in range(len(stns)):
                for j in range(i+1, len(stns)):
                    a, b = stns[i]["id"], stns[j]["id"]
                    if a in graph and b in graph:
                        graph[a].add(b)
                        graph[b].add(a)

    return graph


# Build once at module load
_STATION_GRAPH: dict[str, set[str]] = _build_station_graph()


def _metro_path_bfs(
    src: dict,
    dst: dict,
    blocked_station_ids: set[str] | None = None,
) -> list[dict] | None:
    """
    BFS over the station graph to find the shortest (fewest stops) path
    from src to dst, respecting blocked stations (e.g. closed due to
    disruption events).

    Returns an ordered list of station dicts, or None if no path exists.
    Each station dict in the result carries the actual object from
    METRO_STATIONS (with correct id/line/lat/lon).
    """
    blocked = blocked_station_ids or set()
    if src["id"] in blocked or dst["id"] in blocked:
        return None

    queue: deque[list[str]] = deque([[src["id"]]])
    visited: set[str] = {src["id"]}

    while queue:
        path_ids = queue.popleft()
        current_id = path_ids[-1]

        if current_id == dst["id"]:
            return [_STATION_BY_ID[sid] for sid in path_ids]

        for neighbour_id in sorted(_STATION_GRAPH.get(current_id, [])):
            if neighbour_id in visited or neighbour_id in blocked:
                continue
            visited.add(neighbour_id)
            queue.append(path_ids + [neighbour_id])

    return None


def _path_cost_km(path: list[dict]) -> float:
    """Total haversine distance along a station path."""
    return sum(
        _haversine_km(path[i]["lat"], path[i]["lon"],
                      path[i+1]["lat"], path[i+1]["lon"])
        for i in range(len(path) - 1)
    )


# ── Graph helpers ─────────────────────────────────────────────────────────────

_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.openstreetmap.ru/api",
]
_GRAPH_PLACES = [
    "Kolkata, West Bengal, India",
    "Howrah, West Bengal, India",
    "Bidhannagar, West Bengal, India",
    "North Dum Dum, West Bengal, India",
    "Madhyamgram, West Bengal, India",
    "Barasat I, West Bengal, India",
    "Barasat II, West Bengal, India",
]

# In-process graph cache — each pkl is loaded exactly once per server lifetime
_GRAPH_CACHE: dict[str, object] = {}


def _load_or_download_graph(network_type: str, cache_path: str):
    if cache_path in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_path]
    if os.path.exists(cache_path):
        print(f"  [Route] Loading cached {network_type} graph ...")
        with open(cache_path, "rb") as f:
            g = pickle.load(f)
        print(f"  [Route] {network_type} graph loaded ({len(g.nodes):,} nodes)")
        _GRAPH_CACHE[cache_path] = g
        return g
    print(f"  [Route] Downloading OSM {network_type} network (one-time) ...")
    ox.settings.timeout = 180
    last_err = None
    for ep in _OVERPASS_ENDPOINTS:
        try:
            ox.settings.overpass_url = ep
            g = ox.graph_from_place(_GRAPH_PLACES, network_type=network_type, retain_all=False)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
            _GRAPH_CACHE[cache_path] = g
            return g
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download {network_type} graph: {last_err}")


def _geocode(place: str) -> tuple[float, float]:
    p = (place or "").strip()
    p = p.replace('\u202f', ' ').replace('\xa0', ' ')
    p_clean = re.sub(r'\s+', ' ', re.sub(r'[()]', ' ', p)).strip()

    for s in METRO_STATIONS:
        s_name_clean = re.sub(r'\s+', ' ', re.sub(r'[()]', ' ', s["name"]).replace('\u202f', ' ').replace('\xa0', ' ')).strip()
        if s_name_clean.lower() == p_clean.lower() or s["id"].lower() == p_clean.lower():
            return (s["lat"], s["lon"])

    for s in METRO_STATIONS:
        s_name_clean = re.sub(r'\s+', ' ', re.sub(r'[()]', ' ', s["name"]).replace('\u202f', ' ').replace('\xa0', ' ')).strip()
        if p_clean.lower() in s_name_clean.lower() or s_name_clean.lower() in p_clean.lower():
            return (s["lat"], s["lon"])

    from routing.route_engine import _geocode_with_context
    return _geocode_with_context(place)


# ── GeoJSON helpers ───────────────────────────────────────────────────────────

def _nodes_to_geojson(graph, nodes, label, dist_km, time_min):
    coords = [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in nodes if "x" in graph.nodes[n]]
    return {"type": "FeatureCollection", "features": [{"type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"label": label, "distance_km": dist_km, "travel_time_min": time_min}}]}


def _straight_geojson(points, label, dist_km, time_min):
    coords = [[lon, lat] for lat, lon in points]
    return {"type": "FeatureCollection", "features": [{"type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"label": label, "distance_km": dist_km, "travel_time_min": time_min}}]}


def _multi_segment_geojson(
    feeder1_coords: list[list[float]],   # [[lon, lat], ...]
    metro_segments: list[tuple[str, list[list[float]]]],  # [(line_color, [[lon, lat],...]), ...]
    feeder2_coords: list[list[float]],
    label: str,
    dist_km: float,
    time_min: int,
) -> dict:
    """
    Build a GeoJSON FeatureCollection with separate coloured features for
    each leg of a multimodal journey.

    feeder1 and feeder2 are rendered in grey (#607D8B).
    Each metro segment gets its line colour.
    """
    features = []

    if feeder1_coords:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": feeder1_coords},
            "properties": {"segment": "feeder_start", "color": "#607D8B",
                           "label": "Walk/Bike/Drive to metro"},
        })

    for seg_color, seg_coords in metro_segments:
        if seg_coords:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": seg_coords},
                "properties": {"segment": "metro", "color": seg_color,
                               "label": label},
            })

    if feeder2_coords:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": feeder2_coords},
            "properties": {"segment": "feeder_end", "color": "#607D8B",
                           "label": "Walk/Bike/Drive from metro"},
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "_meta": {"label": label, "distance_km": dist_km, "travel_time_min": time_min},
    }


def _multi_segment_geojson_mixed(
    walk_coords:    list[list[float]],
    metro_segments: list[tuple[str, list[list[float]]]],
    drive_coords:   list[list[float]],
    label:   str,
    dist_km: float,
    time_min: int,
) -> dict:
    """
    GeoJSON for walk → metro → drive (cab) routes.
    Walk leg: green dashed  (#34A853)
    Metro leg: line colour
    Drive leg: blue solid   (#1a73e8)
    """
    features = []
    if walk_coords:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": walk_coords},
            "properties": {"segment": "feeder_start", "color": "#34A853",
                           "label": "Walk to metro"},
        })
    for seg_color, seg_coords in metro_segments:
        if seg_coords:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": seg_coords},
                "properties": {"segment": "metro", "color": seg_color, "label": label},
            })
    if drive_coords:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": drive_coords},
            "properties": {"segment": "feeder_end", "color": "#1a73e8",
                           "label": "Cab / Drive from metro"},
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "_meta": {"label": label, "distance_km": dist_km, "travel_time_min": time_min},
    }
    from routing.route_engine import extract_road_names, _routes_are_distinct
    cache = {"drive": GRAPH_CACHE_PATH, "walk": _GRAPH_WALK, "bike": _GRAPH_BIKE}[network_type]

    graph = _load_or_download_graph(network_type, cache)
    sc = _geocode(source)
    dc = _geocode(destination)
    sn = ox.distance.nearest_nodes(graph, X=sc[1], Y=sc[0])
    dn = ox.distance.nearest_nodes(graph, X=dc[1], Y=dc[0])

    G = graph.copy(); raw = []; seen = []; dupes = 0
    for _ in range(MAX_ROUTES * 6):
        if len(raw) >= MAX_ROUTES or dupes >= MAX_CONSECUTIVE_DUPES:
            break
        r = ox.shortest_path(G, sn, dn, weight="length")
        if r is None:
            break
        es = frozenset(zip(r[:-1], r[1:]))
        new = all(_routes_are_distinct(es, p, MIN_ROUTE_DIVERGENCE) for p in seen)
        for u, v in zip(r[:-1], r[1:]):
            for k in G[u][v]:
                G[u][v][k]["length"] *= 5.0
        if not new:
            dupes += 1
            continue
        raw.append(r); seen.append(es); dupes = 0

    if not raw:
        raise ValueError(f"No {network_type} route found between '{source}' and '{destination}'.")

    out = []
    for idx, nl in enumerate(raw):
        roads = extract_road_names(graph, nl)
        dm = sum(graph.get_edge_data(u, v, 0).get("length", 0) for u, v in zip(nl[:-1], nl[1:]))
        dk = round(dm / 1000, 2)
        tm = round((dk / speed_kmh) * 60)
        coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in nl if "y" in graph.nodes[n]]
        out.append({
            "id": idx, "label": f"Route {idx+1}",
            "road_names": roads, "distance_km": dk, "travel_time_min": tm,
            "geojson": _nodes_to_geojson(graph, nl, f"Route {idx+1}", dk, tm),
            "node_ids": nl, "coords": coords, "mode": network_type,
        })

    return {"source": source, "destination": destination, "src_coords": list(sc),
            "dst_coords": list(dc), "routes": out, "mode": network_type}


def _road_route_single_best(
    source: str,
    destination: str,
    network_type: str,
    speed_kmh: float,
) -> dict | None:
    """
    Best single OSM road route. Returns dict with road_names, distance_km,
    travel_time_min, coords (lat/lon), geojson_coords (lon/lat).
    Returns None on failure or if distance < 50 m.
    """
    try:
        from routing.route_engine import extract_road_names
        cache = {"drive": GRAPH_CACHE_PATH, "walk": _GRAPH_WALK, "bike": _GRAPH_BIKE}[network_type]
        graph = _load_or_download_graph(network_type, cache)
        sc = _geocode(source)
        dc = _geocode(destination)
        if _haversine_km(sc[0], sc[1], dc[0], dc[1]) < 0.05:
            return None
        sn = ox.distance.nearest_nodes(graph, X=sc[1], Y=sc[0])
        dn = ox.distance.nearest_nodes(graph, X=dc[1], Y=dc[0])
        r = ox.shortest_path(graph, sn, dn, weight="length")
        if r is None:
            return None
        roads = extract_road_names(graph, r)
        dm = sum(graph.get_edge_data(u, v, 0).get("length", 0) for u, v in zip(r[:-1], r[1:]))
        dk = round(dm / 1000, 2)
        tm = round((dk / speed_kmh) * 60)
        coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in r if "y" in graph.nodes[n]]
        geojson_coords = [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in r if "x" in graph.nodes[n]]
        return {"road_names": roads, "distance_km": dk, "travel_time_min": tm,
                "coords": coords, "geojson_coords": geojson_coords}
    except Exception as e:
        print(f"  [Route] Feeder leg failed ({source} → {destination}): {e}")
        return None


# ── Disruption context helpers ────────────────────────────────────────────────

def _extract_blocked_stations(events: list[dict]) -> set[str]:
    """
    Given a list of extracted disruption events, return the set of station IDs
    that should be considered blocked/degraded.

    Criteria (conservative — only block when high confidence of closure):
      - event_type contains 'closure' or 'suspension'
      - AND severity is 'high'
      - AND location matches a known metro station name
    """
    blocked: set[str] = set()
    if not events:
        return blocked

    closure_types = {"closure", "suspension", "disruption", "shutdown"}

    for ev in events:
        et = (ev.get("event_type") or "").lower()
        sev = (ev.get("severity") or "").lower()
        loc = (ev.get("location") or "").lower()
        road = (ev.get("road_name") or "").lower()

        is_closure = any(ct in et for ct in closure_types)
        is_severe  = sev in ("high", "critical")

        if not (is_closure and is_severe):
            continue

        # Match location/road against station names
        for stn in METRO_STATIONS:
            stn_name = stn["name"].lower()
            if stn_name in loc or stn_name in road:
                blocked.add(stn["id"])
                print(f"  [Metro] Blocking station '{stn['name']}' due to disruption: {ev.get('reason','')[:60]}")

    return blocked


def _score_metro_path(
    path: list[dict],
    walk_src_km: float,
    walk_dst_km: float,
    events: list[dict],
) -> float:
    """
    Composite score for ranking alternative metro paths.
    Lower is better.

    Components:
      - Total journey distance (km)
      - Number of interchanges × 5-minute penalty
      - Disruption score from events near stations on the path
    """
    metro_km = _path_cost_km(path)
    total_km = walk_src_km + metro_km + walk_dst_km

    # Count line changes
    num_changes = sum(
        1 for i in range(1, len(path))
        if path[i]["line"] != path[i-1]["line"]
    )
    change_penalty = num_changes * (5.0 / 60.0)  # in hour-equivalent distance units

    # Disruption penalty: count high-severity events near each station
    disruption_penalty = 0.0
    if events:
        station_names = {s["name"].lower() for s in path}
        for ev in events:
            loc = (ev.get("location") or "").lower()
            road = (ev.get("road_name") or "").lower()
            sev = ev.get("severity", "low")
            weight = {"high": 2.0, "medium": 1.0, "low": 0.3}.get(sev, 0.3)
            if any(sn in loc or sn in road for sn in station_names):
                disruption_penalty += weight

    return total_km + change_penalty + disruption_penalty


# ── Metro line-segment splitting ──────────────────────────────────────────────

def _direction_on_line(board: dict, alight: dict, line_name: str) -> str:
    stations = _LINE_STATIONS.get(line_name, [])
    names = [s["name"].lower() for s in stations]
    bi = next((i for i, s in enumerate(names) if s == board["name"].lower()), None)
    ai = next((i for i, s in enumerate(names) if s == alight["name"].lower()), None)
    if bi is None or ai is None:
        return "a_to_b"
    return "a_to_b" if ai >= bi else "b_to_a"


def _split_path_by_line(path: list[dict]) -> list[tuple[str, list[dict]]]:
    """
    Split a multi-line BFS path into (line_name, [stations]) segments.
    Interchange transfer steps (same-name, different-ID nodes) are collapsed
    so each segment has a clean board→alight sequence on one line.
    """
    if not path:
        return []

    segments: list[tuple[str, list[dict]]] = []
    current_line  = path[0]["line"]
    current_group = [path[0]]

    for stn in path[1:]:
        if stn["line"] == current_line:
            current_group.append(stn)
        else:
            # Check if this is a transfer node (same name as last in current_group)
            if stn["name"].lower() == current_group[-1]["name"].lower():
                # This is a same-platform interchange transfer step — just switch line
                segments.append((current_line, current_group))
                current_line  = stn["line"]
                current_group = [stn]
            else:
                # Find the canonical interchange station on the new line
                ic_on_new = _station_on_line(current_group[-1], stn["line"])
                segments.append((current_line, current_group))
                current_line  = stn["line"]
                current_group = [ic_on_new] if ic_on_new else []
                if stn.get("id") != (ic_on_new or {}).get("id"):
                    current_group.append(stn)

    if current_group:
        segments.append((current_line, current_group))

    # Remove zero-stop segments (transfer-only nodes)
    return [(ln, stns) for ln, stns in segments if len(stns) >= 1]


# ── Metro route builder ───────────────────────────────────────────────────────

def _build_metro_route_for_pair(
    source: str,
    destination: str,
    sc: tuple[float, float],
    dc: tuple[float, float],
    src_stn: dict,
    dst_stn: dict,
    path: list[dict],
    walk_src_km: float,
    walk_dst_km: float,
    route_id: int,
    label: str,
) -> dict:
    """
    Build a complete metro route dict given a source→dest station pair and path.
    This is the pure metro version (feeder legs are simple walk estimates).
    """
    from routing.metro_timetable import next_train_for_journey, IST

    now = datetime.now(tz=IST)
    line_segments = _split_path_by_line(path)
    interchange   = len(line_segments) > 1

    walk_src_min = round((walk_src_km / 5.0) * 60)
    walk_dst_min = round((walk_dst_km / 5.0) * 60)

    metro_km  = _path_cost_km(path)
    metro_min = round((metro_km / 35.0) * 60) + 3 * len(line_segments)
    total_km  = round(walk_src_km + metro_km + walk_dst_km, 2)
    total_min = walk_src_min + metro_min + walk_dst_min

    segments = []
    segments.append({
        "type": "walk", "from": source, "to": src_stn["name"],
        "distance_km": round(walk_src_km, 2), "time_min": walk_src_min, "line": None,
    })

    for seg_line, seg_stns in line_segments:
        if not seg_stns:
            continue
        board  = seg_stns[0]
        alight = seg_stns[-1]
        seg_km = round(_path_cost_km(seg_stns), 2)
        seg_min = round((seg_km / 35.0) * 60) + 3
        direction = _direction_on_line(board, alight, seg_line)
        next_train = next_train_for_journey(
            src_station=board["name"], dst_station=alight["name"],
            direction=direction, line_name=seg_line, now=now,
        )
        segments.append({
            "type": "metro", "from": board["name"], "to": alight["name"],
            "distance_km": seg_km, "time_min": seg_min, "line": seg_line,
            "stations": [s["name"] for s in seg_stns],
            "num_stops": len(seg_stns) - 1,
            "next_train": next_train,
        })

    segments.append({
        "type": "walk", "from": dst_stn["name"], "to": destination,
        "distance_km": round(walk_dst_km, 2), "time_min": walk_dst_min, "line": None,
    })

    first_metro = next((s for s in segments if s["type"] == "metro"), None)
    next_train_top = first_metro["next_train"] if first_metro else None

    interchange_note = ""
    if interchange:
        ic_names = []
        for (_, seg_stns), (_, next_stns) in zip(line_segments[:-1], line_segments[1:]):
            ic_names.append(seg_stns[-1]["name"])
        interchange_note = "Change at " + ", ".join(ic_names)

    all_pts = (
        [(sc[0], sc[1]), (src_stn["lat"], src_stn["lon"])]
        + [(s["lat"], s["lon"]) for s in path]
        + [(dst_stn["lat"], dst_stn["lon"]), (dc[0], dc[1])]
    )
    geo = _straight_geojson(all_pts, label, total_km, total_min)

    return {
        "id": route_id, "label": label,
        "road_names": [s["name"] for s in path],
        "distance_km": total_km, "travel_time_min": total_min,
        "geojson": geo, "node_ids": [], "coords": list(all_pts),
        "mode": "metro", "segments": segments,
        "metro_stations": [s["name"] for s in path],
        "walk_src_min": walk_src_min, "walk_dst_min": walk_dst_min,
        "metro_min": metro_min, "interchange": interchange,
        "interchange_note": interchange_note,
        "next_train": next_train_top,
        "num_interchanges": len(line_segments) - 1,
    }


def _metro_route(source: str, destination: str, events: list[dict] | None = None) -> dict:
    """
    Find up to MAX_ROUTES distinct metro routes between source and destination.

    Uses BFS over the full station graph to handle any number of interchanges.
    Disruption events are used to:
      1. Block stations reported as closed/suspended (high severity)
      2. Score alternative paths (paths avoiding disrupted stations rank higher)
    """
    sc = _geocode(source)
    dc = _geocode(destination)
    ev = events or []

    # Identify blocked stations from disruption context
    blocked = _extract_blocked_stations(ev)

    # Build candidate boarding/alighting stations
    # Use nearest 3 per line as candidates to generate route diversity
    all_lines = list(_LINE_STATIONS.keys())

    def _top_n_on_line(lat: float, lon: float, line: str, n: int = 3) -> list[dict]:
        stns = sorted(_LINE_STATIONS[line],
                      key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))
        return [s for s in stns[:n] if s["id"] not in blocked]

    src_candidates: list[dict] = []
    dst_candidates: list[dict] = []
    for ln in all_lines:
        src_candidates.extend(_top_n_on_line(sc[0], sc[1], ln, 2))
        dst_candidates.extend(_top_n_on_line(dc[0], dc[1], ln, 2))

    # Deduplicate preserving order
    def _uniq_by_id(lst: list[dict]) -> list[dict]:
        seen: set[str] = set(); out = []
        for x in lst:
            if x["id"] not in seen:
                seen.add(x["id"]); out.append(x)
        return out

    src_candidates = _uniq_by_id(src_candidates)
    dst_candidates = _uniq_by_id(dst_candidates)

    # Try all src×dst combos, find BFS paths, score and rank them
    scored: list[tuple[float, dict, dict, list[dict], float, float]] = []
    seen_path_sigs: set[tuple[str, ...]] = set()

    for s in src_candidates:
        if s["id"] in blocked:
            continue
        for d in dst_candidates:
            if d["id"] in blocked or s["id"] == d["id"]:
                continue
            path = _metro_path_bfs(s, d, blocked_station_ids=blocked)
            if not path or len(path) < 2:
                continue
            sig = tuple(st["id"] for st in path)
            if sig in seen_path_sigs:
                continue
            seen_path_sigs.add(sig)
            wk_src = _haversine_km(sc[0], sc[1], s["lat"], s["lon"])
            wk_dst = _haversine_km(dc[0], dc[1], d["lat"], d["lon"])
            score  = _score_metro_path(path, wk_src, wk_dst, ev)
            scored.append((score, s, d, path, wk_src, wk_dst))

    if not scored:
        # No metro connection — fallback walk route
        total_km  = round(_haversine_km(sc[0], sc[1], dc[0], dc[1]), 2)
        total_min = round((total_km / 5.0) * 60)
        geo = _straight_geojson([(sc[0], sc[1]), (dc[0], dc[1])], "Walk", total_km, total_min)
        return {
            "source": source, "destination": destination,
            "src_coords": list(sc), "dst_coords": list(dc),
            "mode": "metro",
            "routes": [{
                "id": 0, "label": "Walk (no metro)", "road_names": [],
                "distance_km": total_km, "travel_time_min": total_min,
                "geojson": geo, "coords": [(sc[0], sc[1]), (dc[0], dc[1])],
                "mode": "metro", "segments": [],
                "metro_note": "No metro connection found. Showing walk route.",
            }],
        }

    # Sort by score, take top MAX_ROUTES
    scored.sort(key=lambda x: x[0])
    top = scored[:MAX_ROUTES]

    routes_out = []
    for rank, (score, src_stn, dst_stn, path, wk_src, wk_dst) in enumerate(top):
        n_ic = sum(1 for i in range(1, len(path)) if path[i]["line"] != path[i-1]["line"])
        if rank == 0:
            lbl = "Best Metro Route"
        elif n_ic == 0:
            lbl = f"Direct · {path[0]['line'].title()} Line"
        else:
            lbl = f"Via {n_ic} interchange{'s' if n_ic > 1 else ''}"
        r = _build_metro_route_for_pair(
            source, destination, sc, dc,
            src_stn, dst_stn, path, wk_src, wk_dst,
            route_id=rank, label=lbl,
        )
        routes_out.append(r)

    return {
        "source": source, "destination": destination,
        "src_coords": list(sc), "dst_coords": list(dc),
        "mode": "metro", "routes": routes_out,
    }


# ── Multimodal combo route ────────────────────────────────────────────────────

def _build_combo_route(
    source: str,
    destination: str,
    sc: tuple[float, float],
    dc: tuple[float, float],
    src_stn: dict,
    dst_stn: dict,
    path: list[dict],
    wk_src_km: float,
    wk_dst_km: float,
    feeder_mode: str,
    speed_kmh: float,
    route_id: int,
    label: str,
    events: list[dict],
) -> dict:
    """
    Build one multimodal route from a given boarding/alighting station pair.
    Feeder legs are OSM-routed; metro leg is station-list + timetable.
    """
    from routing.metro_timetable import next_train_for_journey, IST

    now = datetime.now(tz=IST)
    combo_mode = f"metro+{feeder_mode}"
    line_segments = _split_path_by_line(path)
    interchange   = len(line_segments) > 1

    # ── OSM-route both feeder legs ────────────────────────────────────────────
    leg1 = _road_route_single_best(source,       src_stn["name"], feeder_mode, speed_kmh)
    leg2 = _road_route_single_best(dst_stn["name"], destination,  feeder_mode, speed_kmh)

    # Feeder leg 1
    if leg1:
        leg1_dk, leg1_tm = leg1["distance_km"], leg1["travel_time_min"]
        leg1_roads = leg1["road_names"]
        leg1_geo   = leg1["geojson_coords"]
        leg1_coords = leg1["coords"]
    else:
        leg1_dk = round(wk_src_km, 2)
        leg1_tm = round((leg1_dk / speed_kmh) * 60)
        leg1_roads = []
        leg1_geo   = [[sc[1], sc[0]], [src_stn["lon"], src_stn["lat"]]]
        leg1_coords = [(sc[0], sc[1]), (src_stn["lat"], src_stn["lon"])]

    # Feeder leg 2
    if leg2:
        leg2_dk, leg2_tm = leg2["distance_km"], leg2["travel_time_min"]
        leg2_roads = leg2["road_names"]
        leg2_geo   = leg2["geojson_coords"]
        leg2_coords = leg2["coords"]
    else:
        leg2_dk = round(wk_dst_km, 2)
        leg2_tm = round((leg2_dk / speed_kmh) * 60)
        leg2_roads = []
        leg2_geo   = [[dst_stn["lon"], dst_stn["lat"]], [dc[1], dc[0]]]
        leg2_coords = [(dst_stn["lat"], dst_stn["lon"]), (dc[0], dc[1])]

    # ── Build metro segments ──────────────────────────────────────────────────
    metro_min_total = 0
    metro_km_total  = 0.0
    metro_geojson_segs: list[tuple[str, list[list[float]]]] = []
    journey_segs: list[dict] = []

    journey_segs.append({
        "type": feeder_mode, "mode": feeder_mode,
        "from": source, "to": src_stn["name"],
        "distance_km": leg1_dk, "time_min": leg1_tm, "line": None,
    })

    # Track running clock: user departs origin at `now`, arrives at boarding
    # station after leg1_tm minutes.  Subsequent interchanges accumulate time.
    # We use this to find the *catchable* train rather than the next train
    # departing right now — which may have already left by the time the user
    # arrives on the platform.
    from datetime import timedelta

    accumulated_min = leg1_tm          # minutes elapsed since departure
    wait_min_total  = 0                # total waiting time added for missed trains

    for seg_idx, (seg_line, seg_stns) in enumerate(line_segments):
        if not seg_stns:
            continue
        board  = seg_stns[0]
        alight = seg_stns[-1]
        seg_km = round(_path_cost_km(seg_stns), 2)
        seg_min = round((seg_km / 35.0) * 60) + 3
        metro_km_total  += seg_km

        # Compute when the user physically arrives at this boarding station
        arrival_at_board = now + timedelta(minutes=accumulated_min)

        direction = _direction_on_line(board, alight, seg_line)
        next_train = next_train_for_journey(
            src_station=board["name"], dst_station=alight["name"],
            direction=direction, line_name=seg_line, now=arrival_at_board,
        )

        # Calculate any extra wait time if the next train is not immediate
        # (minutes_away is measured from arrival_at_board, so it IS the wait)
        wait_for_train = int(next_train["minutes_away"]) if next_train else 0

        metro_min_total += seg_min + wait_for_train
        wait_min_total  += wait_for_train

        # Advance accumulated clock: wait + in-motion metro time
        # (interchange walk ~3 min is already baked into seg_min offset)
        accumulated_min += wait_for_train + seg_min

        journey_segs.append({
            "type": "metro", "from": board["name"], "to": alight["name"],
            "distance_km": seg_km, "time_min": seg_min, "line": seg_line,
            "stations": [s["name"] for s in seg_stns],
            "num_stops": len(seg_stns) - 1,
            "next_train": next_train,
            "wait_min": wait_for_train,
        })

        seg_color = LINE_COLORS.get(seg_line, "#7c4dff")
        seg_geo_pts = [[s["lon"], s["lat"]] for s in seg_stns]
        metro_geojson_segs.append((seg_color, seg_geo_pts))

    journey_segs.append({
        "type": feeder_mode, "mode": feeder_mode,
        "from": dst_stn["name"], "to": destination,
        "distance_km": leg2_dk, "time_min": leg2_tm, "line": None,
    })

    total_km  = round(leg1_dk + metro_km_total + leg2_dk, 2)
    total_min = leg1_tm + metro_min_total + leg2_tm

    # ── Stitch GeoJSON with per-segment colours ───────────────────────────────
    # De-duplicate boundary points between segments
    def _join(a: list, b: list) -> list:
        if not a:
            return b
        if not b:
            return a
        if abs(a[-1][0] - b[0][0]) < 1e-7 and abs(a[-1][1] - b[0][1]) < 1e-7:
            return a + b[1:]
        return a + b

    # Build flat coord list for legacy `coords` (lat/lon) field
    all_coords: list[tuple[float, float]] = list(leg1_coords)
    for _, seg_pts in metro_geojson_segs:
        for pt in seg_pts:
            all_coords.append((pt[1], pt[0]))
    all_coords.extend(leg2_coords)

    geo = _multi_segment_geojson(
        feeder1_coords=leg1_geo,
        metro_segments=metro_geojson_segs,
        feeder2_coords=leg2_geo,
        label=label,
        dist_km=total_km,
        time_min=total_min,
    )

    # ── Disruption context: flag this route if events affect it ──────────────
    station_names_on_path = {s["name"].lower() for s in path}
    all_road_names = list(set(leg1_roads + [s["name"] for s in path] + leg2_roads))
    disruption_flags: list[str] = []
    for ev in events:
        loc = (ev.get("location") or "").lower()
        sev = ev.get("severity", "low")
        if any(sn in loc for sn in station_names_on_path) and sev in ("high", "medium"):
            disruption_flags.append(f"{ev.get('event_type','event')} at {ev.get('location','?')}")

    first_metro_seg = next((s for s in journey_segs if s["type"] == "metro"), None)
    next_train_top  = first_metro_seg["next_train"] if first_metro_seg else None

    interchange_note = ""
    if interchange:
        ic_names = [seg_stns[-1]["name"] for _, seg_stns in line_segments[:-1] if seg_stns]
        interchange_note = "Change at " + ", ".join(ic_names)

    return {
        "id": route_id, "label": label,
        "road_names": all_road_names,
        "distance_km": total_km, "travel_time_min": total_min,
        "geojson": geo, "node_ids": [], "coords": all_coords,
        "mode": combo_mode, "segments": journey_segs,
        "metro_stations": [s["name"] for s in path],
        "walk_src_min": leg1_tm, "walk_dst_min": leg2_tm,
        "metro_min": metro_min_total,
        "interchange": interchange, "interchange_note": interchange_note,
        "next_train": next_train_top,
        "num_interchanges": len(line_segments) - 1,
        "disruption_flags": disruption_flags,
    }


def _build_mixed_combo_route(
    source: str,
    destination: str,
    sc: tuple[float, float],
    dc: tuple[float, float],
    src_stn: dict,
    dst_stn: dict,
    path: list[dict],
    wk_src_km: float,
    wk_dst_km: float,
    route_id: int,
    label: str,
    events: list[dict],
) -> dict:
    """
    Build one walk → metro → drive/cab route.

    Leg 1 (origin → boarding station): walk at 5 km/h
    Metro leg(s): timetable-aware, same as _build_combo_route
    Leg 2 (alighting station → destination): drive at 30 km/h (cab estimate)
    """
    from routing.metro_timetable import next_train_for_journey, IST
    from datetime import timedelta

    WALK_SPEED  = 5.0
    DRIVE_SPEED = 30.0
    now = datetime.now(tz=IST)
    combo_mode  = "metro+walk+drive"
    line_segments = _split_path_by_line(path)
    interchange   = len(line_segments) > 1

    # ── OSM-route feeder legs with their respective modes ─────────────────────
    leg1 = _road_route_single_best(source,           src_stn["name"], "walk",  WALK_SPEED)
    leg2 = _road_route_single_best(dst_stn["name"],  destination,     "drive", DRIVE_SPEED)

    if leg1:
        leg1_dk, leg1_tm = leg1["distance_km"], leg1["travel_time_min"]
        leg1_roads = leg1["road_names"]
        leg1_geo   = leg1["geojson_coords"]
        leg1_coords = leg1["coords"]
    else:
        leg1_dk = round(wk_src_km, 2)
        leg1_tm = round((leg1_dk / WALK_SPEED) * 60)
        leg1_roads  = []
        leg1_geo    = [[sc[1], sc[0]], [src_stn["lon"], src_stn["lat"]]]
        leg1_coords = [(sc[0], sc[1]), (src_stn["lat"], src_stn["lon"])]

    if leg2:
        leg2_dk, leg2_tm = leg2["distance_km"], leg2["travel_time_min"]
        leg2_roads = leg2["road_names"]
        leg2_geo   = leg2["geojson_coords"]
        leg2_coords = leg2["coords"]
    else:
        leg2_dk = round(wk_dst_km, 2)
        leg2_tm = round((leg2_dk / DRIVE_SPEED) * 60)
        leg2_roads  = []
        leg2_geo    = [[dst_stn["lon"], dst_stn["lat"]], [dc[1], dc[0]]]
        leg2_coords = [(dst_stn["lat"], dst_stn["lon"]), (dc[0], dc[1])]

    # ── Metro segments with time-aware next-train lookup ──────────────────────
    metro_min_total = 0
    metro_km_total  = 0.0
    metro_geojson_segs: list[tuple[str, list[list[float]]]] = []
    journey_segs: list[dict] = []

    journey_segs.append({
        "type": "walk", "mode": "walk",
        "from": source, "to": src_stn["name"],
        "distance_km": leg1_dk, "time_min": leg1_tm, "line": None,
    })

    accumulated_min = leg1_tm
    wait_min_total  = 0

    for seg_line, seg_stns in line_segments:
        if not seg_stns:
            continue
        board  = seg_stns[0]
        alight = seg_stns[-1]
        seg_km  = round(_path_cost_km(seg_stns), 2)
        seg_min = round((seg_km / 35.0) * 60) + 3
        metro_km_total += seg_km

        arrival_at_board = now + timedelta(minutes=accumulated_min)
        direction = _direction_on_line(board, alight, seg_line)
        next_train = next_train_for_journey(
            src_station=board["name"], dst_station=alight["name"],
            direction=direction, line_name=seg_line, now=arrival_at_board,
        )
        wait_for_train   = int(next_train["minutes_away"]) if next_train else 0
        metro_min_total += seg_min + wait_for_train
        wait_min_total  += wait_for_train
        accumulated_min += wait_for_train + seg_min

        journey_segs.append({
            "type": "metro", "from": board["name"], "to": alight["name"],
            "distance_km": seg_km, "time_min": seg_min, "line": seg_line,
            "stations": [s["name"] for s in seg_stns],
            "num_stops": len(seg_stns) - 1,
            "next_train": next_train,
            "wait_min": wait_for_train,
        })
        seg_color = LINE_COLORS.get(seg_line, "#7c4dff")
        metro_geojson_segs.append((seg_color, [[s["lon"], s["lat"]] for s in seg_stns]))

    journey_segs.append({
        "type": "drive", "mode": "drive",
        "from": dst_stn["name"], "to": destination,
        "distance_km": leg2_dk, "time_min": leg2_tm, "line": None,
        "cab_note": "Cab / auto-rickshaw recommended",
    })

    total_km  = round(leg1_dk + metro_km_total + leg2_dk, 2)
    total_min = leg1_tm + metro_min_total + leg2_tm

    all_coords: list[tuple[float, float]] = list(leg1_coords)
    for _, seg_pts in metro_geojson_segs:
        for pt in seg_pts:
            all_coords.append((pt[1], pt[0]))
    all_coords.extend(leg2_coords)

    geo = _multi_segment_geojson_mixed(
        walk_coords=leg1_geo,
        metro_segments=metro_geojson_segs,
        drive_coords=leg2_geo,
        label=label, dist_km=total_km, time_min=total_min,
    )

    station_names_on_path = {s["name"].lower() for s in path}
    all_road_names = list(set(leg1_roads + [s["name"] for s in path] + leg2_roads))
    disruption_flags: list[str] = []
    for ev in events:
        loc = (ev.get("location") or "").lower()
        sev = ev.get("severity", "low")
        if any(sn in loc for sn in station_names_on_path) and sev in ("high", "medium"):
            disruption_flags.append(f"{ev.get('event_type','event')} at {ev.get('location','?')}")

    first_metro_seg = next((s for s in journey_segs if s["type"] == "metro"), None)
    next_train_top  = first_metro_seg["next_train"] if first_metro_seg else None

    interchange_note = ""
    if interchange:
        ic_names = [stns[-1]["name"] for _, stns in line_segments[:-1] if stns]
        interchange_note = "Change at " + ", ".join(ic_names)

    return {
        "id": route_id, "label": label,
        "road_names": all_road_names,
        "distance_km": total_km, "travel_time_min": total_min,
        "geojson": geo, "node_ids": [], "coords": all_coords,
        "mode": combo_mode, "segments": journey_segs,
        "metro_stations": [s["name"] for s in path],
        "walk_src_min": leg1_tm, "drive_dst_min": leg2_tm,
        "metro_min": metro_min_total,
        "interchange": interchange, "interchange_note": interchange_note,
        "next_train": next_train_top,
        "num_interchanges": len(line_segments) - 1,
        "disruption_flags": disruption_flags,
    }


def _metro_mixed_route(
    source: str,
    destination: str,
    events: list[dict] | None = None,
) -> dict:
    """
    Generate walk → metro → drive/cab route alternatives.

    Preference order: maximise metro coverage → minimise drive → minimise walk.

    Scoring weights:
      - Metro km covered    : −0.5 × metro_km  (reward, longer metro = lower score)
      - Drive/cab leg       : +4.0 × drive_h   (heavy penalty — drive is last resort)
      - Walk leg            : +2.0 × walk_h    (moderate penalty — preferred over drive)
      - Train wait          : +1.0 × wait_h    (standard penalty)
      - Interchange penalty : baked into _score_metro_path (+5 min per change)
      - Disruption penalty  : baked into _score_metro_path

    Boarding walk cap is relaxed to 2 km so more boarding stations are
    considered — this surfaces options with longer metro legs even when the
    nearest station is a bit further away.
    """
    WALK_SPEED  = 5.0
    DRIVE_SPEED = 30.0

    # Weights — metro preference first, then drive penalty, then walk penalty
    METRO_REWARD   = 0.5   # subtracted per metro km (encourages longer metro leg)
    DRIVE_PENALTY  = 4.0   # multiplied by drive hours (heavy — cab is last resort)
    WALK_PENALTY   = 2.0   # multiplied by walk hours (moderate — prefer short walks)

    # Boarding walk cap: slightly relaxed so we find stations with longer metro legs
    MAX_SRC_KM       = 1.5
    MAX_SRC_KM_RELAX = 2.0
    MAX_SRC_KM_FINAL = 3.0

    ev = events or []
    sc = _geocode(source)
    dc = _geocode(destination)
    blocked = _extract_blocked_stations(ev)
    all_lines = list(_LINE_STATIONS.keys())

    def _top_n_on_line(lat: float, lon: float, line: str, n: int = 3) -> list[dict]:
        stns = sorted(_LINE_STATIONS[line],
                      key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))
        return [s for s in stns[:n] if s["id"] not in blocked]

    def _uniq_by_id(lst: list[dict]) -> list[dict]:
        seen: set[str] = set(); out = []
        for x in lst:
            if x["id"] not in seen:
                seen.add(x["id"]); out.append(x)
        return out

    src_candidates = _uniq_by_id([s for ln in all_lines for s in _top_n_on_line(sc[0], sc[1], ln, 2)])
    dst_candidates = _uniq_by_id([s for ln in all_lines for s in _top_n_on_line(dc[0], dc[1], ln, 2)])

    def _score_mixed(src_cap_km: float) -> list[tuple[float, dict, dict, list[dict], float, float]]:
        scored_inner: list[tuple[float, dict, dict, list[dict], float, float]] = []
        seen_sigs: set[tuple[str, ...]] = set()
        for s_stn in src_candidates:
            if s_stn["id"] in blocked:
                continue
            wk_src = _haversine_km(sc[0], sc[1], s_stn["lat"], s_stn["lon"])
            if wk_src > src_cap_km:
                continue
            for d_stn in dst_candidates:
                if d_stn["id"] in blocked or s_stn["id"] == d_stn["id"]:
                    continue
                path = _metro_path_bfs(s_stn, d_stn, blocked_station_ids=blocked)
                if not path or len(path) < 2:
                    continue
                sig = tuple(st["id"] for st in path)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                wk_dst = _haversine_km(dc[0], dc[1], d_stn["lat"], d_stn["lon"])
                metro_km = _path_cost_km(path)

                # Base score: interchange + disruption penalties (from _score_metro_path)
                score = _score_metro_path(path, wk_src, wk_dst, ev)

                # Metro reward: subtract for longer metro coverage (max metro preference)
                score -= METRO_REWARD * metro_km

                # Drive penalty: heavily penalise the cab leg (hours × 4)
                score += DRIVE_PENALTY * (wk_dst / DRIVE_SPEED)

                # Walk penalty: moderately penalise boarding walk (hours × 2)
                score += WALK_PENALTY * (wk_src / WALK_SPEED)

                # Train-wait penalty
                from routing.metro_timetable import next_train_for_journey, IST
                from datetime import datetime as _dt, timedelta as _td
                _now_s = _dt.now(tz=IST)
                arrive  = _now_s + _td(minutes=(wk_src / WALK_SPEED) * 60)
                _fl = path[0]["line"] if path else None
                _dir = _direction_on_line(path[0], path[-1], _fl) if _fl else None
                if _fl and _dir:
                    _nt = next_train_for_journey(
                        src_station=s_stn["name"], dst_station=d_stn["name"],
                        direction=_dir, line_name=_fl, now=arrive,
                    )
                    score += (float(_nt["minutes_away"]) / 60.0) if _nt else 1.0
                scored_inner.append((score, s_stn, d_stn, path, wk_src, wk_dst))
        return scored_inner

    scored = _score_mixed(MAX_SRC_KM)
    if not scored:
        scored = _score_mixed(MAX_SRC_KM_RELAX)
    if not scored:
        scored = _score_mixed(MAX_SRC_KM_FINAL)

    if not scored:
        print("  [Metro] No metro path found for walk+metro+drive")
        return {
            "source": source, "destination": destination,
            "src_coords": list(sc), "dst_coords": list(dc),
            "mode": "metro+walk+drive", "routes": [],
            "metro_note": (
                "No metro station reachable by walking from your origin. "
                "Try Metro+Drive or a pure drive route."
            ),
        }

    scored.sort(key=lambda x: x[0])
    top = scored[:MAX_ROUTES]

    routes_out = []
    for rank, (score, s_stn, d_stn, path, wk_src, wk_dst) in enumerate(top):
        n_ic = sum(1 for i in range(1, len(path)) if path[i]["line"] != path[i-1]["line"])
        metro_km = round(_path_cost_km(path), 2)
        if rank == 0:
            lbl = "Best · Walk+Metro+Cab"
        elif n_ic == 0:
            lbl = f"Direct {path[0]['line'].title()} · {metro_km} km metro"
        else:
            lbl = f"Via {n_ic} interchange{'s' if n_ic > 1 else ''} · {metro_km} km metro"

        r = _build_mixed_combo_route(
            source=source, destination=destination, sc=sc, dc=dc,
            src_stn=s_stn, dst_stn=d_stn, path=path,
            wk_src_km=wk_src, wk_dst_km=wk_dst,
            route_id=rank, label=lbl, events=ev,
        )
        routes_out.append(r)

    return {
        "source": source, "destination": destination,
        "src_coords": list(sc), "dst_coords": list(dc),
        "mode": "metro+walk+drive", "routes": routes_out,
    }



def _metro_combo_route(
    source: str,
    destination: str,
    feeder_mode: str,
    events: list[dict] | None = None,
) -> dict:
    """
    Generate multiple multimodal (metro + feeder) route alternatives.

    Algorithm:
      1. Identify blocked stations from disruption events.
      2. For each of the top-N candidate boarding stations near source and
         top-N alighting stations near destination, run BFS to find a metro
         path. Score each combination.
      3. Pick the top MAX_ROUTES distinct combinations.
      4. For each selected combination, OSM-route the feeder legs.
      5. Return all routes ranked by composite score (time + disruption).
    """
    if feeder_mode not in ("walk", "bike", "drive"):
        raise ValueError("Metro combos only support 'walk', 'bike', or 'drive'.")

    combo_mode = f"metro+{feeder_mode}"
    speed_kmh  = MODE_SPEED_KMH[feeder_mode]
    ev         = events or []

    sc = _geocode(source)
    dc = _geocode(destination)

    blocked = _extract_blocked_stations(ev)

    # Candidate boarding/alighting stations — top 3 per line
    all_lines = list(_LINE_STATIONS.keys())

    def _top_n_on_line(lat: float, lon: float, line: str, n: int = 3) -> list[dict]:
        stns = sorted(_LINE_STATIONS[line],
                      key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))
        return [s for s in stns[:n] if s["id"] not in blocked]

    def _uniq_by_id(lst: list[dict]) -> list[dict]:
        seen: set[str] = set(); out = []
        for x in lst:
            if x["id"] not in seen:
                seen.add(x["id"]); out.append(x)
        return out

    src_candidates = _uniq_by_id(
        [s for ln in all_lines for s in _top_n_on_line(sc[0], sc[1], ln, 2)]
    )
    dst_candidates = _uniq_by_id(
        [s for ln in all_lines for s in _top_n_on_line(dc[0], dc[1], ln, 2)]
    )

    # ── Walk-distance cap ─────────────────────────────────────────────────────
    # src cap (boarding walk): keep tight — user shouldn't have to walk far to
    #   board, and there are usually alternatives nearby.
    # dst cap (exit walk): keep looser — the destination may simply not be near
    #   any metro station (e.g. Jadavpur is ~2.5 km from Tollygunge). We still
    #   want the metro+walk option; we just want to pick the closest exit station.
    #
    # For bike/drive feeder modes the caps are higher because those modes cover
    # distance much faster than walking.
    if feeder_mode == "walk":
        MAX_SRC_KM         = 1.0   # boarding walk cap (strict)
        MAX_SRC_KM_RELAX   = 1.5
        MAX_SRC_KM_FINAL   = 2.0
        MAX_DST_KM         = 3.0   # exit walk cap (relaxed — metro coverage gap)
    elif feeder_mode == "bike":
        MAX_SRC_KM         = 3.0
        MAX_SRC_KM_RELAX   = 5.0
        MAX_SRC_KM_FINAL   = 8.0
        MAX_DST_KM         = 8.0
    else:  # drive — no cap
        MAX_SRC_KM         = float("inf")
        MAX_SRC_KM_RELAX   = float("inf")
        MAX_SRC_KM_FINAL   = float("inf")
        MAX_DST_KM         = float("inf")

    def _score_candidates(src_cap_km: float) -> list[tuple[float, dict, dict, list[dict], float, float]]:
        scored_inner: list[tuple[float, dict, dict, list[dict], float, float]] = []
        seen_sigs_inner: set[tuple[str, ...]] = set()
        for s_stn in src_candidates:
            if s_stn["id"] in blocked:
                continue
            wk_src = _haversine_km(sc[0], sc[1], s_stn["lat"], s_stn["lon"])
            if wk_src > src_cap_km:
                continue
            for d_stn in dst_candidates:
                if d_stn["id"] in blocked or s_stn["id"] == d_stn["id"]:
                    continue
                wk_dst = _haversine_km(dc[0], dc[1], d_stn["lat"], d_stn["lon"])
                if wk_dst > MAX_DST_KM:
                    continue
                path = _metro_path_bfs(s_stn, d_stn, blocked_station_ids=blocked)
                if not path or len(path) < 2:
                    continue
                sig = tuple(st["id"] for st in path)
                if sig in seen_sigs_inner:
                    continue
                seen_sigs_inner.add(sig)
                score = _score_metro_path(path, wk_src, wk_dst, ev)
                # Add feeder-mode time penalty: faster feeder = lower score
                feeder_km = wk_src + wk_dst
                feeder_time_h = feeder_km / speed_kmh
                score += feeder_time_h
                # Add estimated train-wait penalty so routes where the user
                # will miss the current train rank below ones where they
                # arrive in time.  We estimate arrival time at the boarding
                # station as (walk distance / speed) and check the timetable.
                from routing.metro_timetable import next_train_for_journey, IST
                from datetime import datetime as _dt, timedelta as _td
                _now_score = _dt.now(tz=IST)
                walk_min_to_board = (wk_src / speed_kmh) * 60
                arrive_at_board   = _now_score + _td(minutes=walk_min_to_board)
                _first_line = path[0]["line"] if path else None
                _direction  = _direction_on_line(path[0], path[-1], _first_line) if _first_line else None
                if _first_line and _direction:
                    _nt = next_train_for_journey(
                        src_station=s_stn["name"], dst_station=d_stn["name"],
                        direction=_direction, line_name=_first_line,
                        now=arrive_at_board,
                    )
                    # minutes_away is relative to arrive_at_board, so it's
                    # pure platform-wait time.  Convert to hours for score units.
                    wait_h = (float(_nt["minutes_away"]) / 60.0) if _nt else 1.0
                    score += wait_h
                scored_inner.append((score, s_stn, d_stn, path, wk_src, wk_dst))
        return scored_inner

    scored = _score_candidates(MAX_SRC_KM)
    if not scored:
        scored = _score_candidates(MAX_SRC_KM_RELAX)
    if not scored:
        scored = _score_candidates(MAX_SRC_KM_FINAL)

    if not scored:
        # No metro reachable within reasonable walking distance.
        # Return an empty routes list with a flag rather than a misleading
        # pure-walk route in the metro+walk section.  The API layer will
        # handle this gracefully (empty routes → frontend shows "no route").
        print(f"  [Metro] No metro path found within walk cap for {feeder_mode} feeder")
        return {
            "source": source, "destination": destination,
            "src_coords": list(sc), "dst_coords": list(dc),
            "mode": combo_mode, "routes": [],
            "metro_note": (
                f"No metro station reachable by {feeder_mode} from your origin. "
                "Try a drive or bike feeder, or use a pure walk/drive route."
            ),
        }

    scored.sort(key=lambda x: x[0])
    top = scored[:MAX_ROUTES]

    routes_out = []
    for rank, (score, s_stn, d_stn, path, wk_src, wk_dst) in enumerate(top):
        n_ic = sum(1 for i in range(1, len(path)) if path[i]["line"] != path[i-1]["line"])
        if rank == 0:
            lbl = f"Best · Metro+{feeder_mode.title()}"
        elif n_ic == 0:
            lbl = f"Direct {path[0]['line'].title()} · {feeder_mode.title()} feeder"
        else:
            lbl = f"Via {n_ic} interchange{'s' if n_ic > 1 else ''} · {feeder_mode.title()} feeder"

        r = _build_combo_route(
            source=source, destination=destination, sc=sc, dc=dc,
            src_stn=s_stn, dst_stn=d_stn, path=path,
            wk_src_km=wk_src, wk_dst_km=wk_dst,
            feeder_mode=feeder_mode, speed_kmh=speed_kmh,
            route_id=rank, label=lbl, events=ev,
        )
        routes_out.append(r)

    return {
        "source": source, "destination": destination,
        "src_coords": list(sc), "dst_coords": list(dc),
        "mode": combo_mode, "routes": routes_out,
    }


# ── Bus route ─────────────────────────────────────────────────────────────────

def _resolve_bus_stop(place: str) -> dict | None:
    """
    Resolve a place name string to the nearest bus stop dict.

    Strategy:
      1. Exact / substring match against stop names in stops.csv
      2. Geocode the place via the existing _geocode() helper and snap to the
         nearest stop by haversine distance (≤ 5 km cap).

    Returns {stop_id, name, lat, lon} or None.
    """
    from transit.bus_graph import (
        find_nearest_stop_by_name,
        find_nearest_stop,
    )

    # Try name-based match first (fast, no network)
    hit = find_nearest_stop_by_name(place)
    if hit:
        return hit

    # Fall back to geocode → nearest stop
    try:
        lat, lon = _geocode(place)
        return find_nearest_stop(lat, lon, max_distance_km=5.0)
    except Exception as e:
        print(f"  [Bus] Geocode failed for '{place}': {e}")
        return None


def _bus_route(source: str, destination: str) -> dict:
    """
    Find bus routes between source and destination.

    Uses BFS over the Kolkata bus stop graph (transit/bus_engine.py) to find
    the shortest-stop path, then converts stop IDs → lat/lon via bus_overlay
    and packages the result in the same route dict format used by all other
    modes so the rest of the pipeline (scoring, GeoJSON rendering) works
    without changes.

    Returns a dict with keys: source, destination, src_coords, dst_coords,
    mode, routes — matching the shape returned by _road_routes / _metro_route.
    """
    from transit.bus_engine import find_bus_path
    from transit.bus_overlay import get_path_coordinates
    from transit.bus_graph import (
        load_stops,
        get_routes_through_stops,
    )

    # ── Resolve source / destination to bus stops ─────────────────────────────
    src_stop = _resolve_bus_stop(source)
    dst_stop = _resolve_bus_stop(destination)

    if src_stop is None:
        raise ValueError(
            f"No bus stop found near '{source}'. "
            "Try a well-known location like 'Howrah Station' or 'Esplanade'."
        )
    if dst_stop is None:
        raise ValueError(
            f"No bus stop found near '{destination}'. "
            "Try a well-known location like 'Park Street' or 'Gariahat'."
        )

    src_coords = (src_stop["lat"], src_stop["lon"])
    dst_coords = (dst_stop["lat"], dst_stop["lon"])

    # ── BFS path-find ─────────────────────────────────────────────────────────
    path = find_bus_path(src_stop["stop_id"], dst_stop["stop_id"])

    if path is None or len(path) < 2:
        # No path — return a straight-line fallback (consistent with metro fallback)
        total_km  = round(_haversine_km(*src_coords, *dst_coords), 2)
        total_min = round((total_km / 20.0) * 60)   # ~20 km/h city bus average
        geo = _straight_geojson(
            [src_coords, dst_coords], "Bus (no direct route)", total_km, total_min
        )
        return {
            "source": source, "destination": destination,
            "src_coords": list(src_coords), "dst_coords": list(dst_coords),
            "mode": "bus",
            "routes": [{
                "id": 0, "label": "Bus (estimate)",
                "road_names": [], "distance_km": total_km,
                "travel_time_min": total_min, "geojson": geo,
                "coords": [list(src_coords), list(dst_coords)],
                "mode": "bus", "segments": [],
                "bus_note": (
                    f"No direct bus connection found between "
                    f"'{src_stop['name']}' and '{dst_stop['name']}'. "
                    "Showing straight-line estimate."
                ),
            }],
        }

    # ── Build GeoJSON from stop coordinates ───────────────────────────────────
    all_stops = load_stops()
    coord_pairs = get_path_coordinates(path)     # [[lat, lon], ...]

    # Straight-line distance along stops
    total_km = 0.0
    for i in range(len(coord_pairs) - 1):
        total_km += _haversine_km(
            coord_pairs[i][0],   coord_pairs[i][1],
            coord_pairs[i+1][0], coord_pairs[i+1][1],
        )
    total_km  = round(total_km, 2)
    # City bus average ~20 km/h; add 3 min boarding buffer
    total_min = round((total_km / 20.0) * 60) + 3

    # GeoJSON expects [lon, lat]
    geo_coords = [[c[1], c[0]] for c in coord_pairs]
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": geo_coords},
            "properties": {
                "label": "Bus Route",
                "distance_km": total_km,
                "travel_time_min": total_min,
                "color": "#FF5722",
            },
        }],
    }

    # ── Stop name list for road_names (used for disruption matching) ──────────
    stop_names = [
        all_stops[sid]["name"] for sid in path if sid in all_stops
    ]

    # Which bus route numbers run this path?
    matching_routes = get_routes_through_stops([path[0], path[-1]])
    if matching_routes:
        route_label = matching_routes[0]["route_name"]
    else:
        route_label = f"Bus · {src_stop['name']} → {dst_stop['name']}"

    # Build segment list for the journey breakdown panel
    segments: list[dict] = []
    for i, sid in enumerate(path):
        if sid not in all_stops:
            continue
        segments.append({
            "type":        "bus",
            "stop_id":     sid,
            "name":        all_stops[sid]["name"],
            "lat":         all_stops[sid]["lat"],
            "lon":         all_stops[sid]["lon"],
            "is_first":    i == 0,
            "is_last":     i == len(path) - 1,
        })

    return {
        "source": source, "destination": destination,
        "src_coords": list(src_coords),
        "dst_coords": list(dst_coords),
        "mode": "bus",
        "routes": [{
            "id": 0,
            "label": route_label,
            "road_names": stop_names,
            "distance_km": total_km,
            "travel_time_min": total_min,
            "geojson": geojson,
            "coords": coord_pairs,
            "mode": "bus",
            "segments": segments,
            "bus_stops": stop_names,
            "num_stops": len(path) - 1,
            "src_stop": src_stop["name"],
            "dst_stop": dst_stop["name"],
            "matching_routes": matching_routes,
        }],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def _normalize_modes(modes: list[str]) -> list[str]:
    normalized: list[str] = []
    for mode in modes:
        if not isinstance(mode, str):
            continue
        m = mode.strip().lower()
        if m and m not in normalized:
            normalized.append(m)
    return normalized


def get_routes_for_modes(
    source: str,
    destination: str,
    modes: list[str],
    events: list[dict] | None = None,
) -> dict:
    modes = _normalize_modes(modes)
    if not modes:
        raise ValueError("At least one transport mode must be provided.")
    if len(modes) == 1:
        return get_routes_for_mode(source, destination, modes[0], events=events)
    if len(modes) == 2:
        if "metro" not in modes:
            raise ValueError("Two-mode routing supports only metro + walk/bike/drive.")
        feeder = next(m for m in modes if m != "metro")
        return _metro_combo_route(source, destination, feeder, events=events)
    if len(modes) == 3:
        # The only supported three-mode combo: walk + metro + drive (cab).
        # Accept it in any order as long as all three modes are present.
        if set(modes) == {"walk", "metro", "drive"}:
            return _metro_mixed_route(source, destination, events=events)
        raise ValueError(
            "Three-mode routing only supports walk + metro + drive (cab). "
            f"Got: {modes}"
        )
    raise ValueError("A maximum of three transport modes are supported.")


def get_routes_for_mode(
    source: str,
    destination: str,
    mode: TransportMode,
    events: list[dict] | None = None,
) -> dict:
    if mode == "drive":
        return _road_routes(source, destination, "drive", 30.0)
    if mode == "walk":
        return _road_routes(source, destination, "walk", 5.0)
    if mode == "bike":
        return _road_routes(source, destination, "bike", 15.0)
    if mode == "metro":
        return _metro_route(source, destination, events=events)
    if mode == "bus":
        return _bus_route(source, destination)
    raise ValueError(f"Unknown mode: {mode!r}")


def get_metro_geojson_overlay() -> dict:
    """GeoJSON overlay: all 5 metro lines + all stations."""
    features = []
    for line_name, stations in _LINE_STATIONS.items():
        coords = [[s["lon"], s["lat"]] for s in stations]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "line":  line_name,
                "color": LINE_COLORS.get(line_name, "#888"),
                "name":  f"Kolkata Metro {line_name.title()} Line",
            },
        })
    for s in METRO_STATIONS:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {
                "id":    s["id"],
                "name":  s["name"],
                "line":  s["line"],
                "color": LINE_COLORS.get(s["line"], "#888"),
            },
        })
    return {"type": "FeatureCollection", "features": features}
