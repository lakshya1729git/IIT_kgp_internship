"""
route_engine.py — OSMnx-based routing for Kolkata + North suburbs

Downloads the road network by named administrative places rather than a
bounding box. This avoids the Overpass "area too large" subdivision
problem while still covering all 15 menu localities.

Places covered:
  - Kolkata              (core city — Howrah bridge to Jadavpur/Behala)
  - Bidhannagar          (Salt Lake Sector V + New Town + Rajarhat)
  - North Dum Dum        (Dum Dum corridor, airport area)
  - Barasat              (northernmost menu locality)

WHY graph_from_place INSTEAD OF graph_from_bbox:
  A bounding box large enough to cover Barasat (~22.72 N) triggers
  Overpass's area-subdivision warning (13,000+ sub-queries) because
  the projected area exceeds the server's per-query limit. Named
  administrative polygons are small and well-defined — each is sent
  as a single compact Overpass query with no subdivision.

WHY PICKLE:
  Pickle serialises the Python object directly — no XML parsing,
  loads in ~2 seconds, uses ~3x less peak memory than GraphML.
"""

import os
import pickle
import math
import osmnx as ox

from config import GRAPH_CACHE_PATH, MAX_ROUTES, MIN_ROUTE_DIVERGENCE, MAX_CONSECUTIVE_DUPES


# ── Named places that together cover all 15 menu localities ──────────────────
GRAPH_PLACES = [
    "Kolkata, West Bengal, India",
    "Howrah, West Bengal, India",         # Howrah Station, Shibpur, Bally corridor
    "Bidhannagar, West Bengal, India",    # Salt Lake + New Town + Rajarhat
    "North Dum Dum, West Bengal, India",  # Dum Dum corridor
    "Madhyamgram, West Bengal, India",    # NH-12 corridor bridging Dum Dum → Barasat
    "Barasat I, West Bengal, India",      # Barasat (north block)
    "Barasat II, West Bengal, India",     # Barasat (south block)
]


# ── Graph loader (cached) ─────────────────────────────────────────────────────

# In-process graph cache
_GRAPH_CACHE: dict[str, object] = {}


def _load_graph():
    """
    Load the Kolkata + suburbs drive network.
    - First run  : downloads from OSM by place names, saves to pickle
    - Later runs : loads from the cached pickle file (~2 seconds)
    The loaded graph is kept in memory for the lifetime of the process.
    """
    cache_path = GRAPH_CACHE_PATH

    if cache_path in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_path]

    if os.path.exists(cache_path):
        print(f"  [Route] Loading cached graph from {cache_path} ...")
        with open(cache_path, "rb") as f:
            graph = pickle.load(f)
        print(f"  [Route] Graph loaded ({len(graph.nodes):,} nodes, "
              f"{len(graph.edges):,} edges)")
    else:
        graph = _download_and_cache(cache_path)

    _GRAPH_CACHE[cache_path] = graph
    return graph


def _download_and_cache(cache_path: str):
    """
    Download the OSM road network for the named places and cache it.

    Each place is a small administrative polygon — Overpass handles
    them individually with no subdivision, keeping queries fast.
    Tries multiple public mirrors if the primary is unavailable.
    """
    # OSMnx 2.x appends "/interpreter" to overpass_url automatically.
    # Set the base URL only (no trailing /interpreter).
    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://overpass.openstreetmap.ru/api",
    ]

    print(f"  [Route] Downloading OSM road network for:")
    for p in GRAPH_PLACES:
        print(f"           * {p}")
    print(f"  [Route] One-time download, ~30-60 seconds ...")

    ox.settings.timeout = 180  # 3 min per endpoint before trying next mirror

    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"  [Route] Trying Overpass endpoint: {endpoint}")
            ox.settings.overpass_url = endpoint

            graph = ox.graph_from_place(
                GRAPH_PLACES,
                network_type="drive",
                retain_all=False,
            )

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"  [Route] Graph cached to {cache_path} "
                  f"({len(graph.nodes):,} nodes, {len(graph.edges):,} edges)")
            return graph

        except Exception as e:
            print(f"  [Route] Endpoint failed: {e.__class__.__name__}: {str(e)[:120]}")
            last_error = e
            continue

    raise RuntimeError(
        "All Overpass API endpoints failed.\n"
        "The public servers may be overloaded — wait a few minutes and retry.\n"
        f"Last error: {last_error}"
    )


# ── Geocoding with city context ───────────────────────────────────────────────

# Hardcoded coordinates for Kolkata landmarks that Nominatim frequently
# misresolves (returns wrong country/city).  Shared with multimodal._geocode.
_KOLKATA_LANDMARKS: dict[str, tuple[float, float]] = {
    "ecopark":                (22.5969, 88.4790),
    "eco park":               (22.5969, 88.4790),
    "eco tourism park":       (22.5969, 88.4790),
    "new town":               (22.5890, 88.4700),
    "rajarhat":               (22.5890, 88.4700),
    "action area 1":          (22.5870, 88.4710),
    "action area i":          (22.5870, 88.4710),
    "new town action area 1": (22.5870, 88.4710),
    "biswa bangla gate":      (22.5853, 88.4760),
    "airport":                (22.6548, 88.4467),
    "dum dum airport":        (22.6548, 88.4467),
    "netaji airport":         (22.6548, 88.4467),
    "science city":           (22.5369, 88.3966),
    "victoria memorial":      (22.5448, 88.3426),
    "dakshineswar":           (22.6537, 88.3578),
    "belur math":             (22.6368, 88.3518),
    "botanical garden":       (22.5622, 88.3072),
    "acharya jagadish":       (22.5622, 88.3072),
    "rabindra sarobar":       (22.5091, 88.3509),
    "dhakuria lake":          (22.5091, 88.3509),
    "ruby hospital":          (22.5209, 88.3969),
    "garia":                  (22.4731, 88.3869),
    "newmarket":              (22.5630, 88.3510),
    "new market":             (22.5630, 88.3510),
    "college street":         (22.5796, 88.3634),
    "china town":             (22.5741, 88.3632),
    "chinatown":              (22.5741, 88.3632),
}

def _geocode_with_context(place: str) -> tuple[float, float]:
    """
    Geocode a place name, appending ', West Bengal, India' if no
    regional keyword is present. Uses West Bengal (not just Kolkata)
    so suburbs like Barasat resolve to the correct district.

    Resolution order:
      1. Hardcoded landmark table (fast, no network, no wrong-country risk)
      2. Nominatim with ', West Bengal, India' suffix
      3. Nominatim with ', Kolkata, West Bengal, India' suffix (stricter)
      All Nominatim results are validated against a West Bengal bounding box.
    """
    # ── 1. Landmark table lookup ──────────────────────────────────────────────
    p_lower = place.lower().strip()
    if p_lower in _KOLKATA_LANDMARKS:
        coords = _KOLKATA_LANDMARKS[p_lower]
        print(f"  [Route] Landmark match '{place}' -> {coords}")
        return coords
    for key, coords in _KOLKATA_LANDMARKS.items():
        if key in p_lower or p_lower in key:
            print(f"  [Route] Landmark partial match '{place}' -> {coords}")
            return coords

    # ── 2. Nominatim with bounding-box guard ──────────────────────────────────
    # Approximate bounding box for West Bengal + immediate surroundings
    _WB_LAT_MIN, _WB_LAT_MAX = 21.0, 27.5
    _WB_LON_MIN, _WB_LON_MAX = 85.5, 90.5

    def _in_region(lat: float, lon: float) -> bool:
        return _WB_LAT_MIN <= lat <= _WB_LAT_MAX and _WB_LON_MIN <= lon <= _WB_LON_MAX

    place_lower = place.lower()
    needs_hint = not any(
        kw in place_lower
        for kw in ("kolkata", "calcutta", "india", "west bengal", "wb",
                   "north 24", "south 24", "24 parganas")
    )

    # Always keep the region suffix — never strip it on retry
    query = f"{place}, West Bengal, India" if needs_hint else place

    # Try primary query
    try:
        coords = ox.geocode(query)
        if _in_region(coords[0], coords[1]):
            print(f"  [Route] Geocoded '{place}' -> {coords}")
            return coords
        print(f"  [Route] Geocode for '{place}' returned out-of-region result {coords}, retrying with stricter query ...")
    except Exception as e:
        print(f"  [Route] Geocode failed for '{query}': {e}, retrying ...")

    # Retry with an even more specific qualifier
    fallback_query = f"{place}, Kolkata, West Bengal, India"
    try:
        coords = ox.geocode(fallback_query)
        if _in_region(coords[0], coords[1]):
            print(f"  [Route] Geocoded '{place}' (fallback) -> {coords}")
            return coords
        raise ValueError(
            f"Could not geocode '{place}' within West Bengal. "
            f"Got coordinates {coords} which appear to be outside the region. "
            "Try adding 'Kolkata' or a nearby landmark to the name."
        )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Could not geocode '{place}'. "
            "Try adding 'Kolkata' to the name, or use a well-known landmark."
        ) from e


# ── Haversine distance (sanity check) ────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ── Public API ────────────────────────────────────────────────────────────────

def get_route(source: str, destination: str):
    """
    Compute the shortest driving route between source and destination.

    Returns:
        (graph, route) -- OSMnx MultiDiGraph and list of node IDs
    """
    graph = _load_graph()

    src_coords  = _geocode_with_context(source)
    dest_coords = _geocode_with_context(destination)

    src_node  = ox.distance.nearest_nodes(graph, X=src_coords[1],  Y=src_coords[0])
    dest_node = ox.distance.nearest_nodes(graph, X=dest_coords[1], Y=dest_coords[0])

    route = ox.shortest_path(graph, src_node, dest_node)
    if route is None:
        raise ValueError(
            f"No route found between '{source}' and '{destination}'. "
            "Try a more specific address or well-known landmark."
        )

    return graph, route


def _routes_are_distinct(edge_set_a: frozenset, edge_set_b: frozenset,
                         min_divergence: float = 0.40) -> bool:
    """
    Return True if two routes are genuinely different.

    Two routes are considered duplicates if they share more than
    (1 - min_divergence) of their edges by Jaccard similarity.
    Default: routes must differ by at least 40% of edges.
    """
    if not edge_set_a or not edge_set_b:
        return True
    intersection = len(edge_set_a & edge_set_b)
    union        = len(edge_set_a | edge_set_b)
    jaccard      = intersection / union if union else 1.0
    return jaccard < (1.0 - min_divergence)


def get_k_routes(source: str, destination: str, k: int = None):
    """
    Compute all genuinely distinct driving routes between source and
    destination using edge-weight perturbation.

    When k is None (default), the algorithm runs until no more distinct
    routes can be found, up to MAX_ROUTES (from config / .env).
    Pass an explicit k to override.

    Two routes are considered distinct when their edge sets differ by at
    least MIN_ROUTE_DIVERGENCE (Jaccard threshold, from config / .env).

    Each route dict contains:
        route_id        -- integer index (0-based)
        route_label     -- "Route 1", "Route 2", ...
        nodes           -- list of OSMnx node IDs
        roads           -- list of unique named road segments
        distance_km     -- total distance in km
        travel_time_min -- estimated time at 30 km/h avg

    Returns:
        (graph, routes) -- OSMnx MultiDiGraph and list of route dicts
    """
    PENALTY_FACTOR      = 5.0
    MAX_ATTEMPTS_FACTOR = 6   # total attempts = max_routes * this

    max_routes = k if k is not None else MAX_ROUTES

    graph = _load_graph()

    src_coords  = _geocode_with_context(source)
    dest_coords = _geocode_with_context(destination)

    crow_km = _haversine_km(src_coords[0], src_coords[1],
                            dest_coords[0], dest_coords[1])
    print(f"  [Route] Straight-line distance: {crow_km:.1f} km")
    if crow_km < 1.0:
        print(f"  WARNING: Source and destination are only {crow_km:.2f} km apart "
              f"as the crow flies. Check that the place names are correct.")

    src_node  = ox.distance.nearest_nodes(graph, X=src_coords[1],  Y=src_coords[0])
    dest_node = ox.distance.nearest_nodes(graph, X=dest_coords[1], Y=dest_coords[0])

    G = graph.copy()
    routes_raw: list[list]          = []
    seen_edge_sets: list[frozenset] = []
    consecutive_dupes = 0

    for _attempt in range(max_routes * MAX_ATTEMPTS_FACTOR):
        if len(routes_raw) >= max_routes:
            print(f"  [Route] Reached cap of {max_routes} routes.")
            break
        if consecutive_dupes >= MAX_CONSECUTIVE_DUPES:
            print(f"  [Route] No new distinct routes found after "
                  f"{MAX_CONSECUTIVE_DUPES} consecutive attempts — stopping.")
            break

        route = ox.shortest_path(G, src_node, dest_node, weight="length")
        if route is None:
            break

        edge_set = frozenset(zip(route[:-1], route[1:]))

        is_new = all(
            _routes_are_distinct(edge_set, prev, MIN_ROUTE_DIVERGENCE)
            for prev in seen_edge_sets
        )

        for u, v in zip(route[:-1], route[1:]):
            for key in G[u][v]:
                G[u][v][key]["length"] *= PENALTY_FACTOR

        if not is_new:
            consecutive_dupes += 1
            continue

        routes_raw.append(route)
        seen_edge_sets.append(edge_set)
        consecutive_dupes = 0
        print(f"  [Route] Found Route {len(routes_raw)} "
              f"({len(route)} nodes, attempt {_attempt + 1})")

    if not routes_raw:
        raise ValueError(
            f"No route found between '{source}' and '{destination}'. "
            "Try a more specific address or well-known landmark."
        )

    print(f"  [Route] Total distinct routes found: {len(routes_raw)}")

    # Build route dicts using original (unperturbed) edge lengths
    routes = []
    for idx, node_list in enumerate(routes_raw):
        roads = extract_road_names(graph, node_list)

        total_m = sum(
            graph.get_edge_data(u, v, 0).get("length", 0)
            for u, v in zip(node_list[:-1], node_list[1:])
        )
        distance_km     = round(total_m / 1000, 2)
        travel_time_min = round((distance_km / 30) * 60)

        # Extract (lat, lon) for every node — used for spatial corridor filtering
        coords = [
            (graph.nodes[n]["y"], graph.nodes[n]["x"])
            for n in node_list
            if "y" in graph.nodes[n] and "x" in graph.nodes[n]
        ]

        routes.append({
            "route_id":        idx,
            "route_label":     f"Route {idx + 1}",
            "nodes":           node_list,
            "coords":          coords,   # list of (lat, lon) tuples
            "roads":           roads,
            "distance_km":     distance_km,
            "travel_time_min": travel_time_min,
        })

    return graph, routes


def get_multiple_routes(source: str, destination: str, n_routes: int = None) -> dict:
    """
    Compute all genuinely distinct driving routes and return them in the
    shape expected by the FastAPI frontend.

    When n_routes is None (default), the algorithm runs until no more
    distinct routes are found, up to MAX_ROUTES (from config / .env).
    Pass an explicit n_routes to override.
    """
    PENALTY_FACTOR      = 5.0
    MAX_ATTEMPTS_FACTOR = 6

    max_routes = n_routes if n_routes is not None else MAX_ROUTES

    graph = _load_graph()

    src_coords  = _geocode_with_context(source)
    dest_coords = _geocode_with_context(destination)

    crow_km = _haversine_km(src_coords[0], src_coords[1],
                            dest_coords[0], dest_coords[1])
    print(f"  [Route] Straight-line distance: {crow_km:.1f} km")

    src_node  = ox.distance.nearest_nodes(graph, X=src_coords[1],  Y=src_coords[0])
    dest_node = ox.distance.nearest_nodes(graph, X=dest_coords[1], Y=dest_coords[0])

    G = graph.copy()
    routes_raw: list[list]          = []
    seen_edge_sets: list[frozenset] = []
    consecutive_dupes = 0

    for _attempt in range(max_routes * MAX_ATTEMPTS_FACTOR):
        if len(routes_raw) >= max_routes:
            print(f"  [Route] Reached cap of {max_routes} routes.")
            break
        if consecutive_dupes >= MAX_CONSECUTIVE_DUPES:
            print(f"  [Route] No new distinct routes after "
                  f"{MAX_CONSECUTIVE_DUPES} consecutive attempts — stopping.")
            break

        route = ox.shortest_path(G, src_node, dest_node, weight="length")
        if route is None:
            break

        edge_set = frozenset(zip(route[:-1], route[1:]))

        is_new = all(
            _routes_are_distinct(edge_set, prev, MIN_ROUTE_DIVERGENCE)
            for prev in seen_edge_sets
        )

        for u, v in zip(route[:-1], route[1:]):
            for key in G[u][v]:
                G[u][v][key]["length"] *= PENALTY_FACTOR

        if not is_new:
            consecutive_dupes += 1
            continue

        routes_raw.append(route)
        seen_edge_sets.append(edge_set)
        consecutive_dupes = 0
        print(f"  [Route] Found Route {len(routes_raw)} "
              f"({len(route)} nodes, attempt {_attempt + 1})")

    if not routes_raw:
        raise ValueError(
            f"No route found between '{source}' and '{destination}'. "
            "Try a more specific address or a well-known landmark."
        )

    print(f"  [Route] Total distinct routes found: {len(routes_raw)}")

    routes_out = []
    for idx, node_list in enumerate(routes_raw):
        roads = extract_road_names(graph, node_list)

        total_m = sum(
            graph.get_edge_data(u, v, 0).get("length", 0)
            for u, v in zip(node_list[:-1], node_list[1:])
        )
        distance_km     = round(total_m / 1000, 2)
        travel_time_min = round((distance_km / 30) * 60)

        # (lat, lon) per node — used by spatial corridor filter
        coords = [
            (graph.nodes[n]["y"], graph.nodes[n]["x"])
            for n in node_list
            if "y" in graph.nodes[n] and "x" in graph.nodes[n]
        ]

        # GeoJSON — coordinates must be [lon, lat] per spec
        geojson_coords = [
            [graph.nodes[n]["x"], graph.nodes[n]["y"]]
            for n in node_list
            if "x" in graph.nodes[n] and "y" in graph.nodes[n]
        ]
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": geojson_coords,
                },
                "properties": {
                    "label":           f"Route {idx + 1}",
                    "distance_km":     distance_km,
                    "travel_time_min": travel_time_min,
                },
            }],
        }

        routes_out.append({
            "id":              idx,
            "label":           f"Route {idx + 1}",
            "road_names":      roads,
            "distance_km":     distance_km,
            "travel_time_min": travel_time_min,
            "geojson":         geojson,
            "node_ids":        node_list,
            "coords":          coords,
        })

    return {
        "graph":       graph,        # pop before serialising to JSON
        "source":      source,
        "destination": destination,
        "src_coords":  list(src_coords),
        "dst_coords":  list(dest_coords),
        "routes":      routes_out,
    }


def extract_road_names(graph, route: list) -> list[str]:
    """
    Extract unique named road segments from a route in traversal order.
    Filters out unnamed segments.
    """
    seen  = set()
    roads = []

    for u, v in zip(route[:-1], route[1:]):
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue

        edge = list(edge_data.values())[0]
        name = edge.get("name")
        if not name:
            continue

        names = name if isinstance(name, list) else [name]
        for n in names:
            if n and n not in seen:
                seen.add(n)
                roads.append(n)

    return roads