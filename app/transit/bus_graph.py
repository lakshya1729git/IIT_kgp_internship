"""
bus_graph.py — Kolkata bus network graph builder.

Loads stop coordinates, route definitions, and ordered stop sequences
from the CSV data files and builds a bidirectional adjacency graph
suitable for BFS path-finding.
"""

import csv
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def load_stops() -> dict[str, dict]:
    """Return {stop_id: {name, lat, lon}} for all stops."""
    stops: dict[str, dict] = {}
    with open(DATA_DIR / "stops.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values defensively
            row = {k.strip(): v.strip() for k, v in row.items()}
            stop_id = row.get("stop_id", "").strip()
            if not stop_id:
                continue
            try:
                stops[stop_id] = {
                    "name": row["stop_name"],
                    "lat":  float(row["lat"]),
                    "lon":  float(row["lon"]),
                }
            except (KeyError, ValueError) as e:
                print(f"[BusGraph] Skipping stop '{stop_id}': {e}")
    return stops


def load_routes() -> dict[str, dict]:
    """Return {route_id: row_dict} for all routes."""
    routes: dict[str, dict] = {}
    with open(DATA_DIR / "routes.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            route_id = row.get("route_id", "").strip()
            if route_id:
                routes[route_id] = row
    return routes


def load_route_sequences() -> dict[str, list[tuple[int, str]]]:
    """Return {route_id: [(sequence_num, stop_id), ...]} sorted by sequence."""
    sequences: dict[str, list[tuple[int, str]]] = {}
    with open(DATA_DIR / "route_stop_sequence.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            route_id = row.get("route_id", "").strip()
            if not route_id:
                continue
            try:
                sequences.setdefault(route_id, []).append(
                    (int(row["stop_sequence"]), row["stop_id"])
                )
            except (KeyError, ValueError) as e:
                print(f"[BusGraph] Skipping sequence row for '{route_id}': {e}")
    return sequences


def build_bus_graph() -> dict[str, list[str]]:
    """
    Build a bidirectional adjacency graph over bus stop IDs.

    Nodes  : stop IDs (e.g. "S_HWH")
    Edges  : consecutive stops on the same route (both directions)

    Returns {stop_id: [neighbour_stop_id, ...]}
    """
    graph_sets: dict[str, set[str]] = {}
    sequences = load_route_sequences()

    # Use sets during construction to avoid duplicate edges from overlapping routes
    graph_sets: dict[str, set[str]] = {}

    for route_id, stops in sequences.items():
        stops.sort(key=lambda x: x[0])   # sort by sequence number
        for i in range(len(stops) - 1):
            cur  = stops[i][1]
            nxt  = stops[i + 1][1]
            graph_sets.setdefault(cur, set()).add(nxt)
            graph_sets.setdefault(nxt, set()).add(cur)

    # Convert sets back to sorted lists for deterministic BFS ordering
    return {node: sorted(neighbours) for node, neighbours in graph_sets.items()}


def find_nearest_stop(lat: float, lon: float, max_distance_km: float = 5.0) -> dict | None:
    """
    Return the nearest bus stop to the given coordinates.

    Args:
        lat:             Query latitude
        lon:             Query longitude
        max_distance_km: Ignore stops farther than this (default 5 km)

    Returns:
        {stop_id, name, lat, lon, distance_km} or None if nothing is within range.
    """
    stops = load_stops()
    best: dict | None = None
    best_dist = float("inf")

    for stop_id, info in stops.items():
        d = _haversine_km(lat, lon, info["lat"], info["lon"])
        if d < best_dist:
            best_dist = d
            best = {"stop_id": stop_id, **info, "distance_km": round(d, 3)}

    if best is None or best_dist > max_distance_km:
        return None
    return best


def find_nearest_stop_by_name(name: str) -> dict | None:
    """
    Return the stop whose name best matches *name* (case-insensitive substring
    search), or None if nothing matches.

    Returns {stop_id, name, lat, lon}.
    """
    stops = load_stops()
    q = name.strip().lower()

    # Exact match first
    for stop_id, info in stops.items():
        if info["name"].lower() == q:
            return {"stop_id": stop_id, **info}

    # Substring match
    for stop_id, info in stops.items():
        sn = info["name"].lower()
        if q in sn or sn in q:
            return {"stop_id": stop_id, **info}

    # Token overlap: any word in the query matches any word in the stop name
    q_tokens = set(q.split())
    for stop_id, info in stops.items():
        sn_tokens = set(info["name"].lower().split())
        if q_tokens & sn_tokens:
            return {"stop_id": stop_id, **info}

    return None


def get_routes_through_stops(stop_ids: list[str]) -> list[dict]:
    """
    Return all routes that pass through *all* of the given stop IDs.

    Useful for labelling a BFS path with the bus route number.
    Returns list of {route_id, route_name, route_type}.
    """
    if not stop_ids:
        return []

    routes   = load_routes()
    seqs     = load_route_sequences()
    stop_set = set(stop_ids)

    result = []
    for route_id, stop_seq in seqs.items():
        route_stops = {s for _, s in stop_seq}
        if stop_set.issubset(route_stops):
            meta = routes.get(route_id, {})
            result.append({
                "route_id":   route_id,
                "route_name": meta.get("route_name", route_id),
                "route_type": meta.get("route_type", ""),
            })
    return result


if __name__ == "__main__":
    graph = build_bus_graph()
    print(f"\nTotal nodes : {len(graph)}")
    total_edges = sum(len(v) for v in graph.values()) // 2
    print(f"Total edges : {total_edges}")
    print(f"\nEsplanade connections: {graph.get('S_ESPL')}")

    stops = load_stops()
    print(f"\nTotal stops loaded: {len(stops)}")
