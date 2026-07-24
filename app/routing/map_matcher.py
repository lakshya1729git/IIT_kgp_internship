"""
map_matcher.py — Snap event coordinates to nearest OSMnx road segments
=======================================================================
Converts (lat, lon) from LLM-extracted events into OSMnx node/edge IDs
so that disruption risk can be applied to the correct graph edges during
cost-function weighting.

The module uses a simple Nearest-Node approach (no Hidden Markov Model):
  1. For each event coordinate, find the nearest OSMnx node.
  2. Walk outward ±1 hop to collect affected edge set.
  3. Return matched (u, v, key) tuples — ready for edge attribute updates.

This is the right tradeoff for a city-level system:
  - HMM map-matching is designed for GPS traces (sequences of points).
  - Single LLM-extracted coordinates are better served by nearest-node.
  - Runtime is O(1) per event via OSMnx's spatial index.

Usage:
    from routing.map_matcher import MapMatcher

    matcher = MapMatcher(graph)            # pass the OSMnx drive graph
    results = matcher.match(events)        # list of event dicts with lat/lon
    for r in results:
        print(r["edge_keys"])              # [(u, v, k), ...]
        print(r["nearest_road"])           # "AJC Bose Road" (OSMnx street name)
"""

from __future__ import annotations

from typing import Optional

# Radius (metres) around the matched node — edges within this circle are
# considered "affected" by the event.  Chosen to cover a typical city block.
DEFAULT_RADIUS_M = 250


class MapMatcher:
    """
    Snap event (lat, lon) coordinates to OSMnx road graph edges.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        An OSMnx drive (or walk/bike) graph for the city.
    radius_m : float
        Search radius in metres for affected edges.  Events with imprecise
        coordinates benefit from a larger radius; TomTom incidents (precise)
        can use a smaller value.
    """

    def __init__(self, G, radius_m: float = DEFAULT_RADIUS_M):
        try:
            import osmnx as ox  # noqa: F401
        except ImportError:
            raise ImportError(
                "osmnx is required for map matching. pip install osmnx"
            )
        self._G        = G
        self._radius_m = radius_m

    # ── Public API ────────────────────────────────────────────────────────────

    def match_point(
        self,
        lat: float,
        lon: float,
        radius_m: Optional[float] = None,
    ) -> dict:
        """
        Match a single (lat, lon) point to the nearest road edges.

        Returns a dict with:
            nearest_node  : int        — OSMnx node ID
            nearest_road  : str        — street name (or empty string)
            edge_keys     : list[(u,v,k)] — edges within radius
            dist_m        : float      — distance to nearest node (metres)
        """
        import osmnx as ox

        r = radius_m if radius_m is not None else self._radius_m

        # 1. Nearest node (O(log n) via spatial index)
        node_id, dist_m = ox.distance.nearest_nodes(
            self._G, X=lon, Y=lat, return_dist=True
        )

        # 2. Street name at the nearest node
        road_name = self._road_name_at_node(node_id)

        # 3. Collect edges within radius
        edge_keys = self._edges_within_radius(lat, lon, r)

        return {
            "nearest_node": node_id,
            "nearest_road": road_name,
            "edge_keys":    edge_keys,
            "dist_m":       float(dist_m),
        }

    def match(self, events: list[dict], radius_m: Optional[float] = None) -> list[dict]:
        """
        Batch-match a list of event dicts that contain lat/lon fields.

        Events without coordinates are passed through unchanged (with
        edge_keys=[] so callers can safely iterate).

        Args:
            events:   List of event dicts from the LLM extraction pipeline.
            radius_m: Override the instance radius for this batch.

        Returns:
            Same list, each event augmented with:
                matched_node  : int | None
                matched_road  : str
                matched_edges : list[(u, v, k)]
                match_dist_m  : float | None
        """
        results = []
        for ev in events:
            lat = ev.get("lat")
            lon = ev.get("lon")

            if lat is None or lon is None:
                results.append({
                    **ev,
                    "matched_node":  None,
                    "matched_road":  ev.get("road_name", ""),
                    "matched_edges": [],
                    "match_dist_m":  None,
                })
                continue

            try:
                m = self.match_point(float(lat), float(lon), radius_m)
                results.append({
                    **ev,
                    "matched_node":  m["nearest_node"],
                    "matched_road":  m["nearest_road"] or ev.get("road_name", ""),
                    "matched_edges": m["edge_keys"],
                    "match_dist_m":  m["dist_m"],
                })
            except Exception as exc:
                # Non-fatal — fall back gracefully
                results.append({
                    **ev,
                    "matched_node":  None,
                    "matched_road":  ev.get("road_name", ""),
                    "matched_edges": [],
                    "match_dist_m":  None,
                    "_match_error":  str(exc),
                })

        return results

    def apply_disruption_weights(
        self,
        matched_events: list[dict],
        weight_attr:    str  = "disruption_weight",
        mode:           str  = "max",
    ) -> None:
        """
        Write disruption weights onto matched graph edges in-place.

        For each matched event, the confidence × severity product is written
        to the edge data dict under `weight_attr`.  Subsequent calls to
        route_engine can read this attribute for cost-function weighting.

        Args:
            matched_events: Output of match() — events with matched_edges.
            weight_attr:    Edge attribute name to write.
            mode:           "max" — keep the highest disruption weight if
                            multiple events hit the same edge.
                            "sum" — accumulate all weights (may exceed 1.0).
        """
        from config import SEVERITY_SCORES

        max_score = max(SEVERITY_SCORES.values()) if SEVERITY_SCORES else 10

        for ev in matched_events:
            sev   = ev.get("severity", "low")
            conf  = float(ev.get("confidence", 0.5))
            score = SEVERITY_SCORES.get(sev, 1)
            risk  = conf * (score / max_score)   # normalised ∈ [0, 1]

            for (u, v, k) in ev.get("matched_edges", []):
                edge_data = self._G.get_edge_data(u, v, k)
                if edge_data is None:
                    continue
                current = edge_data.get(weight_attr, 0.0)
                if mode == "max":
                    edge_data[weight_attr] = max(current, risk)
                else:
                    edge_data[weight_attr] = current + risk

    # ── Private helpers ───────────────────────────────────────────────────────

    def _road_name_at_node(self, node_id: int) -> str:
        """
        Return the most common street name among edges incident to node_id.
        Falls back to empty string if no name is found.
        """
        names: list[str] = []
        for _, _, data in self._G.edges(node_id, data=True):
            name = data.get("name")
            if isinstance(name, list):
                names.extend(n for n in name if isinstance(n, str))
            elif isinstance(name, str):
                names.append(name)

        if not names:
            return ""

        # Return the most frequent street name at this node
        from collections import Counter
        return Counter(names).most_common(1)[0][0]

    def _edges_within_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> list[tuple]:
        """
        Return (u, v, key) tuples for all edges whose source node falls
        within radius_m metres of (lat, lon).

        Uses a bounding-box pre-filter for performance, then exact haversine
        distance for final selection.
        """
        import math

        # Approximate degrees per metre at this latitude
        lat_deg_per_m = 1.0 / 111_320.0
        lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)))

        lat_delta = radius_m * lat_deg_per_m
        lon_delta = radius_m * lon_deg_per_m

        lat_min, lat_max = lat - lat_delta, lat + lat_delta
        lon_min, lon_max = lon - lon_delta, lon + lon_delta

        matched_edges = []
        for node_id, node_data in self._G.nodes(data=True):
            n_lat = node_data.get("y")
            n_lon = node_data.get("x")
            if n_lat is None or n_lon is None:
                continue

            # Bounding-box pre-filter
            if not (lat_min <= n_lat <= lat_max and lon_min <= n_lon <= lon_max):
                continue

            # Exact haversine distance
            if _haversine_m(lat, lon, n_lat, n_lon) <= radius_m:
                for u, v, k in self._G.edges(node_id, keys=True):
                    matched_edges.append((u, v, k))

        return matched_edges


# ── Utility ───────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    import math
    R = 6_371_000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def match_events_to_graph(
    G,
    events:   list[dict],
    radius_m: float = DEFAULT_RADIUS_M,
) -> list[dict]:
    """
    Module-level convenience function — wraps MapMatcher for one-liner use.

    Example:
        from routing.map_matcher import match_events_to_graph
        matched = match_events_to_graph(drive_graph, events)
    """
    return MapMatcher(G, radius_m=radius_m).match(events)
