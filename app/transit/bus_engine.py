"""
bus_engine.py — BFS path-finder over the Kolkata bus network graph.
"""

from collections import deque

from transit.bus_graph import build_bus_graph


def find_bus_path(source: str, destination: str) -> list[str] | None:
    """
    Find the shortest (fewest stops) bus path between two stop IDs.

    Args:
        source:      Starting stop ID  (e.g. "S_HWH")
        destination: Target stop ID    (e.g. "S_AIRPORT")

    Returns:
        Ordered list of stop IDs from source to destination,
        or None if no path exists.
    """
    graph = build_bus_graph()

    if source not in graph:
        print(f"[BusEngine] Unknown source stop: {source}")
        return None
    if destination not in graph:
        print(f"[BusEngine] Unknown destination stop: {destination}")
        return None
    if source == destination:
        return [source]

    queue: deque[list[str]] = deque([[source]])
    visited: set[str] = {source}

    while queue:
        path = queue.popleft()
        node = path[-1]

        for neighbour in graph.get(node, []):
            if neighbour in visited:
                continue
            new_path = path + [neighbour]
            if neighbour == destination:
                return new_path
            visited.add(neighbour)
            queue.append(new_path)

    return None


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "S_HWH"
    dst = sys.argv[2] if len(sys.argv) > 2 else "S_AIRPORT"
    path = find_bus_path(src, dst)
    if path:
        print(f"\nRoute found ({len(path) - 1} stops):\n  " + " → ".join(path))
    else:
        print(f"\nNo route found from {src} to {dst}")
