"""
bus_overlay.py — Convert a bus stop ID path to lat/lon coordinates
for rendering as a map polyline.
"""

from transit.bus_graph import load_stops


def get_path_coordinates(path: list[str]) -> list[list[float]]:
    """
    Convert an ordered list of stop IDs to [[lat, lon], ...] coordinates.

    Args:
        path: Ordered list of stop IDs (e.g. from find_bus_path)

    Returns:
        List of [lat, lon] pairs for each stop found in the stops database.
        Stops not found in the database are silently skipped.
    """
    stops = load_stops()
    coordinates: list[list[float]] = []

    for stop_id in path:
        if stop_id in stops:
            coordinates.append([stops[stop_id]["lat"], stops[stop_id]["lon"]])
        else:
            print(f"[BusOverlay] Stop ID not found in database: {stop_id}")

    return coordinates


if __name__ == "__main__":
    sample_path = ["S_HWH", "S_BBDBAG", "S_CENTRAL", "S_GIRISHPARK"]
    coords = get_path_coordinates(sample_path)
    for stop_id, coord in zip(sample_path, coords):
        print(f"  {stop_id}: {coord}")
