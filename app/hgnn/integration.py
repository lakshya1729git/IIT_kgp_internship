"""
integration.py — Drop-in helpers that plug HGNN into the existing pipeline
===========================================================================
Three functions:

  1. enhance_event_confidences(events, all_road_names)
     → Adjusts every event's confidence score using full city-wide graph context.
       Multi-source corroboration, spatial clustering, source reliability all
       influence the adjustment. Safe fallback if HGNN unavailable.

  2. score_route_with_hgnn(route_data, events)
     → Runs HGNN on ONLY the events relevant to this specific route.
       Returns a route-specific score [0, 10] that differs across routes.
       This is what makes Route 1 score differently from Route 3.

  3. get_cascade_road_names(route_road_names, all_events)
     → Uses HGNN road-near-road probabilities to find roads NOT on the route
       but likely to be affected due to spatial cascade (e.g. Howrah Bridge
       blocked → Strand Road congested even without explicit event for it).
       Returns expanded road list for broader event matching.

Blending strategy for confidence:
  final_conf = (1 - W_HGNN) × rule_based_conf + W_HGNN × hgnn_conf
  W_HGNN = 0.25 — enough to move scores meaningfully without overriding LLM
"""

from __future__ import annotations
from typing import Optional

# How much HGNN confidence adjustment blends into rule-based confidence
W_HGNN = 0.25

# Per-route HGNN score scale: road_prob [0,1] → score contribution [0,10]
HGNN_ROUTE_SCORE_SCALE = 10.0

# Cascade threshold: roads with HGNN prob above this are considered affected
CASCADE_PROB_THRESHOLD = 0.45

# Max extra roads to add via cascade expansion
MAX_CASCADE_ROADS = 5

# Default severity override threshold — conservative until eval confirms accuracy
_DEFAULT_SEV_THRESHOLD = 0.85


def _get_sev_override_threshold() -> float:
    """
    Return the severity override threshold.

    After evaluate_hgnn.py runs and confirms HGNN accuracy ≥ 0.80 on
    verified labels, it writes a tuned threshold to a small config file
    (hgnn_eval_threshold.txt).  We read that here so the threshold
    auto-lowers without requiring a code change.

    Falls back to the conservative default (0.85) if no eval has been run.
    """
    import os
    threshold_file = os.path.join(os.path.dirname(__file__), "weights", "eval_threshold.txt")
    try:
        if os.path.exists(threshold_file):
            with open(threshold_file) as f:
                val = float(f.read().strip())
            if 0.50 <= val <= 0.99:
                return val
    except Exception:
        pass
    return _DEFAULT_SEV_THRESHOLD


# ── 1. City-wide confidence enhancement ──────────────────────────────────────

def enhance_event_confidences(
    events:          list[dict],
    all_road_names:  list[str],
) -> list[dict]:
    """
    Blend HGNN graph-informed confidence into each event.

    Runs HGNN on the FULL event pool (city-wide) so the model sees all
    relationships: multi-source corroboration, spatial clustering, etc.
    Each event's confidence gets nudged up or down based on graph context.

    Args:
        events:         All extracted events (city-wide + route-specific)
        all_road_names: All road names across all routes (for graph context)

    Returns:
        Same list, events modified in-place with updated confidence +
        new fields: hgnn_confidence, confidence_pre_hgnn.
        Returns unchanged if HGNN unavailable.
    """
    if not events:
        return events

    from hgnn.inference import get_inference
    from scoring.congestion_score import compute_weighted_score

    hgnn   = get_inference()
    result = hgnn.predict(events, all_road_names)

    if result is None:
        return events   # graceful fallback

    adj_confs = result.get("event_confidence_adj", [])

    # Read threshold once — avoid a file open per event
    SEV_OVERRIDE_THRESHOLD = _get_sev_override_threshold()
    sev_preds = result.get("event_severity_preds", [])
    sev_probs = result.get("event_severity_probs", [])

    adjusted = 0
    for i, ev in enumerate(events):
        if i >= len(adj_confs):
            break

        hgnn_conf     = float(adj_confs[i])
        original_conf = float(ev.get("confidence", 0.5))

        blended = round((1.0 - W_HGNN) * original_conf + W_HGNN * hgnn_conf, 4)
        blended = max(0.0, min(1.0, blended))

        ev["hgnn_confidence"]     = round(hgnn_conf, 4)
        ev["confidence_pre_hgnn"] = original_conf
        ev["confidence"]          = blended
        ev["weighted_score"]      = compute_weighted_score(
            ev.get("severity", "low"), blended
        )

        # ── Severity correction ───────────────────────────────────────────────
        # Override LLM severity only when HGNN max-class probability exceeds
        # the threshold stored in eval_threshold.txt (written by evaluate_hgnn.py).
        # Threshold is 0.70 after eval confirmed 85.6% accuracy; falls back to
        # the conservative 0.85 default if no eval has been run yet.
        if i < len(sev_preds):
            hgnn_sev = sev_preds[i]
            orig_sev = ev.get("severity", "low")
            ev["hgnn_severity"]      = hgnn_sev
            ev["severity_pre_hgnn"]  = orig_sev
            if i < len(sev_probs):
                max_prob = max(sev_probs[i])
                if max_prob > SEV_OVERRIDE_THRESHOLD and hgnn_sev != orig_sev:
                    ev["severity"] = hgnn_sev
                    ev["weighted_score"] = compute_weighted_score(hgnn_sev, blended)
                    ev["severity_corrected"] = True

        adjusted += 1

    print(f"  [HGNN] Enhanced {adjusted} events: confidence + severity correction "
          f"(W_HGNN={W_HGNN}).")
    return events


# ── 2. Per-route HGNN scoring ─────────────────────────────────────────────────

def score_route_with_hgnn(
    route_data:    dict,
    route_events:  list[dict],
) -> float:
    """
    Compute an HGNN disruption score specific to THIS route.

    Key difference from the old approach: runs HGNN only on events that
    are already matched to this route (not city-wide). This means each
    route gets its own graph and its own disruption probabilities —
    Route 1 with 4 disruptions scores differently from Route 3 with 8.

    Score formula:
      - max_prob   : highest single-road probability (worst road on route)
      - avg_prob   : average across all route roads
      - n_affected : number of route roads with prob ≥ 0.4 (breadth signal)
      - combined   = 0.5 × max_prob + 0.3 × avg_prob + 0.2 × (n_affected / n_roads)

    The breadth signal (n_affected/n_roads) differentiates routes where ONE
    road is highly disrupted vs routes where MANY roads are moderately affected.

    Args:
        route_data:   Route dict (must contain 'road_names')
        route_events: Events already filtered to this route

    Returns:
        HGNN score [0, ~10]. 0.0 if HGNN unavailable or no events.
    """
    road_names = route_data.get("road_names") or route_data.get("roads") or []

    if not route_events or not road_names:
        return 0.0

    from hgnn.inference import get_inference

    hgnn   = get_inference()
    result = hgnn.predict(route_events, road_names)

    if result is None:
        return 0.0

    road_probs = result.get("road_disruption_probs", {})
    if not road_probs:
        return 0.0

    route_set = {r.lower() for r in road_names}

    route_road_probs = [
        prob for road, prob in road_probs.items()
        if road.lower() in route_set
    ]

    if not route_road_probs:
        return 0.0

    n_roads    = len(route_road_probs)
    max_prob   = max(route_road_probs)
    avg_prob   = sum(route_road_probs) / n_roads
    n_affected = sum(1 for p in route_road_probs if p >= 0.40)
    breadth    = n_affected / n_roads

    # Weighted combination captures both worst-case and breadth of disruption
    combined   = 0.50 * max_prob + 0.30 * avg_prob + 0.20 * breadth
    hgnn_score = round(combined * HGNN_ROUTE_SCORE_SCALE, 3)

    return hgnn_score


# ── 3. Cascade road expansion ─────────────────────────────────────────────────

def get_cascade_road_names(
    route_road_names: list[str],
    all_events:       list[dict],
    max_extra:        int = MAX_CASCADE_ROADS,
) -> list[str]:
    """
    Find roads NOT on the route but likely disrupted due to cascade effects.

    How it works:
      1. Run HGNN on all events with the route roads marked as is_on_route=1
      2. The road-near-road edges propagate disruption probability to adjacent roads
      3. Roads with high predicted probability (above CASCADE_PROB_THRESHOLD)
         are returned as additional roads to check for events

    Example: Howrah Bridge blocked → Strand Road and Brabourne Road get high
    cascade probability even if no event explicitly mentions them.

    Args:
        route_road_names: Roads on the current route
        all_events:       All city-wide events
        max_extra:        Max cascade roads to return

    Returns:
        List of additional road names to include in event matching.
        Empty list if HGNN unavailable or no cascade detected.
    """
    if not all_events or not route_road_names:
        return []

    from hgnn.inference import get_inference

    hgnn   = get_inference()
    result = hgnn.predict(all_events, route_road_names)

    if result is None:
        return []

    road_probs = result.get("road_disruption_probs", {})
    if not road_probs:
        return []

    route_set = {r.lower() for r in route_road_names}

    # Find high-probability roads that are NOT already on the route
    cascade_roads = [
        (road, prob)
        for road, prob in road_probs.items()
        if road.lower() not in route_set
        and prob >= CASCADE_PROB_THRESHOLD
    ]

    # Sort by probability descending, take top N
    cascade_roads.sort(key=lambda x: x[1], reverse=True)
    result_roads = [road for road, _ in cascade_roads[:max_extra]]

    if result_roads:
        print(f"  [HGNN] Cascade: {len(result_roads)} extra roads → {result_roads}")

    return result_roads


# ── 4. Explainability ─────────────────────────────────────────────────────────

def explain_route_risk(
    route_data:   dict,
    route_events: list[dict],
) -> dict:
    """
    Use HGNN attention weights to explain which events drove the route risk.

    Returns top contributing events with their attention scores —
    directly satisfies the "Explainable AI" claim in the research proposal.

    Example output:
      {
        "top_events": [
          {"event_type": "congestion", "location": "AJC Bose Road",
           "attention": 0.84, "severity": "high", "source": "tomtom_traffic"},
          ...
        ],
        "road_risk_scores": {"ajc bose road": 0.82, ...},
        "explanation": "Route risk driven by congestion on AJC Bose Road
                        (TomTom, attention=0.84) and road_closure on
                        Strand Road (RSS, attention=0.61)"
      }
    """
    road_names = route_data.get("road_names") or []
    if not route_events or not road_names:
        return {"top_events": [], "road_risk_scores": {}, "explanation": "No events to explain."}

    try:
        import torch
        from hgnn.inference import get_inference
        from hgnn.graph_builder import build_graph_from_events
        import numpy as np

        hgnn = get_inference()
        if not hgnn.is_ready():
            return {"top_events": [], "road_risk_scores": {},
                    "explanation": "HGNN not ready — train first."}

        graph_data, meta = build_graph_from_events(route_events, road_names)

        def _t(arr):
            return torch.from_numpy(arr).float()
        def _ei(arr):
            if arr.shape[1] == 0:
                return torch.zeros((2,0), dtype=torch.long)
            return torch.from_numpy(arr).long()

        with torch.no_grad():
            _, event_conf, _, attn_weights = hgnn._model(
                road_x     = _t(graph_data["road_x"]),
                event_x    = _t(graph_data["event_x"]),
                source_x   = _t(graph_data["source_x"]),
                location_x = _t(graph_data["location_x"]),
                edge_ev_affects_road    = _ei(graph_data["edge_ev_affects_road"]),
                edge_ev_reported_by_src = _ei(graph_data["edge_ev_reported_by_src"]),
                edge_ev_located_at_loc  = _ei(graph_data["edge_ev_located_at_loc"]),
                edge_road_near_road     = _ei(graph_data["edge_road_near_road"]),
                return_attention        = True,
            )

        # event meta-path weights: (N_event, 3) — [road_path, src_path, loc_path]
        ev_attn = attn_weights["event_meta_path_weights"]   # (N_event, 3)
        # Use road_path weight (index 0) as the event's contribution to route risk
        ev_road_attn = ev_attn[:, 0].tolist()   # how much each event attends to road context

        # Build top events list
        event_scores = []
        for i, ev in enumerate(route_events):
            if i >= len(ev_road_attn):
                break
            event_scores.append({
                "event_type": ev.get("event_type", "unknown"),
                "location":   ev.get("location") or ev.get("road_name", ""),
                "severity":   ev.get("severity", "low"),
                "source":     ev.get("source", ""),
                "confidence": round(float(ev.get("confidence", 0.5)), 3),
                "attention":  round(float(ev_road_attn[i]), 3),
            })

        event_scores.sort(key=lambda x: x["attention"], reverse=True)
        top_events = event_scores[:5]

        # Road risk from road disruption probabilities
        road_prob_tensor = attn_weights["road_embeddings"]
        # Use norm of road embedding as proxy for risk contribution
        road_ids    = meta["road_ids"]
        idx_to_road = {v: k for k, v in road_ids.items()}
        road_norms  = road_prob_tensor.norm(dim=-1).tolist()
        road_risk   = {
            idx_to_road[i]: round(float(road_norms[i]) / (max(road_norms) + 1e-8), 3)
            for i in range(len(road_norms)) if i in idx_to_road
        }

        # Natural language explanation
        if top_events:
            parts = []
            for ev in top_events[:3]:
                parts.append(
                    f"{ev['event_type'].replace('_',' ')} on {ev['location']} "
                    f"({ev['source']}, attn={ev['attention']:.2f})"
                )
            explanation = "Route risk driven by: " + "; ".join(parts)
        else:
            explanation = "No significant events found on this route."

        return {
            "top_events":       top_events,
            "road_risk_scores": road_risk,
            "explanation":      explanation,
        }

    except Exception as e:
        return {"top_events": [], "road_risk_scores": {},
                "explanation": f"Explainability error: {e}"}


def get_hgnn_status() -> dict:
    """Return HGNN readiness status for /api/hgnn-status endpoint."""
    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        torch_ok = False

    from hgnn.inference import get_inference, DEFAULT_WEIGHTS_PATH
    import os

    hgnn = get_inference()
    # Trigger lazy load so is_ready() reflects actual model state
    if not hgnn.is_ready() and torch_ok:
        hgnn._load_model()

    return {
        "available":     torch_ok,
        "ready":         hgnn.is_ready(),
        "weights_path":  DEFAULT_WEIGHTS_PATH,
        "weights_exist": os.path.exists(DEFAULT_WEIGHTS_PATH),
        "w_hgnn":        W_HGNN,
        "cascade_threshold": CASCADE_PROB_THRESHOLD,
        "message": (
            "HGNN active — confidence enhancement + per-route scoring + cascade detection"
            if hgnn.is_ready()
            else "HGNN weights not found — run 'python -m hgnn.trainer' to train"
            if torch_ok
            else "PyTorch not installed — pip install torch"
        ),
    }
