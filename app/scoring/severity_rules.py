"""
scoring/severity_rules.py — Rule-based severity correction layer
================================================================
Fixes the ONE systematic error the LLM makes on this dataset:
  → road_closure events are always labeled "low" by the LLM
  → Ground truth: closures are low / medium / high depending on road type

This is NOT replacing the LLM. The LLM correctly identifies event_type,
location, reason, and confidence. It just consistently under-weights closures.
These rules replicate what auto_label.py confirmed on 1122 events:

    Pattern                                Result    n in DB
    ─────────────────────────────────────────────────────────
    road_closure + bridge/flyover/setu    → high      ~8
    road_closure + major arterial         → medium   ~282
    road_closure + small side road        → low      ~108
    road_closure + unknown road           → medium   ~134 (default)

    congestion + "HIGH SEVERITY"          → high     ~230  (LLM already correct)
    congestion + "Queuing"                → medium   ~135  (LLM already correct)
    construction + minor road             → low       ~92  (LLM already correct)

This module is called once per event immediately after LLM extraction and
before HGNN enhancement. It is the "floor" correction; HGNN can still nudge
confidence up/down on top of these deterministic labels.

The correction is stored as `severity_source = "rule_corrected"` so the
frontend can show the provenance badge and it can be tracked in evaluate_hgnn.
"""

from __future__ import annotations

# ── Road classification tables ────────────────────────────────────────────────

# Closure on any of these → HIGH
_BRIDGE_FLYOVER_KEYWORDS = [
    "bridge", "flyover", "setu", "overpass", "expressway",
    "howrah bridge", "vidyasagar setu", "nivedita setu",
    "bally bridge", "tallah bridge", "belgharia",
    "maa flyover", "kona expressway", "nh-12", "nh-16",
]

# Closure on any of these (full name match) → HIGH
_MAJOR_ARTERIALS_HIGH = {
    "em bypass", "eastern metropolitan bypass",
    "vip road", "kona expressway", "maa flyover",
    "belgharia expressway", "nh 12", "nh 16",
    # Trunk roads confirmed HIGH from DB ground truth
    "barrackpore trunk road",
    "madhusudan banerjee road",
    "umesh mukherjee road",
    "vivekananda sarani",
    "garf main road",
    "ramlalbazara road",
}

# Closure on any of these → MEDIUM
_MAJOR_ARTERIALS_MEDIUM = {
    "jawaharlal nehru road", "chowringhee road",
    "acharya jagadish chandra bose road", "ajc bose road",
    "strand road", "rashbehari avenue", "rashbehari",
    "diamond harbour road", "bidhan sarani",
    "mahatma gandhi road", "b t road", "bt road",
    "jessore road", "barasat road", "dum dum road",
    "ultadanga road", "ultadanga main road",
    "sp mukherjee road", "s p mukherjee road",
    "park street", "camac street", "shakespeare sarani",
    "justice chandra madhab road",
    "gurusaday dutta road", "sarat bose road",
    "biplabi ganesh ghosh sarani",
    "pandit madan mohan malaviya sarani",
    "lala lajpat rai sarani",
    "abanindra nath tagore sarani",
    "sambhunath pandit street",
    "nandalal basu sarani",
    "dr sisir kumar bose sarani",
    "park street rafi ahmed kidwai road",
    "rafi ahmed kidwai road",
    "short street wood street",
}

# Road name suffix patterns that suggest a minor side road → LOW
_SIDE_ROAD_SUFFIXES = [
    " lane", " ghat road", " para road", " bazar lane",
    " nagar road", " pally road",
]


def _road_text(event: dict) -> str:
    """Combine road_name and location into one searchable string."""
    return (
        (event.get("road_name") or "") + " " +
        (event.get("location") or "")
    ).lower()


def _is_bridge_or_flyover(road_text: str) -> bool:
    return any(kw in road_text for kw in _BRIDGE_FLYOVER_KEYWORDS)


def _is_major_arterial_high(road_text: str) -> bool:
    return any(art in road_text for art in _MAJOR_ARTERIALS_HIGH)


def _is_major_arterial_medium(road_text: str) -> bool:
    return any(art in road_text for art in _MAJOR_ARTERIALS_MEDIUM)


def _is_side_road(road_text: str) -> bool:
    return any(road_text.endswith(sfx) for sfx in _SIDE_ROAD_SUFFIXES)


# ── Public API ────────────────────────────────────────────────────────────────

def correct_severity(event: dict) -> dict:
    """
    Apply rule-based severity correction to a single event dict.

    Only acts on events where the LLM is known to be wrong:
    - road_closure with severity "low" → upgrade based on road type

    All other events are returned unchanged.

    Args:
        event: event dict with keys: event_type, severity, road_name, location, ...

    Returns:
        Same dict, possibly with updated severity + severity_source field.
        Original LLM severity is preserved in severity_llm for audit.
    """
    event_type = (event.get("event_type") or "unknown").lower()
    severity   = (event.get("severity")   or "low").lower()

    # Only intervene on road_closure labeled "low" — all other LLM outputs
    # are already consistent with ground truth on this dataset.
    if event_type != "road_closure" or severity != "low":
        return event

    road = _road_text(event)

    # Rule 1: bridge / flyover / expressway → HIGH
    if _is_bridge_or_flyover(road):
        corrected = "high"
        rule      = "bridge_flyover_closure"

    # Rule 2: major arterial HIGH set → HIGH
    elif _is_major_arterial_high(road):
        corrected = "high"
        rule      = "major_arterial_high_closure"

    # Rule 3: major arterial MEDIUM set → MEDIUM
    elif _is_major_arterial_medium(road):
        corrected = "medium"
        rule      = "major_arterial_closure"

    # Rule 4: side road suffix → keep LOW
    elif _is_side_road(road):
        return event   # already correct

    # Rule 5: unknown / unclassified road closure → MEDIUM (safe default)
    # Rationale: a closure on an unclassified road is still a closure.
    # Empirically, 134/338 uncategorised closures in DB are truly medium.
    else:
        corrected = "medium"
        rule      = "unclassified_road_closure"

    # Write correction — preserve original for audit
    event["severity_llm"]    = severity
    event["severity"]        = corrected
    event["severity_source"] = f"rule:{rule}"

    # Update weighted score to reflect new severity
    from scoring.congestion_score import compute_weighted_score
    event["weighted_score"] = compute_weighted_score(
        corrected, float(event.get("confidence", 0.5))
    )

    return event


def correct_severities(events: list[dict]) -> tuple[list[dict], int]:
    """
    Apply correct_severity() to a list of events.

    Returns:
        (corrected_events, n_changed) — same list in-place, count of changes.
    """
    n_changed = 0
    for ev in events:
        original = ev.get("severity", "low")
        correct_severity(ev)
        if ev.get("severity") != original:
            n_changed += 1
    return events, n_changed
