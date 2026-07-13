"""
Layer 3 — Generalized Edge Cost Function
=========================================
Implements Equation 3 from the project proposal:

    c_e(t) = c_base(t)
           + λ1 · E[τ̃_e(t)]          ← expected travel time (disruption-adjusted)
           + λ2 · Var[τ̃_e(t)]         ← travel time variability / reliability
           + λ3 · κ_e(t) · σ_e(t)     ← disruption risk  (LLM confidence × severity)
           + λ4 · CO2(e)               ← emissions per edge (gCO2)
           + λ5 · Transfers(e)         ← mode-switch penalty (minutes equivalent)

Where:
    c_base(t)   — scheduled travel time / fare from GTFS or OSMnx weight
    τ̃_e(t)     — random travel time modelled as mixture of nominal and
                  disruption distributions (Normal + Gaussian tail)
    κ_e(t)      — LLM confidence score  ∈ [0, 1]  (Layer 1)
    σ_e(t)      — severity score        ∈ {2, 5, 10} mapped to [0, 1]
    CO2(e)      — estimated CO2 for this edge × mode (g CO2-eq)
    Transfers   — count of mode switches on this edge

λ weights are user-configurable via EdgeCostWeights.
Presets: TIME_OPTIMAL, RELIABLE, ECO cover the three research scenarios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

# ── Severity score → normalised disruption magnitude ─────────────────────────
_SEVERITY_TO_FLOAT: Dict[str, float] = {
    "low":    2  / 10.0,   # 0.20
    "medium": 5  / 10.0,   # 0.50
    "high":   10 / 10.0,   # 1.00
}

# ── Emission factors (gCO2-eq / passenger-km) — IPCC / DEFRA 2023 estimates ─
# Used by compute_edge_cost when co2_emissions is not supplied directly.
EMISSION_FACTOR: Dict[str, float] = {
    "drive":  120.0,   # private car (avg)
    "taxi":    90.0,   # shared / modern fleet
    "bike":     0.0,   # zero direct emissions
    "walk":     0.0,
    "metro":    8.0,   # Kolkata metro (grid mix ~0.5 kg CO2/kWh)
    "bus":     68.0,   # diesel city bus (full load assumed)
    "train":   41.0,   # Indian Railways average
}

# ── Disruption travel-time distribution parameters ────────────────────────────
# When a disruption is present, expected travel time is modelled as:
#   E[τ̃] = τ_base × (1 + DISRUPTION_DELAY_FACTOR × disruption_risk)
# And variance as:
#   Var[τ̃] = (τ_base × DISRUPTION_STD_FACTOR × disruption_risk) ** 2
DISRUPTION_DELAY_FACTOR = 0.60   # 60% extra travel time at full disruption risk
DISRUPTION_STD_FACTOR   = 0.35   # σ = 35% of base time at full risk


@dataclass
class EdgeCostWeights:
    """
    User preference weights for the generalized cost function (Equation 3).

    Attributes:
        lambda1: Weight on expected (disruption-adjusted) travel time.
        lambda2: Weight on travel time variance — higher = more risk-averse.
        lambda3: Weight on disruption risk term  (κ × σ).
        lambda4: Weight on CO2 emissions.
        lambda5: Transfer penalty per mode switch (minutes equivalent).
    """
    lambda1: float = 1.0   # expected travel time
    lambda2: float = 0.5   # travel time variance (reliability)
    lambda3: float = 1.0   # disruption risk
    lambda4: float = 0.2   # CO2 emissions (gCO2 → minutes equivalent at 1 g = 0.001 min)
    lambda5: float = 2.0   # transfer penalty (minutes equivalent per switch)


# ── Ready-made presets ────────────────────────────────────────────────────────

TIME_OPTIMAL = EdgeCostWeights(
    lambda1=2.0, lambda2=0.0, lambda3=0.5, lambda4=0.0, lambda5=1.0
)
"""Minimise total travel time; ignore variance and emissions."""

RELIABLE = EdgeCostWeights(
    lambda1=1.0, lambda2=1.5, lambda3=1.5, lambda4=0.1, lambda5=2.0
)
"""Prefer routes with low disruption risk and low variance — commuter default."""

ECO = EdgeCostWeights(
    lambda1=1.0, lambda2=0.5, lambda3=0.5, lambda4=2.0, lambda5=1.0
)
"""Minimise CO2 emissions; useful for sustainability-focused routing."""


# ── Core implementation ───────────────────────────────────────────────────────

def compute_edge_cost(
    c_base:               float,
    expected_travel_time: float,
    travel_time_variance: float,
    disruption_risk:      float,
    co2_emissions:        float,
    transfer_penalty:     float,
    weights:              Optional[EdgeCostWeights] = None,
) -> float:
    """
    Compute generalized cost for one edge using Equation 3.

    Args:
        c_base:               Base scheduled travel time in minutes (or fare units).
        expected_travel_time: E[τ̃_e(t)] — disruption-adjusted mean travel time.
        travel_time_variance: Var[τ̃_e(t)] — variance (minutes²) for reliability.
        disruption_risk:      κ_e(t) × σ_e(t) — product of LLM confidence and
                              severity magnitude ∈ [0, 1].
        co2_emissions:        CO2 contribution in grams for this edge.
        transfer_penalty:     Number of mode switches on this edge.
        weights:              λ preference weights. Defaults to RELIABLE preset.

    Returns:
        Scalar generalized edge cost c_e(t) ≥ 0.

    Notes:
        - All time values should use the same unit (minutes recommended).
        - co2_emissions is scaled by 0.001 to convert grams → minutes-equivalent
          so that a 120 g/km drive over 1 km adds ~0.12 "penalty minutes".
        - The variance term uses its square-root (standard deviation) to keep
          units consistent with travel time in minutes.
    """
    if weights is None:
        weights = RELIABLE

    # Clamp inputs to valid ranges
    disruption_risk    = max(0.0, min(1.0, disruption_risk))
    travel_time_variance = max(0.0, travel_time_variance)
    co2_emissions      = max(0.0, co2_emissions)
    transfer_penalty   = max(0.0, transfer_penalty)

    # Travel time standard deviation keeps units in minutes
    travel_time_std = math.sqrt(travel_time_variance)

    # Equation 3:
    #   c_e(t) = c_base
    #          + λ1 · E[τ̃]
    #          + λ2 · std(τ̃)          ← std (not variance) to preserve units
    #          + λ3 · κ·σ
    #          + λ4 · CO2 × 0.001     ← g → minutes-equivalent scaling
    #          + λ5 · Transfers
    cost = (
        c_base
        + weights.lambda1 * expected_travel_time
        + weights.lambda2 * travel_time_std
        + weights.lambda3 * disruption_risk
        + weights.lambda4 * co2_emissions * 0.001
        + weights.lambda5 * transfer_penalty
    )
    return max(0.0, cost)


def compute_edge_cost_from_event(
    c_base:           float,
    severity:         str  = "low",
    confidence:       float = 0.0,
    transfer_count:   int   = 0,
    distance_km:      float = 0.0,
    mode:             str   = "drive",
    weights:          Optional[EdgeCostWeights] = None,
) -> float:
    """
    Convenience wrapper: derive all cost components from raw event fields.

    This is the entry-point for the route_engine — it maps Layer 1 outputs
    (severity string + confidence float) directly into the generalized cost
    without requiring the caller to pre-compute distributions.

    Travel time distributions are derived analytically:
      E[τ̃] = c_base × (1 + DISRUPTION_DELAY_FACTOR × risk)
      Var[τ̃] = (c_base × DISRUPTION_STD_FACTOR × risk) ²

    CO2 is estimated from mode × distance using EMISSION_FACTOR table.

    Args:
        c_base:         Scheduled travel time in minutes (OSMnx 'travel_time').
        severity:       "low" | "medium" | "high" — from Layer 1.
        confidence:     LLM confidence κ ∈ [0, 1] — from Layer 1.
        transfer_count: Mode-switch count on this edge (0 for single-mode).
        distance_km:    Edge length in km — used for CO2 estimation.
        mode:           Transport mode key (see EMISSION_FACTOR).
        weights:        λ preference weights. Defaults to RELIABLE.

    Returns:
        Scalar generalized edge cost.
    """
    sev_float      = _SEVERITY_TO_FLOAT.get(severity, 0.0)
    disruption_risk = confidence * sev_float   # κ × σ ∈ [0, 1]

    expected_travel_time = c_base * (1.0 + DISRUPTION_DELAY_FACTOR * disruption_risk)
    variance             = (c_base * DISRUPTION_STD_FACTOR * disruption_risk) ** 2

    emission_rate        = EMISSION_FACTOR.get(mode.lower(), EMISSION_FACTOR["drive"])
    co2_emissions        = emission_rate * distance_km   # g CO2

    return compute_edge_cost(
        c_base               = c_base,
        expected_travel_time = expected_travel_time,
        travel_time_variance = variance,
        disruption_risk      = disruption_risk,
        co2_emissions        = co2_emissions,
        transfer_penalty     = float(transfer_count),
        weights              = weights,
    )


def severity_to_float(severity: str) -> float:
    """
    Map severity label to normalised float ∈ [0, 1].

    Used externally by scoring modules that need a scalar disruption magnitude.
    """
    return _SEVERITY_TO_FLOAT.get(severity, 0.0)
