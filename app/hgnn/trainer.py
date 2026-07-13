"""
trainer.py — Training loop for Temporal TrafficHGNN (V3)
=========================================================
Three training objectives:
  1. Road disruption regression    (MSE)          — P(disruption) per road
  2. Event confidence regression   (MSE)          — replicates + refines rule-based conf
  3. Event severity classification (CrossEntropy) — low / medium / high from graph context

Loss = W_ROAD * MSE_road + W_CONF * MSE_conf + W_SEV * CE_severity

V3 additions:
  - Verified-label boosting: events with human-verified severity get
    VERIFIED_LABEL_WEIGHT × higher loss weight in the severity CE loss.
    This anchors the model to ground truth without discarding synthetic data.
  - Val set uses ONLY verified-label events when ≥30 are available, giving
    a clean ground-truth validation signal instead of proxy labels.
  - --verified-only flag: train exclusively on verified events (use when
    you have ≥200 verified labels).

HOW TO RUN:
  cd app
  python -m hgnn.trainer                        # standard run
  python -m hgnn.trainer --epochs 300 --lr 0.001 --patience 40 --limit 1500
  python -m hgnn.trainer --verified-only        # verified labels only
"""

from __future__ import annotations

import os
import time
from typing import Optional

DEFAULT_WEIGHTS_DIR  = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_WEIGHTS_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "model.pt")

DEFAULT_EPOCHS  = 200
DEFAULT_LR      = 1e-3
DEFAULT_WD      = 1e-4
PATIENCE        = 30
VAL_SPLIT       = 0.20   # most-recent 20% of events used for validation

# Loss weights
W_ROAD  = 1.0
W_CONF  = 1.0
W_SEV   = 0.5

# Verified-label boosting
# Events with human-verified severity get this multiplier on their CE loss.
# Effect: even with 100 verified events mixed into 1500 total, verified rows
# contribute ~5× more gradient signal to the severity head.
VERIFIED_LABEL_WEIGHT = 5.0

# Min verified labels to switch val set to ground-truth-only mode
MIN_VERIFIED_FOR_GT_VAL = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_tensors(graph_data: dict) -> dict:
    import torch
    import numpy as np
    out = {}
    for k, v in graph_data.items():
        if isinstance(v, np.ndarray):
            out[k] = (torch.from_numpy(v).long()
                      if v.dtype == np.int64
                      else torch.from_numpy(v).float())
        else:
            out[k] = v
    return out


def _build_targets(graph_data: dict):
    """
    Extract regression + classification targets.

    V3: road and confidence targets are independent oracle signals from
    graph_builder.py — NOT derived from the input feature columns.
    This prevents the model from trivially reproducing its own inputs.
    """
    import torch
    road_target  = torch.from_numpy(graph_data["road_disruption_targets"]).float()
    event_target = torch.from_numpy(graph_data["event_conf_targets"]).float()
    sev_target   = torch.from_numpy(graph_data["event_sev_labels"]).long()
    return road_target, event_target, sev_target


def _build_verified_mask(graph_data: dict) -> "torch.Tensor":
    """
    Return a boolean tensor marking events that have human-verified severity.
    Falls back to all-False if no verified labels are in the graph data.
    """
    import torch
    import numpy as np
    verified = graph_data.get("event_verified_mask")
    if verified is None:
        return torch.zeros(graph_data["n_events"], dtype=torch.bool)
    return torch.from_numpy(verified.astype(bool))


def _severity_class_weights(sev_target) -> "torch.Tensor":
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    With ~67% low / ~12% medium / ~20% high the model would learn to
    always predict 'low'. Weighting corrects this so medium/high get
    proportionally more gradient signal.
    """
    import torch
    counts = torch.bincount(sev_target, minlength=3).float()
    counts = counts.clamp(min=1)                       # avoid div/0
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3              # normalise so mean weight = 1
    return weights


def _temporal_split(n_events: int, val_frac: float = VAL_SPLIT):
    """
    Returns (train_mask, val_mask) as boolean tensors over event indices.
    Events are ordered by recency (DB query is ORDER BY fetched_at DESC),
    so index 0 is newest. We hold out the OLDEST val_frac as validation
    (more realistic than random split — model trains on recent, validates on older).
    Actually: DB returns newest first, so last 20% of indices = oldest events.
    """
    import torch
    n_val   = max(1, int(n_events * val_frac))
    n_train = n_events - n_val
    train_mask = torch.zeros(n_events, dtype=torch.bool)
    val_mask   = torch.zeros(n_events, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:]   = True
    return train_mask, val_mask


def _diagnostics(graph_data: dict) -> None:
    """Print pre-training graph statistics."""
    n_ev  = graph_data["n_events"]
    n_rd  = graph_data["n_roads"]
    n_src = graph_data["n_sources"]
    n_loc = graph_data["n_locations"]

    edge_rr    = graph_data["edge_road_near_road"].shape[1]
    edge_er    = graph_data["edge_ev_affects_road"].shape[1]
    edge_es    = graph_data["edge_ev_reported_by_src"].shape[1]
    edge_el    = graph_data["edge_ev_located_at_loc"].shape[1]

    import numpy as np
    sev_labels = graph_data["event_sev_labels"]
    low_ct  = int((sev_labels == 0).sum())
    med_ct  = int((sev_labels == 1).sum())
    high_ct = int((sev_labels == 2).sum())

    print(f"  [HGNN] ── Graph Diagnostics ──────────────────────────────────")
    print(f"  [HGNN]  Nodes : {n_ev} events | {n_rd} roads | {n_src} sources | {n_loc} locations")
    print(f"  [HGNN]  Edges : ev→road={edge_er} | ev→src={edge_es} | ev→loc={edge_el} | road↔road={edge_rr}")
    print(f"  [HGNN]  Severity dist: low={low_ct} ({100*low_ct/n_ev:.0f}%) "
          f"med={med_ct} ({100*med_ct/n_ev:.0f}%) "
          f"high={high_ct} ({100*high_ct/n_ev:.0f}%)")

    if edge_rr == 0:
        print(f"  [HGNN]  WARNING: road↔road edges = 0. "
              f"No coordinates in DB → run 'python backfill_coords.py' first "
              f"for spatial graph structure.")
    else:
        print(f"  [HGNN]  Spatial graph: {edge_rr} road-road adjacency edges "
              f"(from {graph_data['event_x'].shape} event features with coords)")

    # Event feature dimension check
    ev_dim = graph_data["event_x"].shape[1]
    from hgnn.graph_builder import EVENT_FEAT_DIM, LOCATION_FEAT_DIM
    if ev_dim != EVENT_FEAT_DIM:
        print(f"  [HGNN]  WARNING: event_x has {ev_dim} dims but model expects "
              f"{EVENT_FEAT_DIM}. Check graph_builder.py.")
    loc_dim = graph_data["location_x"].shape[1] if graph_data["n_locations"] > 0 else 0
    if loc_dim and loc_dim != LOCATION_FEAT_DIM:
        print(f"  [HGNN]  WARNING: location_x has {loc_dim} dims but model expects "
              f"{LOCATION_FEAT_DIM}. Check graph_builder.py.")
    # Verify independent targets exist (V3)
    if "road_disruption_targets" not in graph_data:
        print(f"  [HGNN]  WARNING: road_disruption_targets missing — old graph_builder?")
    if "event_conf_targets" not in graph_data:
        print(f"  [HGNN]  WARNING: event_conf_targets missing — old graph_builder?")
    print(f"  [HGNN] ────────────────────────────────────────────────────────")


# ── Verified-label weighted CE loss ──────────────────────────────────────────

def _weighted_ce_loss(
    logits:        "torch.Tensor",
    targets:       "torch.Tensor",
    verified_mask: "torch.Tensor",
    class_weights: "torch.Tensor",
) -> "torch.Tensor":
    """
    CrossEntropy loss where verified events contribute VERIFIED_LABEL_WEIGHT×
    more than proxy-labeled events.

    Per-sample CE is computed without reduction, then multiplied by a
    per-sample weight (1.0 for proxy, VERIFIED_LABEL_WEIGHT for verified)
    before averaging.  This anchors the model to ground truth while still
    learning from the full dataset.
    """
    import torch
    import torch.nn.functional as F

    if logits.shape[0] == 0:
        return torch.tensor(0.0)

    # per-sample cross entropy (no reduction)
    per_sample = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")

    # sample weights: verified → VERIFIED_LABEL_WEIGHT, proxy → 1.0
    sample_w = torch.ones(logits.shape[0], device=logits.device)
    if verified_mask.any():
        sample_w[verified_mask] = VERIFIED_LABEL_WEIGHT

    return (per_sample * sample_w).mean()


# ── Main training function ────────────────────────────────────────────────────

def train(
    epochs:         int   = DEFAULT_EPOCHS,
    lr:             float = DEFAULT_LR,
    wd:             float = DEFAULT_WD,
    save_path:      str   = DEFAULT_WEIGHTS_PATH,
    db_url:         Optional[str] = None,
    verbose:        bool  = True,
    limit:          int   = 1000,
    verified_only:  bool  = False,   # train only on verified-label events
) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch required. pip install torch")

    from hgnn.graph_builder import build_graph_from_db
    from hgnn.model import build_model

    if verbose:
        print(f"  [HGNN] Loading up to {limit} events from DB ...")

    graph_data, meta = build_graph_from_db(
        route_road_names=[],
        db_url=db_url,
        limit=limit,
        verified_only=verified_only,
    )
    n_events = graph_data["n_events"]

    # ── Threshold check ───────────────────────────────────────────────────────
    if n_events < 50:
        print(f"  [HGNN] WARNING: only {n_events} events loaded. "
              f"Recommend ≥200 for meaningful training.")
    elif n_events < 200:
        print(f"  [HGNN] NOTE: {n_events} events — model will train but "
              f"may overfit. Recommend ≥200.")
    else:
        print(f"  [HGNN] Good: {n_events} events loaded for training.")

    if verbose:
        _diagnostics(graph_data)

    tensors = _to_tensors(graph_data)
    road_target, event_target, sev_target = _build_targets(graph_data)
    verified_mask = _build_verified_mask(graph_data)

    n_verified = int(verified_mask.sum().item())
    if verbose and n_verified > 0:
        print(f"  [HGNN] Verified labels: {n_verified}/{n_events} events "
              f"(weight={VERIFIED_LABEL_WEIGHT}×)")

    # ── Train / val split ─────────────────────────────────────────────────────
    # Strategy: use 80% of verified events for training (with 5× boost weight)
    # and 20% of verified events for validation (ground-truth val signal).
    # Unverified (proxy-labeled) events always go into training.
    #
    # Previous approach (all verified → val) meant the model never trained on
    # real medium/high patterns and just memorised synthetic distributions.
    if n_verified >= MIN_VERIFIED_FOR_GT_VAL:
        import torch as _torch_split
        verified_indices = _torch_split.where(verified_mask)[0]
        n_val_verified   = max(1, int(len(verified_indices) * 0.20))
        # Hold out the last 20% of verified indices as val (temporally oldest)
        val_verified_idx   = set(verified_indices[-n_val_verified:].tolist())

        val_mask   = _torch_split.zeros(n_events, dtype=_torch_split.bool)
        train_mask = _torch_split.zeros(n_events, dtype=_torch_split.bool)
        for i in range(n_events):
            if i in val_verified_idx:
                val_mask[i]   = True
            else:
                train_mask[i] = True   # all unverified + 80% verified → train
        # Make sure we have at least some training data
        if train_mask.sum().item() < 10:
            # Fall back to temporal split if almost everything is verified
            train_mask, val_mask = _temporal_split(n_events)
        val_mode = "ground-truth 20% verified (80% verified + all proxy → train)"
    else:
        train_mask, val_mask = _temporal_split(n_events)
        val_mode = "temporal proxy (no verified labels in val set)"

    n_train = int(train_mask.sum().item())
    n_val   = int(val_mask.sum().item())
    if verbose:
        print(f"  [HGNN] Train/val split: {n_train} train / {n_val} val  [{val_mode}]")

    # ── Class-weighted loss ───────────────────────────────────────────────────
    sev_weights = _severity_class_weights(sev_target)
    if verbose:
        print(f"  [HGNN] Severity class weights: "
              f"low={sev_weights[0]:.3f} "
              f"med={sev_weights[1]:.3f} "
              f"high={sev_weights[2]:.3f}")

    model     = build_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5, min_lr=1e-5
    )

    mse_fn = nn.MSELoss()
    ce_fn  = nn.CrossEntropyLoss(weight=sev_weights)   # for val (no per-sample weight)

    best_val_loss = float("inf")
    patience_ct   = 0
    history: dict[int, dict] = {}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if verbose:
        print(f"  [HGNN] Training up to {epochs} epochs  "
              f"(patience={PATIENCE}, lr={lr}, wd={wd})")
        boost_note = (f" + {VERIFIED_LABEL_WEIGHT}× verified boost"
                      if n_verified > 0 else "")
        print(f"  [HGNN] Loss = {W_ROAD}×road_MSE + {W_CONF}×conf_MSE "
              f"+ {W_SEV}×sev_CE{boost_note}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):

        # ── Train forward pass ────────────────────────────────────────────────
        model.train()
        optimiser.zero_grad()

        road_prob, event_conf, sev_logits = model(
            road_x     = tensors["road_x"],
            event_x    = tensors["event_x"],
            source_x   = tensors["source_x"],
            location_x = tensors["location_x"],
            edge_ev_affects_road    = tensors["edge_ev_affects_road"],
            edge_ev_reported_by_src = tensors["edge_ev_reported_by_src"],
            edge_ev_located_at_loc  = tensors["edge_ev_located_at_loc"],
            edge_road_near_road     = tensors["edge_road_near_road"],
        )

        loss_road = mse_fn(road_prob, road_target) if road_target.numel() > 0 else torch.tensor(0.0)
        loss_conf = mse_fn(event_conf[train_mask], event_target[train_mask]) \
                    if train_mask.any() else torch.tensor(0.0)

        # Severity loss with verified-label boosting on training events
        if train_mask.any():
            train_verified = verified_mask[train_mask]
            loss_sev = _weighted_ce_loss(
                sev_logits[train_mask], sev_target[train_mask],
                train_verified, sev_weights,
            )
        else:
            loss_sev = torch.tensor(0.0)

        loss = W_ROAD * loss_road + W_CONF * loss_conf + W_SEV * loss_sev
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        train_loss = loss.item()

        # ── Validation forward pass ───────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            _, val_conf, val_sev_logits = model(
                road_x     = tensors["road_x"],
                event_x    = tensors["event_x"],
                source_x   = tensors["source_x"],
                location_x = tensors["location_x"],
                edge_ev_affects_road    = tensors["edge_ev_affects_road"],
                edge_ev_reported_by_src = tensors["edge_ev_reported_by_src"],
                edge_ev_located_at_loc  = tensors["edge_ev_located_at_loc"],
                edge_road_near_road     = tensors["edge_road_near_road"],
            )

            val_loss_conf = mse_fn(val_conf[val_mask], event_target[val_mask]) \
                            if val_mask.any() else torch.tensor(0.0)
            val_loss_sev  = ce_fn(val_sev_logits[val_mask], sev_target[val_mask]) \
                            if val_mask.any() else torch.tensor(0.0)
            val_loss  = (W_CONF * val_loss_conf + W_SEV * val_loss_sev).item()

            val_preds   = val_sev_logits[val_mask].argmax(dim=-1)
            val_correct = (val_preds == sev_target[val_mask]).float().mean().item() \
                          if val_mask.any() else 0.0

        scheduler.step(val_loss)

        history[epoch] = {
            "train_total": train_loss,
            "train_road":  loss_road.item(),
            "train_conf":  loss_conf.item(),
            "train_sev":   loss_sev.item(),
            "val_loss":    val_loss,
            "val_acc":     val_correct,
        }

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_ct   = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ct += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            elapsed = time.time() - t0
            gt_tag = " [GT-val]" if n_verified >= MIN_VERIFIED_FOR_GT_VAL else ""
            print(
                f"  [HGNN] Epoch {epoch:4d}/{epochs}  "
                f"train={train_loss:.5f} "
                f"(road={loss_road.item():.4f} conf={loss_conf.item():.4f} sev={loss_sev.item():.4f})  "
                f"val={val_loss:.5f}  val_sev_acc={val_correct:.3f}{gt_tag}  "
                f"best_val={best_val_loss:.5f}  ({elapsed:.1f}s)"
            )

        if patience_ct >= PATIENCE:
            if verbose:
                print(f"  [HGNN] Early stop at epoch {epoch} "
                      f"(no val improvement for {PATIENCE} epochs).")
            break

    elapsed_total = time.time() - t0
    if verbose:
        print(f"\n  [HGNN] Training complete.")
        print(f"  [HGNN]   Best val loss  : {best_val_loss:.5f}")
        print(f"  [HGNN]   Weights saved  : {save_path}")
        print(f"  [HGNN]   Total time     : {elapsed_total:.1f}s  "
              f"({elapsed_total/epoch:.2f}s/epoch)")

        model.eval()
        with torch.no_grad():
            _, _, final_sev = model(
                road_x     = tensors["road_x"],
                event_x    = tensors["event_x"],
                source_x   = tensors["source_x"],
                location_x = tensors["location_x"],
                edge_ev_affects_road    = tensors["edge_ev_affects_road"],
                edge_ev_reported_by_src = tensors["edge_ev_reported_by_src"],
                edge_ev_located_at_loc  = tensors["edge_ev_located_at_loc"],
                edge_road_near_road     = tensors["edge_road_near_road"],
            )
        if val_mask.any():
            preds  = final_sev[val_mask].argmax(dim=-1)
            labels = sev_target[val_mask]
            gt_note = " (ground-truth val)" if n_verified >= MIN_VERIFIED_FOR_GT_VAL else ""
            for cls_idx, cls_name in enumerate(["low", "med", "high"]):
                cls_mask = labels == cls_idx
                if cls_mask.any():
                    acc = (preds[cls_mask] == labels[cls_mask]).float().mean().item()
                    print(f"  [HGNN]   Val acc [{cls_name:4s}]: {acc:.3f}  "
                          f"(n={cls_mask.sum().item()}){gt_note}")

    return history


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train TrafficHGNN on DB events")
    p.add_argument("--epochs",         type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",             type=float, default=DEFAULT_LR)
    p.add_argument("--patience",       type=int,   default=PATIENCE)
    p.add_argument("--save-path",      type=str,   default=DEFAULT_WEIGHTS_PATH)
    p.add_argument("--db-url",         type=str,   default=None)
    p.add_argument("--limit",          type=int,   default=1000,
                   help="Max events to load (0=all)")
    p.add_argument("--verified-only",  action="store_true",
                   help="Train only on human-verified events")
    args = p.parse_args()

    PATIENCE = args.patience
    train(
        epochs        = args.epochs,
        lr            = args.lr,
        save_path     = args.save_path,
        db_url        = args.db_url,
        limit         = args.limit if args.limit > 0 else 9999,
        verified_only = args.verified_only,
    )
