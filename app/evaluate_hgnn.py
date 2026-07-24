"""
evaluate_hgnn.py — HGNN evaluation against ground-truth verified labels
========================================================================
Requires verified_severity labels (written by auto_label.py / label_events.py).
Run from app/ directory:

  python evaluate_hgnn.py                     # full eval report
  python evaluate_hgnn.py --ablation          # + ablation: spatial edges disabled
  python evaluate_hgnn.py --min-labels 30     # lower threshold (small label sets)
  python evaluate_hgnn.py --export results/   # save CSV + write threshold file
  python evaluate_hgnn.py --blind             # blind eval: zero out LLM severity feat

Reports:
  1. Label coverage summary
  2. LLM baseline metrics  (precision / recall / F1 per class, macro avg)
  3. HGNN model metrics    (same format — compare directly)
  4. Per-class improvement over LLM baseline
  5. Confusion matrices for both
  6. Ablation (optional): HGNN with road-road spatial edges disabled
  7. Confidence calibration: HGNN confidence vs. verified label correctness

KEY DESIGN NOTES
----------------
The HGNN input feature event_x[:,0] encodes severity/10.  In standard mode,
this is the LLM-assigned severity — so if LLM says "low" the model sees 0.2.

Two evaluation modes:
  1. Standard (default): severity feature = LLM severity.
     Measures: "does HGNN improve on the LLM signal given graph context?"
     Expect HGNN ≈ LLM on events LLM got right, and partial correction of errors.

  2. Blind (--blind): severity feature zeroed out.
     Measures: "can HGNN predict severity from graph structure alone?"
     This isolates the pure graph learning signal.

The standard mode is the relevant one for the real system (LLM labels always exist).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))

DB_URL      = "sqlite:///traffic_events.db"
SEV_CLASSES = ["low", "medium", "high"]
SEV_IDX     = {s: i for i, s in enumerate(SEV_CLASSES)}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _precision_recall_f1(
    y_true: list[int], y_pred: list[int], n_classes: int = 3
) -> dict:
    """Compute per-class and macro-averaged P / R / F1."""
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    results = {}
    for i, cls in enumerate(SEV_CLASSES):
        p  = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        r  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * p * r / (p + r)     if (p + r) > 0          else 0.0
        results[cls] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": tp[i] + fn[i]
        }

    macro_p  = sum(v["precision"] for v in results.values()) / n_classes
    macro_r  = sum(v["recall"]    for v in results.values()) / n_classes
    macro_f1 = sum(v["f1"]        for v in results.values()) / n_classes
    acc      = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

    results["macro"]    = {
        "precision": round(macro_p, 4), "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4), "support": len(y_true)
    }
    results["accuracy"] = round(acc, 4)
    return results


def _confusion_matrix(
    y_true: list[int], y_pred: list[int], n_classes: int = 3
) -> list[list[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1
    return cm


def _print_metrics(
    name: str,
    metrics: dict,
    show_improvement: dict | None = None,
) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")
    print(f"  {'Class':10s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print(f"  {'─'*60}")
    for cls in SEV_CLASSES + ["macro"]:
        m = metrics.get(cls, {})
        imp_str = ""
        if show_improvement and cls in show_improvement:
            delta   = show_improvement[cls]
            imp_str = f"  (+{delta:.4f})" if delta >= 0 else f"  ({delta:.4f})"
        print(f"  {cls:10s}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  "
              f"{m['f1']:>8.4f}  {m['support']:>8}{imp_str}")
    print(f"  {'─'*60}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")


def _print_confusion(name: str, cm: list[list[int]]) -> None:
    print(f"\n  Confusion Matrix — {name}")
    print(f"  (rows = true, cols = predicted)")
    header = f"  {'':12s}" + "".join(f"  {cls:>8}" for cls in SEV_CLASSES)
    print(header)
    for i, cls in enumerate(SEV_CLASSES):
        row_str = f"  {cls:12s}" + "".join(
            f"  {cm[i][j]:>8}" for j in range(len(SEV_CLASSES))
        )
        print(row_str)


# ── HGNN inference ────────────────────────────────────────────────────────────

def _run_hgnn_on_events(
    events:          list[dict],
    road_names:      list[str],
    disable_spatial: bool = False,
    blind_severity:  bool = False,
) -> list[str] | None:
    """
    Run HGNN inference on a list of event dicts.

    Parameters
    ----------
    disable_spatial : zero out road-road edges (ablation study).
    blind_severity  : zero out event_x[:,0] (severity feature) — measures
                      how much HGNN relies on the LLM severity signal vs.
                      pure graph structure.

    Returns list of predicted severity strings, or None if HGNN unavailable.
    """
    try:
        import torch
        import numpy as np
        from hgnn.graph_builder import build_graph_from_events
        from hgnn.inference import get_inference

        hgnn = get_inference()
        if not hgnn.is_ready():
            hgnn._load_model()
        if not hgnn.is_ready():
            return None

        graph_data, _ = build_graph_from_events(events, road_names)

        if disable_spatial:
            graph_data["edge_road_near_road"] = np.zeros((2, 0), dtype=np.int64)

        if blind_severity:
            # Zero out event_x[:,0] so the model cannot use the LLM severity
            # as a shortcut — tests pure graph reasoning ability
            ev_x = graph_data["event_x"].copy()
            ev_x[:, 0] = 0.0
            graph_data["event_x"] = ev_x

        def _t(arr):
            return torch.from_numpy(arr).float()

        def _ei(arr):
            if arr.shape[1] == 0:
                return torch.zeros((2, 0), dtype=torch.long)
            return torch.from_numpy(arr).long()

        with torch.no_grad():
            _, _, sev_logits = hgnn._model(
                road_x                  = _t(graph_data["road_x"]),
                event_x                 = _t(graph_data["event_x"]),
                source_x                = _t(graph_data["source_x"]),
                location_x              = _t(graph_data["location_x"]),
                edge_ev_affects_road    = _ei(graph_data["edge_ev_affects_road"]),
                edge_ev_reported_by_src = _ei(graph_data["edge_ev_reported_by_src"]),
                edge_ev_located_at_loc  = _ei(graph_data["edge_ev_located_at_loc"]),
                edge_road_near_road     = _ei(graph_data["edge_road_near_road"]),
            )

        preds = sev_logits.argmax(dim=-1).tolist()
        return [SEV_CLASSES[p] for p in preds]

    except Exception as e:
        print(f"  [HGNN] Inference error: {e}")
        import traceback; traceback.print_exc()
        return None


# ── Calibration ───────────────────────────────────────────────────────────────

def _calibration_report(
    events:       list[dict],
    hgnn_preds:   list[str],
    true_labels:  list[str],
) -> None:
    bins       = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    confs      = [float(ev.get("confidence", 0.5)) for ev in events]

    print(f"\n  Confidence Calibration (HGNN confidence vs. accuracy on verified labels)")
    print(f"  {'Conf bucket':12s}  {'Count':>6}  {'Accuracy':>10}  {'Calibrated?':>18}")
    print(f"  {'─'*56}")

    for (lo, hi), lbl in zip(bins, bin_labels):
        idxs = [i for i, c in enumerate(confs) if lo <= c < hi]
        if not idxs:
            continue
        correct = sum(1 for i in idxs if hgnn_preds[i] == true_labels[i])
        acc     = correct / len(idxs)
        mid     = (lo + hi) / 2
        diff    = abs(acc - mid)
        cal_str = (
            "✓ good"          if diff < 0.15 else
            "⚠ over-conf"     if acc < mid   else
            "⚠ under-conf"
        )
        print(f"  {lbl:12s}  {len(idxs):>6}  {acc:>10.3f}  {cal_str:>18}")


# ── LLM-error correction analysis ────────────────────────────────────────────

def _correction_analysis(
    llm_preds:   list[str],
    hgnn_preds:  list[str],
    true_labels: list[str],
) -> None:
    """
    On the subset where LLM was wrong, how often does HGNN correct it?
    This is the most meaningful signal for the production system — the HGNN
    only matters on events where LLM made a mistake.
    """
    wrong_idxs   = [i for i, (l, t) in enumerate(zip(llm_preds, true_labels)) if l != t]
    if not wrong_idxs:
        print(f"\n  LLM correction: no LLM errors in this set (perfect LLM baseline)")
        return

    corrected    = sum(1 for i in wrong_idxs if hgnn_preds[i] == true_labels[i])
    still_wrong  = len(wrong_idxs) - corrected

    # Breakdown by error type
    correction_types: Counter = Counter()
    for i in wrong_idxs:
        if hgnn_preds[i] == true_labels[i]:
            correction_types[f"{llm_preds[i]}→{true_labels[i]} (✓ fixed)"] += 1
        else:
            correction_types[f"{llm_preds[i]}→{hgnn_preds[i]} (✗ still {true_labels[i]})"] += 1

    print(f"\n  LLM-Error Correction (on {len(wrong_idxs)} events where LLM was wrong):")
    print(f"  HGNN corrected: {corrected}/{len(wrong_idxs)} "
          f"({100*corrected/len(wrong_idxs):.1f}%)")
    print(f"  Still wrong   : {still_wrong}/{len(wrong_idxs)} "
          f"({100*still_wrong/len(wrong_idxs):.1f}%)")
    print(f"  Error types   :")
    for k, v in sorted(correction_types.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")


# ── Export helpers ────────────────────────────────────────────────────────────

def _export_csv(
    events:       list[dict],
    llm_preds:    list[str],
    hgnn_preds:   list[str] | None,
    true_labels:  list[str],
    export_dir:   str,
) -> None:
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, "hgnn_eval_results.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("id,event_type,road_name,true_severity,llm_severity,hgnn_severity,"
                "llm_correct,hgnn_correct,confidence\n")
        for i, ev in enumerate(events):
            llm_c  = int(llm_preds[i] == true_labels[i])
            hgnn_c = int(hgnn_preds[i] == true_labels[i]) if hgnn_preds else ""
            h_pred = hgnn_preds[i] if hgnn_preds else ""
            rn     = (ev.get("road_name") or "").replace(",", "_")
            f.write(f"{ev.get('id','')},{ev.get('event_type','')},{rn},"
                    f"{true_labels[i]},{llm_preds[i]},{h_pred},"
                    f"{llm_c},{hgnn_c},{ev.get('confidence', 0.5):.4f}\n")
    print(f"\n  CSV saved → {path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    min_labels:    int  = 30,
    run_ablation:  bool = False,
    export_dir:    str | None = None,
    blind_mode:    bool = False,
) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_URL, echo=False)

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(traffic_events)"))}
        if "verified_severity" not in existing:
            print("\n  ✗ No verified_severity column found.")
            print("  Run 'python auto_label.py' first to create ground-truth labels.")
            return

        rows = conn.execute(text(
            "SELECT id, event_type, severity, confidence, road_name, location, "
            "reason, raw_text, source, verified_severity, lat, lon "
            "FROM traffic_events "
            "WHERE verified_severity IS NOT NULL AND raw_text NOT LIKE '[SYNTHETIC]%' "
            "ORDER BY id DESC"
        )).fetchall()

    n = len(rows)
    if n < min_labels:
        print(f"\n  ✗ Only {n} verified labels found (need ≥{min_labels}).")
        print(f"  Run 'python auto_label.py' or 'python label_events.py' to add labels.")
        print(f"  Use --min-labels {n} to force evaluation with current count.")
        return

    print(f"\n{'═'*62}")
    print(f"  HGNN Evaluation Report")
    print(f"{'═'*62}")
    print(f"  Ground-truth labels : {n}")
    if blind_mode:
        print(f"  Mode                : BLIND (LLM severity feature zeroed out)")
    else:
        print(f"  Mode                : STANDARD (LLM severity feature present)")

    # ── Build event dicts ─────────────────────────────────────────────────
    events:       list[dict] = []
    true_labels:  list[str]  = []
    llm_preds:    list[str]  = []

    for row in rows:
        (ev_id, ev_type, llm_sev, conf, road, loc, reason,
         raw_text, source, verified_sev, lat, lon) = row

        events.append({
            "id":         ev_id,
            "event_type": ev_type   or "unknown",
            "severity":   llm_sev   or "low",    # LLM label — input to HGNN
            "confidence": float(conf or 0.5),
            "road_name":  road,
            "location":   loc,
            "reason":     reason,
            "source":     source or "unknown",
            "is_recent":  True,
            "lat":        float(lat) if lat  is not None else None,
            "lon":        float(lon) if lon  is not None else None,
        })
        true_labels.append(verified_sev or "low")
        llm_preds.append(llm_sev or "low")

    road_names = list({ev["road_name"] for ev in events if ev["road_name"]})

    # ── Label distributions ───────────────────────────────────────────────
    dist_true = Counter(true_labels)
    dist_llm  = Counter(llm_preds)
    print(f"\n  True label dist  : {dict(sorted(dist_true.items()))}")
    print(f"  LLM label dist   : {dict(sorted(dist_llm.items()))}")

    y_true = [SEV_IDX.get(s, 0) for s in true_labels]
    y_llm  = [SEV_IDX.get(s, 0) for s in llm_preds]

    # ── 1. LLM Baseline ──────────────────────────────────────────────────
    llm_metrics = _precision_recall_f1(y_true, y_llm)
    _print_metrics("LLM Baseline (no HGNN)", llm_metrics)
    _print_confusion("LLM Baseline", _confusion_matrix(y_true, y_llm))

    # Explain what the LLM-baseline confusion means
    wrong_count = sum(1 for l, t in zip(llm_preds, true_labels) if l != t)
    wrong_dist  = Counter(
        f"{l}→{t}" for l, t in zip(llm_preds, true_labels) if l != t
    )
    print(f"\n  LLM error breakdown ({wrong_count} errors):")
    for pattern, cnt in sorted(wrong_dist.items(), key=lambda x: -x[1]):
        print(f"    {pattern}: {cnt}")

    # ── 2. HGNN (standard or blind) ──────────────────────────────────────
    mode_label = "BLIND — no severity feature" if blind_mode else "with LLM severity feature"
    print(f"\n  Running HGNN inference ({mode_label}) on {n} events...")
    hgnn_preds = _run_hgnn_on_events(
        events, road_names,
        blind_severity = blind_mode,
    )

    if hgnn_preds is None:
        print("\n  ✗ HGNN not available — train model first: python -m hgnn.trainer")
        return

    y_hgnn = [SEV_IDX.get(s, 0) for s in hgnn_preds]
    hgnn_metrics = _precision_recall_f1(y_true, y_hgnn)
    improvement  = {
        cls: round(hgnn_metrics[cls]["f1"] - llm_metrics[cls]["f1"], 4)
        for cls in SEV_CLASSES + ["macro"]
    }
    _print_metrics(
        f"HGNN ({mode_label})",
        hgnn_metrics,
        show_improvement = improvement,
    )
    _print_confusion("HGNN", _confusion_matrix(y_true, y_hgnn))

    # Distribution comparison
    dist_hgnn = Counter(hgnn_preds)
    print(f"\n  HGNN prediction dist : {dict(sorted(dist_hgnn.items()))}")
    print(f"  LLM prediction dist  : {dict(sorted(dist_llm.items()))}")
    print(f"  True label dist      : {dict(sorted(dist_true.items()))}")

    _correction_analysis(llm_preds, hgnn_preds, true_labels)
    _calibration_report(events, hgnn_preds, true_labels)

    # ── 3. Blind mode comparison (if running standard mode) ───────────────
    if not blind_mode:
        print(f"\n  Running BLIND mode comparison (severity feature zeroed)...")
        blind_preds = _run_hgnn_on_events(
            events, road_names, blind_severity=True
        )
        if blind_preds:
            y_blind      = [SEV_IDX.get(s, 0) for s in blind_preds]
            blind_metrics = _precision_recall_f1(y_true, y_blind)
            blind_imp    = {
                cls: round(hgnn_metrics[cls]["f1"] - blind_metrics[cls]["f1"], 4)
                for cls in SEV_CLASSES + ["macro"]
            }
            _print_metrics(
                "HGNN — BLIND (no severity feat) vs. standard",
                blind_metrics,
            )
            print(f"\n  LLM-severity feature contribution (standard − blind):")
            for cls in SEV_CLASSES + ["macro"]:
                delta = blind_imp[cls]
                print(f"    {cls:10s}: {'+' if delta>=0 else ''}{delta:+.4f} F1")

    # ── 4. Ablation: no spatial edges ────────────────────────────────────
    if run_ablation:
        print(f"\n  Running ablation: road-road spatial edges DISABLED...")
        ablation_preds = _run_hgnn_on_events(
            events, road_names,
            disable_spatial = True,
            blind_severity  = blind_mode,
        )
        if ablation_preds:
            y_ablation       = [SEV_IDX.get(s, 0) for s in ablation_preds]
            ablation_metrics = _precision_recall_f1(y_true, y_ablation)
            spatial_imp      = {
                cls: round(hgnn_metrics[cls]["f1"] - ablation_metrics[cls]["f1"], 4)
                for cls in SEV_CLASSES + ["macro"]
            }
            _print_metrics(
                "HGNN — spatial edges DISABLED (ablation)",
                ablation_metrics,
            )
            print(f"\n  Spatial edge contribution (full − no-spatial):")
            for cls in SEV_CLASSES + ["macro"]:
                delta = spatial_imp[cls]
                print(f"    {cls:10s}: {'+' if delta>=0 else ''}{delta:+.4f} F1")

    # ── 5. Summary + threshold decision ──────────────────────────────────
    macro_delta = improvement.get("macro", 0)
    delta_acc   = hgnn_metrics["accuracy"] - llm_metrics["accuracy"]

    print(f"\n{'═'*62}")
    print(f"  Summary")
    print(f"{'═'*62}")
    print(f"  Labeled events         : {n}")
    print(f"  LLM baseline accuracy  : {llm_metrics['accuracy']:.4f}")
    print(f"  HGNN accuracy          : {hgnn_metrics['accuracy']:.4f}")
    print(f"  Accuracy delta         : {'+' if delta_acc>=0 else ''}{delta_acc:.4f}")
    print(f"  Macro F1 delta         : {'+' if macro_delta>=0 else ''}{macro_delta:.4f}")

    # Correction rate on LLM errors (the metric that matters in production)
    wrong_idxs = [i for i, (l, t) in enumerate(zip(llm_preds, true_labels)) if l != t]
    if wrong_idxs:
        correction_rate = sum(
            1 for i in wrong_idxs if hgnn_preds[i] == true_labels[i]
        ) / len(wrong_idxs)
        print(f"  LLM-error correction   : {correction_rate:.4f} "
              f"({int(correction_rate*len(wrong_idxs))}/{len(wrong_idxs)} LLM errors fixed)")

    print()

    # Threshold decision and auto-write
    threshold_file = os.path.join(
        os.path.dirname(__file__), "hgnn", "weights", "eval_threshold.txt"
    )
    hgnn_acc = hgnn_metrics["accuracy"]

    if hgnn_acc >= 0.85:
        new_threshold = 0.70
        msg = f"✓ HGNN accuracy ≥ 0.85 — activating severity override (threshold→{new_threshold})"
    elif hgnn_acc >= 0.80:
        new_threshold = 0.75
        msg = f"✓ HGNN accuracy ≥ 0.80 — lowering threshold (0.85→{new_threshold})"
    elif hgnn_acc >= 0.70:
        new_threshold = 0.85
        msg = "⚡ HGNN accuracy 0.70–0.80 — keeping conservative threshold (0.85)"
    else:
        new_threshold = None
        msg = "⚠ HGNN accuracy < 0.70 — severity override stays disabled"

    print(f"  {msg}")

    if export_dir and new_threshold is not None:
        try:
            os.makedirs(os.path.dirname(threshold_file), exist_ok=True)
            with open(threshold_file, "w") as f:
                f.write(str(new_threshold))
            print(f"    Threshold {new_threshold} written → {threshold_file}")
            print(f"    Severity correction head is now ACTIVE in integration.py")
        except Exception as e:
            print(f"    Could not write threshold: {e}")
    elif new_threshold is None and os.path.exists(threshold_file):
        # Accuracy dropped — remove the threshold file to disable override
        try:
            os.remove(threshold_file)
            print(f"    Removed {threshold_file} — severity override disabled")
        except Exception:
            pass

    if export_dir:
        _export_csv(events, llm_preds, hgnn_preds, true_labels, export_dir)

    print(f"\n{'═'*62}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HGNN severity predictions against verified labels"
    )
    parser.add_argument("--min-labels", type=int,  default=30,
                        help="Minimum verified labels required (default 30)")
    parser.add_argument("--ablation",   action="store_true",
                        help="Run ablation study without spatial edges")
    parser.add_argument("--blind",      action="store_true",
                        help="Blind eval: zero out LLM severity feature (tests pure graph reasoning)")
    parser.add_argument("--export",     type=str,  default=None,
                        help="Directory for CSV export and threshold file write")
    args = parser.parse_args()

    evaluate(
        min_labels   = args.min_labels,
        run_ablation = args.ablation,
        export_dir   = args.export,
        blind_mode   = args.blind,
    )


if __name__ == "__main__":
    main()
