"""
metrics.py
----------
Evaluation metrics for the anomaly detection experiment.

Computes:
  - Detection Accuracy, Precision, Recall, F1-score
  - False Positive Rate, False Negative Rate
  - 95% Wilson confidence intervals (matching Table 8 in paper)
  - Layer-wise detection performance (Table 9)
  - Root cause distribution (Table 10)
  - Temporal detection analysis (Table 11)
"""

import numpy as np
from scipy import stats


def wilson_confidence_interval(successes, total, confidence=0.95):
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) as percentages.
    """
    if total == 0:
        return 0.0, 0.0

    p     = successes / total
    z     = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / total
    centre_adj = p + z**2 / (2 * total)
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    lower = (centre_adj - margin) / denom
    upper = (centre_adj + margin) / denom
    return round(max(lower, 0.0) * 100, 1), round(min(upper, 1.0) * 100, 1)


def compute_overall_metrics(results):
    """
    Compute overall detection performance metrics (Table 8).

    Args:
        results: list of window result dicts from FusionEngine

    Returns:
        dict with accuracy, precision, recall, F1, FPR, FNR + CIs
    """
    TP = FP = TN = FN = 0

    for r in results:
        predicted = r["is_anomaly"]
        actual    = r["ground_truth"]
        if actual is None:
            continue
        if predicted and actual:
            TP += 1
        elif predicted and not actual:
            FP += 1
        elif not predicted and not actual:
            TN += 1
        elif not predicted and actual:
            FN += 1

    total        = TP + FP + TN + FN
    n_positive   = TP + FN   # actual anomalies
    n_negative   = TN + FP   # actual normals
    n_detected   = TP + FP   # predicted as anomaly

    accuracy  = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP)    if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN)    if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr       = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # Wilson CIs
    acc_ci  = wilson_confidence_interval(TP + TN, total)
    prec_ci = wilson_confidence_interval(TP, TP + FP) if (TP + FP) > 0 else (0.0, 100.0)
    rec_ci  = wilson_confidence_interval(TP, TP + FN) if (TP + FN) > 0 else (0.0, 100.0)
    f1_ci   = wilson_confidence_interval(
        int(f1 * total), total)   # approximate CI for F1
    fpr_ci  = wilson_confidence_interval(FP, FP + TN) if (FP + TN) > 0 else (0.0, 100.0)
    fnr_ci  = wilson_confidence_interval(FN, FN + TP) if (FN + TP) > 0 else (0.0, 100.0)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "total":      total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "accuracy":   round(accuracy  * 100, 1),
        "precision":  round(precision * 100, 1),
        "recall":     round(recall    * 100, 1),
        "f1":         round(f1        * 100, 1),
        "fpr":        round(fpr       * 100, 1),
        "fnr":        round(fnr       * 100, 1),
        "accuracy_ci":  f"[{acc_ci[0]}%, {acc_ci[1]}%]",
        "precision_ci": f"[{prec_ci[0]}%, {prec_ci[1]}%]",
        "recall_ci":    f"[{rec_ci[0]}%, {rec_ci[1]}%]",
        "f1_ci":        f"[{f1_ci[0]}%, {f1_ci[1]}%]",
        "fpr_ci":       f"[{fpr_ci[0]}%, {fpr_ci[1]}%]",
        "fnr_ci":       f"[{fnr_ci[0]}%, {fnr_ci[1]}%]",
    }


def compute_layerwise_metrics(results):
    """
    Layer-wise detection performance (Table 9).
    Only considers anomaly windows (ground_truth=True).
    """
    anomaly_windows = [r for r in results if r.get("ground_truth") is True]
    n_anomaly = len(anomaly_windows)

    layers = {
        "metrics": {"detected": 0, "confidences": []},
        "logs":    {"detected": 0, "confidences": []},
        "traces":  {"detected": 0, "confidences": []},
        "fusion":  {"detected": 0, "confidences": []},
    }

    for r in anomaly_windows:
        # Metrics detected if metrics_rate > 0
        if r["metrics_rate"] > 0:
            layers["metrics"]["detected"] += 1
            layers["metrics"]["confidences"].append(min(r["sm"] * 100, 100.0))

        # Logs detected if logs_rate > 0
        if r["logs_rate"] > 0:
            layers["logs"]["detected"] += 1
            layers["logs"]["confidences"].append(r["sl"] * 100)

        # Traces detected if traces_rate > 0
        if r["traces_rate"] > 0:
            layers["traces"]["detected"] += 1
            layers["traces"]["confidences"].append(r["st"] * 100)

        # Fusion detected if is_anomaly
        if r["is_anomaly"]:
            layers["fusion"]["detected"] += 1
            layers["fusion"]["confidences"].append(r["confidence_pct"])

    layer_results = {}
    for name, data in layers.items():
        n_det  = data["detected"]
        confs  = data["confidences"]
        layer_results[name] = {
            "anomaly_windows":  n_anomaly,
            "detection_count":  n_det,
            "detection_rate":   round(n_det / n_anomaly * 100, 1) if n_anomaly > 0 else 0.0,
            "avg_confidence":   round(float(np.mean(confs)), 1) if confs else 0.0,
        }
    return layer_results


def compute_root_cause_distribution(results):
    """
    Root cause distribution (Table 10).
    """
    anomaly_windows = [r for r in results if r.get("is_anomaly")]
    n_anomaly = len(anomaly_windows)

    counts = {}
    for r in anomaly_windows:
        rc = r.get("root_cause", "Unknown pattern")
        counts[rc] = counts.get(rc, 0) + 1

    distribution = {}
    for rc, count in sorted(counts.items(), key=lambda x: -x[1]):
        distribution[rc] = {
            "frequency":  count,
            "percentage": round(count / n_anomaly * 100, 1) if n_anomaly > 0 else 0.0,
        }

    return distribution


def compute_temporal_analysis(results):
    """
    Temporal detection analysis (Table 11).
    Groups windows into Normal / Early Anomaly / Established Anomaly.
    """
    normal_windows        = [r for r in results if not r.get("ground_truth")]
    early_anomaly_windows = [r for r in results
                              if r.get("ground_truth") and r["window"] in (12, 13)]
    established_windows   = [r for r in results
                              if r.get("ground_truth") and r["window"] >= 14]

    def group_stats(windows, label):
        if not windows:
            return {"label": label, "anomaly_rate": "0%",
                    "avg_confidence": "N/A", "primary_detector": "N/A"}
        anomaly_rate  = np.mean([r["is_anomaly"] for r in windows]) * 100
        avg_conf      = np.mean([r["confidence_pct"] for r in windows])
        # Dominant layer: most frequent dominant_layer value
        layers = [r["dominant_layer"] for r in windows if r["dominant_layer"] != "—"]
        primary = max(set(layers), key=layers.count).capitalize() if layers else "N/A"
        return {
            "label":           label,
            "anomaly_rate":    f"{anomaly_rate:.0f}%",
            "avg_confidence":  f"{avg_conf:.1f}%",
            "primary_detector": primary
        }

    return [
        group_stats(normal_windows,        "1-11 (Normal)"),
        group_stats(early_anomaly_windows,  "12-13 (Early Anomaly)"),
        group_stats(established_windows,    "14-20 (Established)"),
    ]
