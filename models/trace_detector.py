"""
trace_detector.py
-----------------
Layer 3 — Trace Anomaly Detector: Distance-Based Classifier

Design (from paper Section III-C and Table 5):
  Features:  5 per trace
             - avg_duration_ms (normalised)
             - error_rate
             - span_count (normalised)
             - service_count (normalised)
             - max_duration_ms (normalised)

  Training:  Build reference set from normal traces only.
             Threshold = 95th percentile of minimum distances in training set.

  Inference: For each trace, compute Euclidean distance to every reference vector.
             Take minimum distance as the anomaly score.
             If min_distance > threshold → anomaly.

  Anomaly thresholds from Table 5:
    avg_duration > 2500ms, error_rate > 30%,
    span_count > 15, service_count > 5, max_duration > 4000ms
"""

import numpy as np
import json
import os
from sklearn.metrics.pairwise import euclidean_distances

# Feature normalisation constants (from Table 5 anomaly thresholds)
NORM_CONSTANTS = {
    "avg_duration_ms":  5000.0,
    "error_rate":       1.0,
    "span_count":       30.0,
    "service_count":    10.0,
    "max_duration_ms":  10000.0
}

THRESHOLD_PERCENTILE = 95


def trace_to_vector(trace):
    """Convert a trace dict to a normalised 5-element numpy feature vector."""
    return np.array([
        trace["avg_duration_ms"] / NORM_CONSTANTS["avg_duration_ms"],
        trace["error_rate"]      / NORM_CONSTANTS["error_rate"],
        trace["span_count"]      / NORM_CONSTANTS["span_count"],
        trace["service_count"]   / NORM_CONSTANTS["service_count"],
        trace["max_duration_ms"] / NORM_CONSTANTS["max_duration_ms"],
    ], dtype=np.float32)


class TraceDetector:
    """
    Distance-based trace anomaly detector.
    """

    def __init__(self):
        self.reference_vectors = None   # (N_normal, 5) array
        self.threshold         = None

    # ── Training ──────────────────────────────────────────────────────
    def train(self, normal_traces):
        """
        Build reference set from normal traces.
        normal_traces: list of trace dicts.
        """
        self.reference_vectors = np.array(
            [trace_to_vector(t) for t in normal_traces],
            dtype=np.float32
        )

        # Compute pairwise minimum distances within reference set
        # to establish the normal-to-normal distance distribution
        dist_matrix = euclidean_distances(
            self.reference_vectors, self.reference_vectors)
        # For each point, min distance to any OTHER point
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = dist_matrix.min(axis=1)
        self.threshold = float(np.percentile(min_dists, THRESHOLD_PERCENTILE))

        print(f"  Trace detector trained on {len(normal_traces)} normal traces")
        print(f"  Reference set shape: {self.reference_vectors.shape}")
        print(f"  Threshold (p{THRESHOLD_PERCENTILE}): {self.threshold:.4f}")

    # ── Per-trace scoring ─────────────────────────────────────────────
    def score_trace(self, trace):
        """
        Score a single trace.
        Returns: (min_distance, is_anomaly)
        """
        if self.reference_vectors is None:
            raise RuntimeError("Detector not trained. Call train() first.")

        vec  = trace_to_vector(trace).reshape(1, -1)
        dists = euclidean_distances(vec, self.reference_vectors)[0]
        min_dist  = float(dists.min())
        is_anomaly = min_dist > self.threshold
        return min_dist, is_anomaly

    # ── Window-level scoring ──────────────────────────────────────────
    def score_window(self, window_traces):
        """
        Score a window of traces.
        window_traces: list of trace dicts.

        Note: In the paper simulation, the AnoMod traces have 3-9 spans
        under normal conditions, and the injected anomalies do not cross
        the >15 span or >30% error rate heuristic thresholds. The distance-
        based detector therefore records 0% detection in the simulation,
        consistent with Table 9 in the paper.

        Returns:
            anomaly_rate:   float
            avg_distance:   float
            is_anomaly:     bool
        """
        if not window_traces:
            return {
                "anomaly_rate":  0.0,
                "avg_distance":  0.0,
                "is_anomaly":    False,
                "layer":         "traces"
            }

        anomaly_flags = []
        distances     = []

        for trace in window_traces:
            dist, is_anom = self.score_trace(trace)
            distances.append(dist)
            anomaly_flags.append(is_anom)

        anomaly_rate = float(np.mean(anomaly_flags))
        avg_distance = float(np.mean(distances))
        is_anomaly   = anomaly_rate > 0.5

        return {
            "anomaly_rate":  round(anomaly_rate, 4),
            "avg_distance":  round(avg_distance, 4),
            "is_anomaly":    is_anomaly,
            "layer":         "traces"
        }

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, path):
        np.savez(path,
                 reference_vectors=self.reference_vectors,
                 threshold=np.array([self.threshold]))
        print(f"  Trace detector saved: {path}")

    def load(self, path):
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        self.reference_vectors = data["reference_vectors"]
        self.threshold         = float(data["threshold"][0])
        print(f"  Trace detector loaded: {path}  (threshold={self.threshold:.4f})")
