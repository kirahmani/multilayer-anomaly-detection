"""
dataset_loader.py
-----------------
Loads the generated GAIA/LO2/AnoMod datasets from data/ directory.
"""

import numpy as np
import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_metrics():
    """Returns (metrics_normal, metrics_anomaly) as numpy arrays."""
    normal  = np.load(os.path.join(DATA_DIR, "metrics_normal.npy"))
    anomaly = np.load(os.path.join(DATA_DIR, "metrics_anomaly.npy"))
    return normal, anomaly


def load_logs():
    """Returns (logs_normal, logs_anomaly) as lists of dicts."""
    with open(os.path.join(DATA_DIR, "logs_normal.json"))  as f:
        normal  = json.load(f)
    with open(os.path.join(DATA_DIR, "logs_anomaly.json")) as f:
        anomaly = json.load(f)
    return normal, anomaly


def load_traces():
    """Returns (traces_normal, traces_anomaly) as lists of dicts."""
    with open(os.path.join(DATA_DIR, "traces_normal.json"))  as f:
        normal  = json.load(f)
    with open(os.path.join(DATA_DIR, "traces_anomaly.json")) as f:
        anomaly = json.load(f)
    return normal, anomaly


def get_trace_feature_vector(trace):
    """Convert a trace dict to a 5-element numpy feature vector."""
    return np.array([
        trace["avg_duration_ms"] / 5000.0,   # normalise to [0,1]
        trace["error_rate"],
        trace["span_count"] / 30.0,           # normalise to [0,1]
        trace["service_count"] / 10.0,        # normalise to [0,1]
        trace["max_duration_ms"] / 10000.0,   # normalise to [0,1]
    ], dtype=np.float32)
