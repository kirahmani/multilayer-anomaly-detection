"""
log_classifier.py
-----------------
Layer 2 — Log Anomaly Detector: Rule-Based Classifier

Design (from paper Section III-B and Tables 4, 6):
  Keyword matching -> severity taxonomy -> confidence score
  CRITICAL (1.0), ERROR (0.9), WARNING (0.6), NORMAL (0.1)
  Window: aggregate 100 log messages -> window-level anomaly rate + confidence
  Output: anomaly_rate, avg_confidence, is_anomaly
"""

import numpy as np
import os
import pickle

# ── Severity taxonomy (Tables 4 and 6) ───────────────────────────────
SEVERITY_RULES = [
    {
        "severity":   "CRITICAL",
        "confidence": 1.00,
        "is_anomaly": True,
        "keywords":   [
            "crashloopbackoff", "oomkilled", "fatal error", "fatal",
            "panic", "critical"
        ]
    },
    {
        "severity":   "ERROR",
        "confidence": 0.90,
        "is_anomaly": True,
        "keywords":   [
            "connection timeout", "timeout", " 503 ", " 500 ",
            "nullpointerexception", "outofmemoryerror", "outofmemory",
            "connection refused", "deadlock", "http 503", "http 500",
            "nullpointer", "exception", "error:", "failed:", "failure"
        ]
    },
    {
        "severity":   "WARNING",
        "confidence": 0.60,
        "is_anomaly": False,   # WARNING is NOT anomalous — matches paper
        "keywords":   [
            "retry", "fallback", "slow response", "high memory",
            "warning", "warn", "queue depth", "circuit breaker"
        ]
    },
]


class LogClassifier:
    """
    Rule-based log anomaly detector.
    Pure keyword matching — no TF-IDF fallback.
    This ensures clean 0% anomaly rate for normal logs and
    exact control over anomaly rates in detection windows.
    """

    def __init__(self):
        pass

    def train(self, normal_logs):
        """No training needed for pure rule-based classifier."""
        print(f"  Log classifier ready (rule-based, {len(normal_logs)} normal messages indexed)")

    def classify_message(self, message):
        """
        Classify a single log message using keyword matching only.
        Returns: (is_anomaly, confidence, severity)
        """
        msg_lower = message.lower()

        for rule in SEVERITY_RULES:
            for kw in rule["keywords"]:
                if kw in msg_lower:
                    return rule["is_anomaly"], rule["confidence"], rule["severity"]

        # No match — normal
        return False, 0.10, "NORMAL"

    def score_window(self, window_logs):
        """
        Score a window of log messages.
        Returns anomaly_rate, avg_confidence, is_anomaly.
        """
        if not window_logs:
            return {
                "anomaly_rate":   0.0,
                "avg_confidence": 0.1,
                "is_anomaly":     False,
                "severity_counts": {},
                "layer":           "logs"
            }

        anomaly_flags   = []
        confidences     = []
        severity_counts = {"CRITICAL": 0, "ERROR": 0,
                           "WARNING":  0, "NORMAL": 0}

        for log in window_logs:
            msg = log["message"] if isinstance(log, dict) else str(log)
            is_anom, conf, sev = self.classify_message(msg)
            anomaly_flags.append(is_anom)
            confidences.append(conf)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        anomaly_rate   = float(np.mean(anomaly_flags))
        avg_confidence = float(np.mean(confidences))
        is_anomaly     = anomaly_rate > 0.0

        return {
            "anomaly_rate":   round(anomaly_rate, 4),
            "avg_confidence": round(avg_confidence, 4),
            "is_anomaly":     is_anomaly,
            "severity_counts": severity_counts,
            "layer":           "logs"
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"type": "rule_based"}, f)
        print(f"  Log classifier saved: {path}")

    def load(self, path):
        # Rule-based — nothing to load, just confirm
        print(f"  Log classifier loaded: {path}")
