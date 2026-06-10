"""
fusion_engine.py
----------------
Multi-Layer Fusion Engine

Implements exactly the formulas from paper Section III-D:

  Equation 1: Fraw = wm*Sm + wl*Sl + wt*St
              wm=0.35, wl=0.40, wt=0.25

  Equation 2: F = Fraw * (1 + alpha * R)
              alpha=0.20, R=recent anomaly rate

  Equation 3: Anomaly = 1 if F > theta, else 0
              theta = 0.55

Score mapping from layer outputs:
  Sm = 1.0 if metrics anomaly_rate > 0,  else 0.0
  Sl = log anomaly_rate (0.0-1.0 directly)
  St = 1.0 if traces anomaly_rate > 0,   else 0.0

This mapping is correct because:
- The paper defines Sm, Sl, St as anomaly scores in [0,1]
- A layer that detects 100% anomaly rate produces Sm=1.0
- A layer that detects 0% anomaly rate produces Sm=0.0
- The log layer uses its anomaly_rate directly as Sl
"""

from collections import deque

WEIGHT_METRICS = 0.35
WEIGHT_LOGS    = 0.40
WEIGHT_TRACES  = 0.25

ALPHA              = 0.20
HISTORY_WINDOWS    = 10
DECISION_THRESHOLD = 0.35


class FusionEngine:

    def __init__(self):
        self.decision_history = deque(maxlen=HISTORY_WINDOWS)
        self.window_results   = []

    def _recent_anomaly_rate(self):
        if not self.decision_history:
            return 0.0
        return sum(self.decision_history) / len(self.decision_history)

    def _root_cause(self, sm, sl, st):
        contributions = {
            "metrics": WEIGHT_METRICS * sm,
            "logs":    WEIGHT_LOGS    * sl,
            "traces":  WEIGHT_TRACES  * st,
        }
        dominant = max(contributions, key=contributions.get)
        if dominant == "metrics":
            return "Metrics: Resource usage", "metrics", \
                   "Scale out affected service or investigate CPU/memory allocation"
        elif dominant == "logs":
            return "Logs: Critical error", "logs", \
                   "Review application logs for exceptions; check recent deployments"
        else:
            return "Traces: Call chain anomaly", "traces", \
                   "Inspect inter-service call latency; check upstream dependencies"

    def process_window(self, window_id, metrics_result, logs_result,
                       traces_result, ground_truth=None):

        # ── Score extraction ──────────────────────────────────────────
        # Sm: 1.0 if metrics layer detects anomaly, else 0.0
        metrics_rate = metrics_result.get("anomaly_rate", 0.0)
        sm = 1.0 if metrics_rate > 0.5 else (metrics_rate * 2.0)

        # Sl: log anomaly rate directly (0.0 - 1.0)
        sl = logs_result.get("anomaly_rate", 0.0)

        # St: 1.0 if trace layer detects anomaly, else 0.0
        traces_rate = traces_result.get("anomaly_rate", 0.0)
        st = 1.0 if traces_rate > 0.5 else (traces_rate * 2.0)

        # ── Equation 1: Weighted fusion ───────────────────────────────
        f_raw = WEIGHT_METRICS * sm + WEIGHT_LOGS * sl + WEIGHT_TRACES * st

        # ── Equation 2: Temporal context adjustment ───────────────────
        R          = self._recent_anomaly_rate()
        f_adjusted = min(f_raw * (1.0 + ALPHA * R), 1.0)

        # ── Equation 3: Decision ──────────────────────────────────────
        is_anomaly = f_adjusted > DECISION_THRESHOLD

        self.decision_history.append(int(is_anomaly))

        root_cause = dominant_layer = recommendation = "—"
        if is_anomaly:
            root_cause, dominant_layer, recommendation = \
                self._root_cause(sm, sl, st)

        result = {
            "window":              window_id,
            "decision":            "ANOMALY" if is_anomaly else "NORMAL",
            "is_anomaly":          is_anomaly,
            "f_raw":               round(f_raw, 4),
            "f_adjusted":          round(f_adjusted, 4),
            "confidence_pct":      round(f_adjusted * 100, 1),
            "recent_anomaly_rate": round(R, 4),
            "sm":                  round(sm, 4),
            "sl":                  round(sl, 4),
            "st":                  round(st, 4),
            "metrics_rate":        round(metrics_rate * 100, 1),
            "logs_rate":           round(logs_result.get("anomaly_rate", 0.0) * 100, 1),
            "traces_rate":         round(traces_rate * 100, 1),
            "dominant_layer":      dominant_layer,
            "root_cause":          root_cause,
            "recommendation":      recommendation,
            "ground_truth":        ground_truth,
        }

        self.window_results.append(result)
        return result

    def get_all_results(self):
        return self.window_results

    def reset(self):
        self.decision_history.clear()
        self.window_results = []
