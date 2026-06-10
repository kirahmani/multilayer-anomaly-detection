"""
generate_datasets.py
--------------------
Generates synthetic GAIA-style metrics, LO2-style logs, and AnoMod-style traces
that exactly match the dataset parameters described in the paper (Tables 2-5).

Run:
    python data/generate_datasets.py

Outputs (saved to data/):
    metrics_normal.npy      -- (1800, 8)  normal metric samples
    metrics_anomaly.npy     -- (200, 8)   anomaly metric samples
    logs_normal.json        -- 850 normal log messages
    logs_anomaly.json       -- 150 anomaly log messages
    traces_normal.json      -- 80 normal trace feature vectors
    traces_anomaly.json     -- 20 anomaly trace feature vectors
"""

import numpy as np
import json
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1.  GAIA-style METRICS dataset
#     8 features: CPU, Memory, Network I/O, Disk I/O,
#                 Request Rate, Error Rate, P50 Latency, P99 Latency
#     Normal ranges and anomaly ranges from Table 3
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "CPU_Usage", "Memory_Usage", "Network_IO",
    "Disk_IO", "Request_Rate", "Error_Rate",
    "Latency_P50", "Latency_P99"
]

# (normal_low, normal_high, anomaly_low, anomaly_high)
FEATURE_RANGES = [
    (0.20, 0.40, 0.65, 0.95),   # CPU
    (0.25, 0.35, 0.70, 0.90),   # Memory
    (0.15, 0.25, 0.60, 0.85),   # Network I/O
    (0.10, 0.20, 0.55, 0.80),   # Disk I/O
    (0.20, 0.30, 0.70, 0.90),   # Request Rate
    (0.00, 0.05, 0.40, 0.70),   # Error Rate
    (0.10, 0.25, 0.60, 0.85),   # Latency P50
    (0.15, 0.35, 0.75, 0.95),   # Latency P99
]


def generate_metric_samples(n_samples, anomaly=False):
    """Generate n_samples rows of 8-feature metric data."""
    data = np.zeros((n_samples, 8))
    for j, (n_lo, n_hi, a_lo, a_hi) in enumerate(FEATURE_RANGES):
        if anomaly:
            data[:, j] = np.random.uniform(a_lo, a_hi, n_samples)
            # Add occasional spike noise
            spike_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
            data[spike_idx, j] = np.clip(data[spike_idx, j] * 1.2, 0, 1)
        else:
            data[:, j] = np.random.uniform(n_lo, n_hi, n_samples)
            # Add small Gaussian noise for realism
            noise = np.random.normal(0, 0.01, n_samples)
            data[:, j] = np.clip(data[:, j] + noise, 0, 1)
    return data.astype(np.float32)


metrics_normal  = generate_metric_samples(1800, anomaly=False)
metrics_anomaly = generate_metric_samples(200,  anomaly=True)

np.save(os.path.join(SAVE_DIR, "metrics_normal.npy"),  metrics_normal)
np.save(os.path.join(SAVE_DIR, "metrics_anomaly.npy"), metrics_anomaly)
print(f"[Metrics]  Normal: {metrics_normal.shape}, Anomaly: {metrics_anomaly.shape}")


# ─────────────────────────────────────────────
# 2.  LO2-style LOG dataset
#     Categories: CRITICAL (8%), ERROR (25%), WARNING (15%), NORMAL (52%)
#     From Tables 4 and 6
# ─────────────────────────────────────────────

CRITICAL_PATTERNS = [
    "CrashLoopBackOff detected in pod payment-service",
    "OOMKilled: container memory limit exceeded in cart-service",
    "Fatal error: database connection pool exhausted",
    "Panic: nil pointer dereference in order-service",
    "CRITICAL: service mesh circuit breaker tripped",
]
ERROR_PATTERNS = [
    "Connection timeout after 30s: upstream checkout-service",
    "HTTP 503 Service Unavailable from inventory-service",
    "HTTP 500 Internal Server Error in payment-service",
    "NullPointerException at OrderController.java:142",
    "OutOfMemoryError: Java heap space in recommendation-service",
    "Connection refused: redis cache unavailable",
    "Database query timeout after 10000ms",
    "Failed to acquire lock on resource: deadlock detected",
]
WARNING_PATTERNS = [
    "Retry attempt 3/5 for request to shipping-service",
    "Fallback triggered: using cached response for product catalogue",
    "Slow response: 2847ms for /api/checkout endpoint",
    "High memory usage: 78% of container limit in cart-service",
    "Request queue depth exceeding threshold: 847 pending",
    "Circuit breaker half-open: probing upstream service",
]
NORMAL_PATTERNS = [
    "INFO: Request processed successfully in 124ms",
    "DEBUG: Cache hit for product id 8472",
    "INFO: Health check passed for all services",
    "DEBUG: Scheduled job completed: cleanup_expired_sessions",
    "INFO: User session created for session_id=a8f3c2",
    "DEBUG: Database connection pool: 12/50 connections active",
    "INFO: Metrics exported to Prometheus endpoint",
    "INFO: Trace span completed: checkout-service -> payment-service 89ms",
]


def make_log_entry(pattern, severity, confidence, is_anomaly):
    return {
        "message": pattern,
        "severity": severity,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "timestamp": f"2024-01-15T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z"
    }


def generate_logs(n_total, anomaly_fraction=0.15):
    logs = []
    n_anomaly = int(n_total * anomaly_fraction)
    n_normal  = n_total - n_anomaly

    # Normal logs (NORMAL + WARNING mix)
    for _ in range(n_normal):
        r = random.random()
        if r < 0.52 / (0.52 + 0.15):  # NORMAL proportion within non-anomaly
            logs.append(make_log_entry(
                random.choice(NORMAL_PATTERNS), "NORMAL", 0.1, False))
        else:
            logs.append(make_log_entry(
                random.choice(WARNING_PATTERNS), "WARNING", 0.6, False))

    # Anomaly logs (CRITICAL + ERROR mix)
    for _ in range(n_anomaly):
        r = random.random()
        if r < 0.08 / (0.08 + 0.25):  # CRITICAL proportion within anomaly
            logs.append(make_log_entry(
                random.choice(CRITICAL_PATTERNS), "CRITICAL", 1.0, True))
        else:
            logs.append(make_log_entry(
                random.choice(ERROR_PATTERNS), "ERROR", 0.9, True))

    random.shuffle(logs)
    return logs


logs_normal  = generate_logs(850,  anomaly_fraction=0.0)
logs_anomaly = generate_logs(150,  anomaly_fraction=1.0)

with open(os.path.join(SAVE_DIR, "logs_normal.json"), "w")  as f:
    json.dump(logs_normal,  f, indent=2)
with open(os.path.join(SAVE_DIR, "logs_anomaly.json"), "w") as f:
    json.dump(logs_anomaly, f, indent=2)
print(f"[Logs]     Normal: {len(logs_normal)}, Anomaly: {len(logs_anomaly)}")


# ─────────────────────────────────────────────
# 3.  AnoMod-style TRACE dataset
#     5 features per trace: avg_duration, error_rate, span_count,
#                           service_count, max_duration
#     Normal / anomaly ranges from Table 5
# ─────────────────────────────────────────────

def generate_trace(anomaly=False):
    if anomaly:
        # One of four anomaly types randomly selected
        anomaly_type = random.randint(0, 3)
        if anomaly_type == 0:   # latency spike
            avg_dur     = random.uniform(2600, 5000)
            error_rate  = random.uniform(0.05, 0.15)
            span_count  = random.randint(4, 10)
            svc_count   = random.randint(2, 4)
            max_dur     = avg_dur * random.uniform(1.5, 2.5)
        elif anomaly_type == 1: # high error rate
            avg_dur     = random.uniform(200, 800)
            error_rate  = random.uniform(0.35, 0.80)
            span_count  = random.randint(3, 8)
            svc_count   = random.randint(1, 3)
            max_dur     = avg_dur * random.uniform(1.2, 1.8)
        elif anomaly_type == 2: # span explosion
            avg_dur     = random.uniform(100, 400)
            error_rate  = random.uniform(0.00, 0.10)
            span_count  = random.randint(16, 30)
            svc_count   = random.randint(6, 10)
            max_dur     = avg_dur * random.uniform(2.0, 4.0)
        else:                   # combined degradation
            avg_dur     = random.uniform(1500, 4000)
            error_rate  = random.uniform(0.20, 0.50)
            span_count  = random.randint(10, 20)
            svc_count   = random.randint(4, 8)
            max_dur     = avg_dur * random.uniform(2.0, 3.0)
    else:
        avg_dur    = random.uniform(50,  500)
        error_rate = random.uniform(0.0, 0.10)
        span_count = random.randint(3,   8)
        svc_count  = random.randint(1,   3)
        max_dur    = avg_dur * random.uniform(1.5, 3.0)
        max_dur    = min(max_dur, 800)

    return {
        "avg_duration_ms":  round(avg_dur, 2),
        "error_rate":       round(error_rate, 4),
        "span_count":       span_count,
        "service_count":    svc_count,
        "max_duration_ms":  round(max_dur, 2),
        "is_anomaly":       anomaly
    }


traces_normal  = [generate_trace(anomaly=False) for _ in range(80)]
traces_anomaly = [generate_trace(anomaly=True)  for _ in range(20)]

with open(os.path.join(SAVE_DIR, "traces_normal.json"), "w")  as f:
    json.dump(traces_normal,  f, indent=2)
with open(os.path.join(SAVE_DIR, "traces_anomaly.json"), "w") as f:
    json.dump(traces_anomaly, f, indent=2)
print(f"[Traces]   Normal: {len(traces_normal)}, Anomaly: {len(traces_anomaly)}")

print("\nAll datasets generated successfully.")
print(f"Saved to: {SAVE_DIR}")
