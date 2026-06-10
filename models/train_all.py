"""
train_all.py
------------
Trains all three Layer 1 detection models and saves them to models/saved/.

Run:
    python models/train_all.py

Requires datasets to be generated first:
    python data/generate_datasets.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_metrics, load_logs, load_traces
from models.lstm_autoencoder import MetricsDetector
from models.log_classifier   import LogClassifier
from models.trace_detector   import TraceDetector

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(SAVE_DIR, exist_ok=True)


def train_metrics_detector():
    print("\n" + "="*60)
    print("Training Layer 1: LSTM-Autoencoder (Metrics Detector)")
    print("="*60)
    metrics_normal, _ = load_metrics()
    print(f"  Normal metrics shape: {metrics_normal.shape}")

    detector = MetricsDetector()
    detector.train(metrics_normal)
    detector.save(os.path.join(SAVE_DIR, "metrics_detector.pt"))
    return detector


def train_log_classifier():
    print("\n" + "="*60)
    print("Training Layer 2: Rule-Based + TF-IDF (Log Classifier)")
    print("="*60)
    logs_normal, _ = load_logs()
    print(f"  Normal log messages: {len(logs_normal)}")

    classifier = LogClassifier()
    classifier.train(logs_normal)
    classifier.save(os.path.join(SAVE_DIR, "log_classifier.pkl"))
    return classifier


def train_trace_detector():
    print("\n" + "="*60)
    print("Training Layer 3: Distance-Based (Trace Detector)")
    print("="*60)
    traces_normal, _ = load_traces()
    print(f"  Normal traces: {len(traces_normal)}")

    detector = TraceDetector()
    detector.train(traces_normal)
    detector.save(os.path.join(SAVE_DIR, "trace_detector"))
    return detector


if __name__ == "__main__":
    print("Multi-Layer Fusion Architecture — Model Training")
    print("Hardware: CPU mode (reproducible without GPU)")
    print()

    metrics_detector = train_metrics_detector()
    log_classifier   = train_log_classifier()
    trace_detector   = train_trace_detector()

    print("\n" + "="*60)
    print("All models trained and saved successfully.")
    print(f"Saved to: {SAVE_DIR}")
    print("="*60)
    print("\nNext step: python evaluate.py")
