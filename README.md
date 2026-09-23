# Multi-Layer Fusion Architecture for Anomaly Detection in Cloud Microservices

Experimental code for the paper:

**"Intelligent Anomaly Detection and Proactive Failure Prevention in Cloud Microservices Using a Multi-Layer Fusion Architecture"**  
Mohammad Khalid Imam Rahmani, Abdulmajeed Aljuhani  
*IEEE Access, Manuscript ID: Access-2026-32430*

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook multilayer_anomaly_detection.ipynb
```

Run all cells in order to produce all tables and figures from the paper.

---

## Requirements

- Python 3.10 or higher
- See `requirements.txt` for full dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
multilayer_anomaly_detection.ipynb    # Complete experiment notebook
requirements.txt                       # Python dependencies
README.md                              # This file
```

---

## Notebook Structure

The notebook is organised in two parts.

### Part A — Dataset and Layer Computation (Section 0)

Generates synthetic telemetry calibrated to GAIA, LO2, and AnoMod
statistics (Table V) and runs the three detection layers.

| Subsection | Description |
|---|---|
| 0-A | Metric time-series — 2,000 samples × 8 features (Table II, GAIA-calibrated) |
| 0-B | LSTM-Autoencoder training on normal operating data (Section III-A) |
| 0-C | Sm score computation per detection window |
| 0-D | Log message streams — 1,000 messages across 20 windows (Table III, LO2-calibrated) |
| 0-E | Sl score computation per detection window |
| 0-F | Trace data — 100 traces (Table IV, AnoMod-calibrated) |
| 0-G | St score computation per detection window |
| 0-H | Per-window layer score summary |

### Part B — Results (Sections 1–7)

| Section | Description |
|---|---|
| 1. Table I | Architecture notation and hyperparameters |
| 2. Table II | Metric feature taxonomy (Layer 1 specification) |
| 3. Table III | Log severity taxonomy (Layer 2 specification) |
| 4. Table IV | Trace feature thresholds (Layer 3 specification) |
| 5. Table VI | Simulation parameters |
| 6. Per-window scores | Computed Sm, Sl, St values (Section IV-C) |
| 7. Results | Tables VII–X, XIII, Figures 4–6, Hypothesis validation |

---

## Expected Output

Running all cells produces:

| Output | Description |
|---|---|
| Table VII | Per-window detection summary with F_adj confidence values |
| Table VIII | Overall performance metrics with 95% Wilson confidence intervals |
| Table IX | Layer-wise detection performance across anomalous windows |
| Table X | Root cause distribution (4 categories) |
| Table XIII | Ablation study — 7 fusion weight configurations |
| Figure 4 | Per-window anomaly detection rates (W1–W20) |
| Figure 5 | F1-score by fusion weight configuration |
| Figure 6 | Threshold sensitivity (θ = 0.10 to 0.80) |
| CSVs | table7_per_window.csv, table8_overall_metrics.csv, table9_layerwise.csv, table10_root_cause.csv, table13_ablation.csv, threshold_sensitivity.csv |
| PNGs | figure4_per_window_rates.png, figure5_ablation_f1.png, figure6_threshold_sensitivity.png |

## Architecture

The paper implements a three-layer fusion architecture:

- **Layer 1 — Metrics:** LSTM-Autoencoder trained on normal metric sequences.
  Sm = fraction of detection-window sub-sequences with reconstruction error
  above the training distribution threshold.
- **Layer 2 — Logs:** Rule-based keyword severity classifier.
  Sl = (CRITICAL + ERROR messages) / total messages per window.
- **Layer 3 — Traces:** Distance-based anomaly detector.
  St = normalised Euclidean distance to nearest normal reference trace.
- **Fusion Engine:** F = min(F_raw × (1 + α × R), 1.0); anomaly if F > θ.
  θ = 0.35, α = 0.20, weights (w_m = 0.35, w_l = 0.40, w_t = 0.25).

---

## Citation

```bibtex
@article{rahmani2026multilayer,
  title   = {Intelligent Anomaly Detection and Proactive Failure Prevention
             in Cloud Microservices Using a Multi-Layer Fusion Architecture},
  author  = {Rahmani, Mohammad Khalid Imam and Aljuhani, Abdulmajeed},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript ID: Access-2026-32430}
}
```

---

## License

This code is released for academic purposes.
Please cite the paper if you use this code in your research.
