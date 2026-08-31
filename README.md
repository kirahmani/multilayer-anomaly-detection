# Multi-Layer Fusion Architecture for Anomaly Detection in Cloud Microservices

Reproducible experimental code for the paper:

**"Intelligent Anomaly Detection and Proactive Failure Prevention in Cloud Microservices Using a Multi-Layer Fusion Architecture"**  
Mohammad Khalid Imam Rahmani, Abdulmajeed Aljuhani  
*IEEE Access, Manuscript ID: Access-2026-32430*

---

## Requirements

- Python 3.10+
- PyTorch 2.0
- See `requirements.txt` for full dependencies

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook multilayer_anomaly_detection.ipynb
```

Run all cells in order to reproduce all tables and figures from the paper.

---

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter notebook multilayer_anomaly_detection.ipynb
   ```

3. Run all cells to reproduce Tables VII–XIII and Figures 4–6.

---

## Repository Structure

```
multilayer_anomaly_detection.ipynb   # Complete experiment notebook
requirements.txt                      # Python dependencies
README.md
```

---

## Notebook Contents

| Section | Description |
|---------|-------------|
| 1. Imports & Configuration | All hyperparameters: θ=0.35, α=0.20, weights (0.35, 0.40, 0.25) |
| 2. Synthetic Dataset Generation | Metrics (GAIA-style), Logs (LO2-style), Traces (AnoMod-style) |
| 3. Layer 1 — LSTM-Autoencoder | Training on normal data, Sm score computation |
| 4. Layer 2 — Rule-Based Log Classifier | Keyword taxonomy, Sl score computation |
| 5. Layer 3 — Distance-Based Trace Detector | Reference set construction, St score computation |
| 6. Multi-Layer Fusion Engine | Weighted fusion + temporal context adjustment |
| 7. Results | Tables VII–X, Figure 4 |
| 8. Ablation Study | Table XIII, Figures 5–6 |
| 9. Export | All CSVs and figures saved |

---

## Expected Output

Running all notebook cells produces:

- **Table VII** — Per-window detection summary
- **Table VIII** — Overall performance metrics with 95% Wilson confidence intervals
- **Table IX** — Layer-wise detection performance across anomalous windows
- **Table X** — Root cause distribution
- **Table XIII** — Ablation study: fusion weight sensitivity
- **Figure 4** — Per-window anomaly detection rates across 20 monitoring windows
- **Figure 5** — Threshold sensitivity analysis (θ = 0.10 to 0.80)
- **Figure 6** — F1-score by fusion weight configuration

---

## Hardware Used

- Intel Core i7-12650H (2.30 GHz), 16 GB RAM
- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) — CPU mode used for reproducibility
- Windows 11 64-bit, Python 3.10, PyTorch 2.0

All LSTM-Autoencoder training and inference were performed on CPU to ensure reproducibility on standard hardware without a GPU.

---

## Dataset

The experiment uses synthetic telemetry calibrated to match three publicly available AIOps benchmark datasets:

- **GAIA** — for system metrics (8-dimensional feature vectors)
- **LO2** — for application logs (severity distribution and keyword patterns)
- **AnoMod** — for distributed traces (span depth and latency specifications)

The original datasets were not directly evaluated. Synthetic generation ensures controlled anomaly injection with precise ground-truth labels.

---

## Citation

If you use this code, please cite:

```
M. K. I. Rahmani and A. Aljuhani, "Intelligent Anomaly Detection and Proactive 
Failure Prevention in Cloud Microservices Using a Multi-Layer Fusion Architecture," 
IEEE Access, Manuscript ID: Access-2026-32430.
```
