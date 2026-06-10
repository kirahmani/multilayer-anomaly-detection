# Multi-Layer Fusion Architecture for Anomaly Detection in Cloud Microservices

Reproducible experimental code for the paper:
**"A Multi-Layer Fusion Architecture for Intelligent Anomaly Detection and Proactive Failure Prevention in Cloud Microservices"**

## Requirements

- Python 3.10+
- PyTorch 2.0
- See requirements.txt for full dependencies

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic datasets (GAIA, LO2, AnoMod simulation)
python data/generate_datasets.py

# 3. Train all three detection layer models
python models/train_all.py

# 4. Run the full 20-window experiment
python evaluate.py

# 5. View results tables
python results/generate_tables.py
```

## Repository Structure

```
├── data/
│   ├── generate_datasets.py     # Synthetic GAIA/LO2/AnoMod data generation
│   └── dataset_loader.py        # Dataset loading utilities
├── models/
│   ├── lstm_autoencoder.py      # Layer 1: LSTM-AE metrics detector
│   ├── log_classifier.py        # Layer 2: Rule-based + TF-IDF log detector
│   ├── trace_detector.py        # Layer 3: Distance-based trace detector
│   └── train_all.py             # Train all three models
├── fusion/
│   └── fusion_engine.py         # Weighted fusion + temporal adjustment + RCA
├── evaluation/
│   └── metrics.py               # Precision, recall, F1, confidence intervals
├── results/
│   └── generate_tables.py       # Reproduce Tables 7-12 from the paper
├── evaluate.py                  # Main experiment runner (20-window simulation)
├── requirements.txt
└── README.md
```

## Hardware Used

- Intel Core i7-12650H (2.30 GHz), 16 GB RAM
- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) — CPU mode used for reproducibility
- Windows 11 64-bit, Python 3.10, PyTorch 2.0

## Expected Output

Running evaluate.py produces:
- Per-window detection table (Table 7)
- Overall performance metrics with 95% CI (Table 8)
- Layer-wise detection performance (Table 9)
- Root cause distribution (Table 10)
- Temporal detection analysis (Table 11)
- Hypothesis validation (Table 12)
