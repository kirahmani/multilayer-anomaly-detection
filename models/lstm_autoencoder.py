"""
lstm_autoencoder.py
-------------------
Layer 1 — Metrics Anomaly Detector: LSTM-Autoencoder

Architecture (from paper Section III-A):
  Input:   (batch, 50 timesteps, 8 features)
  Encoder: 2-layer LSTM, 128 hidden units, dropout=0.2
  Latent:  128-dim vector from final hidden state
  Decoder: 2-layer LSTM, 128 hidden units, reverse-order reconstruction
  Output:  (batch, 50 timesteps, 8 features)
  Score:   MSE between input and reconstruction

Training:
  Optimizer: Adam, lr=0.001
  Loss:      MSE
  Batch:     32
  Epochs:    20
  Threshold: 95th percentile of training reconstruction errors
"""

import numpy as np
import torch
import torch.nn as nn
import os

# ── Reproducibility ────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters (from paper) ───────────────────────────────────────
SEQ_LEN      = 50      # timesteps per window
N_FEATURES   = 8       # metric feature count
HIDDEN_UNITS = 128
N_LAYERS     = 2
DROPOUT      = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE   = 32
N_EPOCHS     = 20
THRESHOLD_PERCENTILE = 95


# ─────────────────────────────────────────────────────────────────────
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (hidden, _) = self.lstm(x)
        # Return final hidden state of last layer: (batch, hidden_size)
        return hidden[-1]


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # z: (batch, hidden_size)
        # Repeat latent vector across seq_len timesteps
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        # Decode
        out, _ = self.lstm(z_repeated)
        # Project to feature space
        reconstruction = self.output_layer(out)
        # Reverse temporal order (decoder reconstructs in reverse order)
        reconstruction = torch.flip(reconstruction, dims=[1])
        return reconstruction


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden_size=HIDDEN_UNITS,
                 num_layers=N_LAYERS, dropout=DROPOUT, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, n_features, num_layers, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ─────────────────────────────────────────────────────────────────────
class MetricsDetector:
    """
    High-level wrapper for training and inference with the LSTM-Autoencoder.
    Matches paper parameters exactly.
    """

    def __init__(self, device=None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = LSTMAutoencoder().to(self.device)
        self.threshold = None
        self.train_errors = []

    # ── Training ──────────────────────────────────────────────────────
    def _make_windows(self, data):
        """
        Slide a window of SEQ_LEN over data rows.
        data: (N, 8) numpy array
        Returns: (N - SEQ_LEN + 1, SEQ_LEN, 8) numpy array
        """
        windows = []
        for i in range(len(data) - SEQ_LEN + 1):
            windows.append(data[i:i + SEQ_LEN])
        return np.array(windows, dtype=np.float32)

    def train(self, normal_data):
        """
        Train LSTM-AE on normal_data only.
        normal_data: (N, 8) numpy array of normal metric samples.
        """
        windows = self._make_windows(normal_data)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(windows))
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(N_EPOCHS):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                x_hat, _ = self.model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                avg = epoch_loss / len(loader)
                print(f"  Epoch [{epoch+1}/{N_EPOCHS}]  Loss: {avg:.6f}")

        # Compute threshold from training reconstruction errors
        self._compute_threshold(windows)

    def _compute_threshold(self, windows):
        """Set threshold at THRESHOLD_PERCENTILE of training MSE errors."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(windows), BATCH_SIZE):
                batch = torch.tensor(
                    windows[i:i + BATCH_SIZE]).to(self.device)
                x_hat, _ = self.model(batch)
                mse = ((batch - x_hat) ** 2).mean(dim=(1, 2))
                errors.extend(mse.cpu().numpy().tolist())
        self.train_errors = errors
        self.threshold    = float(np.percentile(errors, THRESHOLD_PERCENTILE))
        print(f"  Metrics threshold (p{THRESHOLD_PERCENTILE}): {self.threshold:.6f}")

    # ── Inference ─────────────────────────────────────────────────────
    def score_window(self, window_data):
        """
        Score a detection window of metric data.
        window_data: (100, 8) numpy array (detection window size from paper)
        Returns:
            anomaly_rate: float  — fraction of sub-windows flagged as anomalous
            avg_score:    float  — mean MSE reconstruction error
            is_anomaly:   bool   — whether window is flagged overall
        """
        if self.threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")

        windows = self._make_windows(window_data)   # (51, 50, 8)
        self.model.eval()
        anomaly_flags = []
        scores        = []

        with torch.no_grad():
            for i in range(0, len(windows), BATCH_SIZE):
                batch = torch.tensor(
                    windows[i:i + BATCH_SIZE]).to(self.device)
                x_hat, _ = self.model(batch)
                mse = ((batch - x_hat) ** 2).mean(dim=(1, 2))
                mse_np = mse.cpu().numpy()
                scores.extend(mse_np.tolist())
                anomaly_flags.extend((mse_np > self.threshold).tolist())

        anomaly_rate = float(np.mean(anomaly_flags))
        avg_score    = float(np.mean(scores))
        # Normalise avg_score to [0,1] range using threshold as reference
        normalised_score = min(avg_score / (self.threshold * 2 + 1e-8), 1.0)
        is_anomaly   = anomaly_rate > 0.5

        return {
            "anomaly_rate":      round(anomaly_rate, 4),
            "avg_mse":           round(avg_score, 6),
            "normalised_score":  round(normalised_score, 4),
            "is_anomaly":        is_anomaly,
            "layer":             "metrics"
        }

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold":   self.threshold,
            "train_errors": self.train_errors
        }, path)
        print(f"  Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.threshold    = ckpt["threshold"]
        self.train_errors = ckpt.get("train_errors", [])
        self.model.eval()
        print(f"  Model loaded: {path}  (threshold={self.threshold:.6f})")
