"""Forecasting models: persistence baseline, small LSTM, small Transformer.

Target: next-day mean temperature (degrees C), one-step (t+1) forecast.
Chronological split, scaler fit on TRAIN ONLY, fixed seeds for reproducibility.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn

SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sequences(values: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def persistence_forecast(y_true_series: np.ndarray):
    """y_hat(t+1) = y(t). Returns (y_true, y_pred) aligned."""
    y_pred = y_true_series[:-1]
    y_true = y_true_series[1:]
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, layers=2, ff=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


# --------------------------------------------------------------------------- #
# Training loop (shared)
# --------------------------------------------------------------------------- #
def train_model(model, Xtr, ytr, Xva, yva, epochs=15, patience=3, lr=1e-3, batch=64):
    set_seed()
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)

    n = len(Xtr_t)
    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva_t), yva_t).item()
        if val < best_val - 1e-6:
            best_val, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0
    return model, best_val, train_time
