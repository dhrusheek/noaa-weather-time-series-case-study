"""Forecasting models: baselines, small LSTM, small Transformer.

Primary target: next-day mean temperature (degC), one-step (t+1).
Also supports direct multi-horizon output (t+1 ... t+H) for horizon analysis.

Design guarantees for a leakage-free evaluation:
  * chronological split (no shuffling across the time axis),
  * StandardScaler fit on the TRAIN partition only,
  * input windows never contain any target value,
  * fixed seeds (Python / NumPy / PyTorch) for reproducibility.
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


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------------------- #
# Baselines (all operate on the ORIGINAL degC scale)
# --------------------------------------------------------------------------- #
def persistence_forecast(series_slice: np.ndarray):
    """Naive persistence  y_hat(t+1) = y(t).  Returns (y_true, y_pred)."""
    return series_slice[1:], series_slice[:-1]


def seasonal_naive_forecast(values: np.ndarray, test_idx, m: int):
    """Seasonal naive  y_hat(t) = y(t-m).  Evaluated on the test indices only.

    Clean (no leakage) for any horizon because y(t-m) is strictly in the past.
    """
    y_true = np.array([values[t] for t in test_idx])
    y_pred = np.array([values[t - m] for t in test_idx])
    return y_true, y_pred


def mean_forecast(train_values: np.ndarray, n_test: int):
    """Climatological mean forecast: predict the TRAIN mean everywhere."""
    return np.full(n_test, float(np.mean(train_values)))


# --------------------------------------------------------------------------- #
# LSTM  (direct multi-horizon head; horizon=1 recovers the one-step model)
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden=32, layers=1, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])          # (batch, horizon)


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
    def __init__(self, input_dim=1, d_model=32, nhead=4, layers=2, ff=64,
                 dropout=0.1, horizon=1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.head(x[:, -1, :])            # (batch, horizon)


# --------------------------------------------------------------------------- #
# Shared training loop  (returns loss history for train/val curves)
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
    if ytr_t.ndim == 1:                          # one-step -> (n,1) for a uniform head
        ytr_t = ytr_t.unsqueeze(-1); yva_t = yva_t.unsqueeze(-1)

    n = len(Xtr_t)
    best_val, best_state, wait = float("inf"), None, 0
    hist = {"train": [], "val": []}
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva_t), yva_t).item()
        hist["train"].append(ep_loss / n); hist["val"].append(val)
        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, time.time() - t0, hist


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return out
