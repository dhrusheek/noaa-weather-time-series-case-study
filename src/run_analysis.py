"""End-to-end NOAA weather time-series pipeline.

Runs: preprocessing -> EDA figures -> trend/seasonality (STL) ->
stationarity (ADF/KPSS) -> ACF/PACF -> persistence baseline -> LSTM ->
Transformer -> comparison. Saves all figures to ../figures and a machine
readable results.json for the report/README.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

import models as M
import statistical_analysis as S
from preprocessing import build_processed

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 150
RESULTS = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=DPI, bbox_inches="tight")
    plt.close()


def main():
    M.set_seed()
    df = build_processed()
    temp = df["temperature_c_interp"].copy()  # continuous series for modelling/stats
    temp_raw = df["temperature_c"].copy()
    prcp = df["precipitation_mm"].copy()

    RESULTS["station"] = {
        "name": str(df["NAME"].iloc[0]),
        "id": str(df["STATION"].iloc[0]),
        "lat": float(df["LATITUDE"].iloc[0]),
        "lon": float(df["LONGITUDE"].iloc[0]),
        "elevation_m": float(df["ELEVATION"].iloc[0]),
    }
    RESULTS["dataset"] = {
        "n_obs": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "missing_temperature_c": int(temp_raw.isna().sum()),
        "missing_precipitation_mm": int(prcp.isna().sum()),
        "temp_describe": {k: float(v) for k, v in temp_raw.describe().items()},
        "prcp_describe": {k: float(v) for k, v in prcp.describe().items()},
    }

    # ------------------------------------------------------------------ #
    # 1-2. EDA figures
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(13, 4))
    plt.plot(temp.index, temp.values, lw=0.7, color="#c0392b")
    plt.title("Daily Mean Temperature — Bengaluru (VOBL), 2019–2024")
    plt.xlabel("Date"); plt.ylabel("Temperature (°C)")
    savefig("temperature_timeseries.png")

    plt.figure(figsize=(13, 4))
    plt.bar(prcp.index, prcp.values, width=1.0, color="#2471a3")
    plt.title("Daily Precipitation — Bengaluru (VOBL), 2019–2024")
    plt.xlabel("Date"); plt.ylabel("Precipitation (mm)")
    savefig("precipitation_timeseries.png")

    # rolling stats
    roll_mean = temp.rolling(30).mean()
    roll_std = temp.rolling(30).std()
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(temp.index, temp.values, lw=0.5, alpha=0.4, label="Daily", color="grey")
    ax[0].plot(roll_mean.index, roll_mean.values, lw=1.6, color="#c0392b", label="30-day mean")
    ax[0].set_ylabel("Temperature (°C)"); ax[0].legend(); ax[0].set_title("Temperature with 30-day Rolling Mean")
    ax[1].plot(roll_std.index, roll_std.values, lw=1.3, color="#8e44ad")
    ax[1].set_ylabel("Std (°C)"); ax[1].set_xlabel("Date"); ax[1].set_title("30-day Rolling Standard Deviation")
    savefig("rolling_statistics.png")

    # monthly mean temperature
    monthly_t = temp.resample("MS").mean()
    plt.figure(figsize=(13, 4))
    plt.plot(monthly_t.index, monthly_t.values, marker="o", ms=3, color="#c0392b")
    plt.title("Monthly Mean Temperature — Bengaluru")
    plt.xlabel("Month"); plt.ylabel("Mean Temperature (°C)")
    savefig("monthly_temperature.png")

    # monthly climatology of precipitation (seasonal signature)
    prcp_clim = prcp.groupby(prcp.index.month).mean()
    plt.figure(figsize=(9, 4))
    plt.bar(prcp_clim.index, prcp_clim.values, color="#2471a3")
    plt.title("Mean Daily Precipitation by Calendar Month (2019–2024)")
    plt.xlabel("Month"); plt.ylabel("Mean daily precipitation (mm)")
    plt.xticks(range(1, 13))
    savefig("monthly_precipitation.png")

    # temperature monthly boxplot (seasonality spread)
    tdf = temp.to_frame("t"); tdf["month"] = tdf.index.month
    plt.figure(figsize=(11, 4))
    tdf.boxplot(column="t", by="month", grid=False)
    plt.title("Temperature Distribution by Month"); plt.suptitle("")
    plt.xlabel("Month"); plt.ylabel("Temperature (°C)")
    savefig("monthly_temperature_boxplot.png")

    # ------------------------------------------------------------------ #
    # 3. STL decomposition (annual period)
    # ------------------------------------------------------------------ #
    stl = STL(temp, period=365, robust=True).fit()
    fig = stl.plot(); fig.set_size_inches(13, 9)
    fig.suptitle("STL Decomposition of Daily Temperature (period=365)", y=1.01)
    savefig("seasonal_decomposition.png")
    seasonal_strength = max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
    trend_strength = max(0.0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())
    RESULTS["decomposition"] = {
        "period": 365,
        "seasonal_strength": float(seasonal_strength),
        "trend_strength": float(trend_strength),
        "trend_first": float(stl.trend.dropna().iloc[0]),
        "trend_last": float(stl.trend.dropna().iloc[-1]),
    }

    # ------------------------------------------------------------------ #
    # 4. Stationarity: raw + first difference
    # ------------------------------------------------------------------ #
    temp_diff = temp.diff().dropna()
    adf_raw, kpss_raw = S.adf_test(temp), S.kpss_test(temp)
    adf_diff, kpss_diff = S.adf_test(temp_diff), S.kpss_test(temp_diff)
    RESULTS["stationarity"] = {
        "raw": {"adf": adf_raw, "kpss": kpss_raw,
                "conclusion": S.stationarity_conclusion(adf_raw, kpss_raw)},
        "diff": {"adf": adf_diff, "kpss": kpss_diff,
                 "conclusion": S.stationarity_conclusion(adf_diff, kpss_diff)},
    }

    # ------------------------------------------------------------------ #
    # 5. ACF / PACF (raw + differenced)
    # ------------------------------------------------------------------ #
    for series, tag, title in [
        (temp, "raw", "Raw Daily Temperature"),
        (temp_diff, "stationary", "First-differenced Temperature"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_acf(series.dropna(), lags=60, ax=ax)
        ax.set_title(f"ACF — {title}"); ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
        savefig(f"acf_{tag}.png")
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_pacf(series.dropna(), lags=60, ax=ax, method="ywm")
        ax.set_title(f"PACF — {title}"); ax.set_xlabel("Lag (days)"); ax.set_ylabel("PACF")
        savefig(f"pacf_{tag}.png")
    # long-lag ACF to expose annual cycle
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(temp.dropna(), lags=400, ax=ax)
    ax.set_title("ACF — Raw Temperature (400 lags, annual cycle)")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    savefig("acf_longlag.png")
    RESULTS["acf"] = {
        "lag1_raw": float(temp.autocorr(1)),
        "lag1_diff": float(temp_diff.autocorr(1)),
        "acf_lag365_raw": float(temp.autocorr(365)),
    }

    # ------------------------------------------------------------------ #
    # 6. Forecasting: chronological split, scale on train only
    # ------------------------------------------------------------------ #
    SEQ = 30
    vals = temp.values.astype(float)
    n = len(vals)
    i_tr, i_va = int(n * 0.7), int(n * 0.85)

    scaler = StandardScaler().fit(vals[:i_tr].reshape(-1, 1))
    scaled = scaler.transform(vals.reshape(-1, 1)).ravel()

    def split_seqs(a, lo, hi):
        # build sequences whose TARGET index falls in [lo, hi)
        X, y = [], []
        for t in range(SEQ, n):
            if lo <= t < hi:
                X.append(a[t - SEQ : t]); y.append(a[t])
        return np.array(X)[..., None], np.array(y)

    Xtr, ytr = split_seqs(scaled, SEQ, i_tr)
    Xva, yva = split_seqs(scaled, i_tr, i_va)
    Xte, yte = split_seqs(scaled, i_va, n)

    def invert(a):
        return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()

    test_dates = temp.index[i_va:n]

    # persistence baseline (on original scale, test region)
    y_true_p, y_pred_p = M.persistence_forecast(vals[i_va - 1 : n])
    base = {"mae": M.mae(y_true_p, y_pred_p), "rmse": M.rmse(y_true_p, y_pred_p), "train_time": 0.0}

    # LSTM
    lstm = M.LSTMForecaster(hidden=32, layers=1)
    lstm, lstm_val, lstm_tt = M.train_model(lstm, Xtr, ytr, Xva, yva, epochs=15, patience=3)
    import torch
    with torch.no_grad():
        pred_lstm = invert(lstm(torch.tensor(Xte, dtype=torch.float32)).numpy())
    yte_orig = invert(yte)
    lstm_res = {"mae": M.mae(yte_orig, pred_lstm), "rmse": M.rmse(yte_orig, pred_lstm),
                "val_loss": float(lstm_val), "train_time": float(lstm_tt)}

    # Transformer
    trf = M.TransformerForecaster(d_model=32, nhead=4, layers=2, ff=64, dropout=0.1)
    trf, trf_val, trf_tt = M.train_model(trf, Xtr, ytr, Xva, yva, epochs=15, patience=3)
    with torch.no_grad():
        pred_trf = invert(trf(torch.tensor(Xte, dtype=torch.float32)).numpy())
    trf_res = {"mae": M.mae(yte_orig, pred_trf), "rmse": M.rmse(yte_orig, pred_trf),
               "val_loss": float(trf_val), "train_time": float(trf_tt)}

    RESULTS["forecast"] = {
        "seq_len": SEQ, "n_train_seq": int(len(Xtr)), "n_val_seq": int(len(Xva)),
        "n_test_seq": int(len(Xte)),
        "persistence": base, "lstm": lstm_res, "transformer": trf_res,
    }

    # forecast plots
    for pred, name, fname, color in [
        (pred_lstm, "LSTM", "lstm_forecast.png", "#c0392b"),
        (pred_trf, "Transformer", "transformer_forecast.png", "#8e44ad"),
    ]:
        plt.figure(figsize=(13, 4))
        plt.plot(test_dates, yte_orig, label="Actual", lw=1.2, color="black")
        plt.plot(test_dates, pred, label=f"{name} predicted", lw=1.2, color=color, alpha=0.8)
        plt.title(f"{name} — Next-day Temperature Forecast (test set)")
        plt.xlabel("Date"); plt.ylabel("Temperature (°C)"); plt.legend()
        savefig(fname)

    # comparison bar chart
    labels = ["Persistence", "LSTM", "Transformer"]
    maes = [base["mae"], lstm_res["mae"], trf_res["mae"]]
    rmses = [base["rmse"], lstm_res["rmse"], trf_res["rmse"]]
    x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, maes, w, label="MAE", color="#2471a3")
    plt.bar(x + w/2, rmses, w, label="RMSE", color="#c0392b")
    plt.xticks(x, labels); plt.ylabel("Error (°C)")
    plt.title("Model Comparison — Test MAE / RMSE"); plt.legend()
    for i, (m, r) in enumerate(zip(maes, rmses)):
        plt.text(i - w/2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + w/2, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    savefig("model_comparison.png")

    with open(os.path.join(HERE, "..", "results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()
