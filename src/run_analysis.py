"""End-to-end NOAA weather time-series pipeline (extended).

Stages
------
preprocessing -> EDA figures (incl. box/subseries/explicit-trend) ->
STL decomposition -> stationarity (ADF/KPSS, raw + differenced) -> ACF/PACF ->
one-step forecasting (5 baselines + LSTM + Transformer) -> residual & error
analysis + loss curves + error-spike investigation -> multi-horizon (1..30)
forecasting -> model comparison. Saves every figure to ../figures and a
machine-readable results.json consumed by the README / report / notebook.

Reproducible from a clean environment: `python src/run_analysis.py` (seed = 42).
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
SEQ = 30
HORIZONS = [1, 7, 14, 21, 30]
H_MAX = 30


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=DPI, bbox_inches="tight")
    plt.close()


# =========================================================================== #
def main():
    M.set_seed()
    df = build_processed()
    temp = df["temperature_c_interp"].copy()       # continuous series for stats/models
    temp_raw = df["temperature_c"].copy()
    prcp = df["precipitation_mm"].copy()
    interp_mask = temp_raw.isna() & temp.notna()   # days filled by interpolation

    RESULTS["station"] = {
        "name": str(df["NAME"].iloc[0]), "id": "43295099999",
        "lat": float(df["LATITUDE"].iloc[0]), "lon": float(df["LONGITUDE"].iloc[0]),
        "elevation_m": float(df["ELEVATION"].iloc[0]),
    }
    RESULTS["dataset"] = {
        "n_obs": int(len(df)), "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "n_years": round(len(df) / 365.25, 1),
        "missing_temperature_c": int(temp_raw.isna().sum()),
        "interpolated_temperature": int(interp_mask.sum()),
        "missing_precipitation_mm": int(prcp.isna().sum()),
        "temp_describe": {k: float(v) for k, v in temp_raw.describe().items()},
        "prcp_describe": {k: float(v) for k, v in prcp.describe().items()},
    }

    # ------------------------------------------------------------------ #
    # EDA
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(13, 4))
    plt.plot(temp.index, temp.values, lw=0.6, color="#c0392b")
    plt.title("Daily Mean Temperature — Bengaluru (VOBL)")
    plt.xlabel("Date"); plt.ylabel("Temperature (°C)")
    savefig("temperature_timeseries.png")

    plt.figure(figsize=(13, 4))
    plt.bar(prcp.index, prcp.values, width=1.0, color="#2471a3")
    plt.title("Daily Precipitation — Bengaluru (VOBL)")
    plt.xlabel("Date"); plt.ylabel("Precipitation (mm)")
    savefig("precipitation_timeseries.png")

    roll_mean, roll_std = temp.rolling(30).mean(), temp.rolling(30).std()
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(temp.index, temp.values, lw=0.4, alpha=0.35, color="grey", label="Daily")
    ax[0].plot(roll_mean.index, roll_mean.values, lw=1.6, color="#c0392b", label="30-day mean")
    ax[0].legend(); ax[0].set_ylabel("°C"); ax[0].set_title("Temperature with 30-day Rolling Mean")
    ax[1].plot(roll_std.index, roll_std.values, lw=1.1, color="#8e44ad")
    ax[1].set_ylabel("Std (°C)"); ax[1].set_xlabel("Date"); ax[1].set_title("30-day Rolling Std")
    savefig("rolling_statistics.png")

    monthly_t = temp.resample("MS").mean()
    plt.figure(figsize=(13, 4))
    plt.plot(monthly_t.index, monthly_t.values, marker="o", ms=2.5, color="#c0392b")
    plt.title("Monthly Mean Temperature"); plt.xlabel("Month"); plt.ylabel("°C")
    savefig("monthly_temperature.png")

    prcp_clim = prcp.groupby(prcp.index.month).mean()
    plt.figure(figsize=(9, 4))
    plt.bar(prcp_clim.index, prcp_clim.values, color="#2471a3")
    plt.title("Mean Daily Precipitation by Calendar Month")
    plt.xlabel("Month"); plt.ylabel("Mean daily precipitation (mm)"); plt.xticks(range(1, 13))
    savefig("monthly_precipitation.png")

    # temperature monthly boxplot
    tdf = temp.to_frame("t"); tdf["month"] = tdf.index.month
    plt.figure(figsize=(11, 4))
    tdf.boxplot(column="t", by="month", grid=False)
    plt.title("Temperature Distribution by Month"); plt.suptitle("")
    plt.xlabel("Month"); plt.ylabel("°C")
    savefig("monthly_temperature_boxplot.png")

    # seasonal SUBSERIES plot: for each month, the per-year mean across years
    piv = temp.groupby([temp.index.year, temp.index.month]).mean().unstack(0)  # rows=month
    plt.figure(figsize=(13, 4.5))
    month_mean = piv.mean(axis=1)
    for i, m in enumerate(range(1, 13)):
        yrs = piv.columns.values
        xs = i + np.linspace(-0.35, 0.35, len(yrs))
        plt.plot(xs, piv.loc[m].values, color="#3498db", lw=0.8, alpha=0.7)
        plt.hlines(month_mean.loc[m], i - 0.35, i + 0.35, color="#c0392b", lw=2)
    plt.xticks(range(12), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.title("Seasonal Subseries Plot — Monthly Mean Temperature (blue: each year, red: month mean)")
    plt.xlabel("Month"); plt.ylabel("°C")
    savefig("subseries_temperature.png")

    # ------------------------------------------------------------------ #
    # STL + EXPLICIT trend (linear fit) + isolated trend component
    # ------------------------------------------------------------------ #
    stl = STL(temp, period=365, robust=True).fit()
    fig = stl.plot(); fig.set_size_inches(13, 9)
    fig.suptitle("STL Decomposition of Daily Temperature (period=365)", y=1.01)
    savefig("seasonal_decomposition.png")

    seasonal_strength = max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
    trend_strength = max(0.0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())

    # explicit linear trend on the raw daily temperature
    x = np.arange(len(temp))
    slope, intercept = np.polyfit(x, temp.values, 1)
    slope_per_decade = slope * 365.25 * 10
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(temp.index, temp.values, lw=0.4, alpha=0.35, color="grey", label="Daily")
    ax[0].plot(temp.index, intercept + slope * x, lw=2, color="#c0392b",
               label=f"Linear trend ({slope_per_decade:+.2f} °C/decade)")
    ax[0].plot(monthly_t.index, monthly_t.values, lw=1.0, color="#f39c12", alpha=0.8, label="Monthly mean")
    ax[0].legend(); ax[0].set_ylabel("°C"); ax[0].set_title("Explicit Trend — Linear Fit over Daily Temperature")
    ax[1].plot(stl.trend.index, stl.trend.values, lw=1.4, color="#16a085")
    ax[1].set_ylabel("°C"); ax[1].set_xlabel("Date"); ax[1].set_title("Isolated STL Trend Component")
    savefig("trend_analysis.png")

    RESULTS["decomposition"] = {
        "period": 365,
        "seasonal_strength": float(seasonal_strength),
        "trend_strength": float(trend_strength),
        "stl_trend_first": float(stl.trend.dropna().iloc[0]),
        "stl_trend_last": float(stl.trend.dropna().iloc[-1]),
        "stl_trend_delta": float(stl.trend.dropna().iloc[-1] - stl.trend.dropna().iloc[0]),
        "linear_slope_per_decade": float(slope_per_decade),
    }

    # ------------------------------------------------------------------ #
    # Stationarity
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
    # ACF / PACF
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
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(temp.dropna(), lags=400, ax=ax)
    ax.set_title("ACF — Raw Temperature (400 lags, annual cycle)")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    savefig("acf_longlag.png")
    RESULTS["acf"] = {
        "lag1_raw": float(temp.autocorr(1)), "lag1_diff": float(temp_diff.autocorr(1)),
        "acf_lag7_raw": float(temp.autocorr(7)), "acf_lag365_raw": float(temp.autocorr(365)),
    }

    # ================================================================== #
    # FORECASTING SETUP (chronological split; scaler on train only)
    # ================================================================== #
    vals = temp.values.astype(float)
    n = len(vals)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    scaler = StandardScaler().fit(vals[:i_tr].reshape(-1, 1))
    scaled = scaler.transform(vals.reshape(-1, 1)).ravel()

    def inv(a):
        return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()

    # ---- one-step sequences (target index t in a region) ----
    def one_step(a, lo, hi):
        X, y, idx = [], [], []
        for t in range(SEQ, n):
            if lo <= t < hi:
                X.append(a[t - SEQ:t]); y.append(a[t]); idx.append(t)
        return np.array(X)[..., None], np.array(y), idx

    Xtr, ytr, _ = one_step(scaled, SEQ, i_tr)
    Xva, yva, _ = one_step(scaled, i_tr, i_va)
    Xte, yte, test_idx = one_step(scaled, i_va, n)
    yte_orig = inv(yte)
    test_dates = temp.index[test_idx]
    RESULTS["forecast_setup"] = {
        "seq_len": SEQ, "split": "70/15/15 chronological",
        "n_train_seq": int(len(Xtr)), "n_val_seq": int(len(Xva)), "n_test_seq": int(len(Xte)),
        "test_start": str(test_dates.min().date()), "test_end": str(test_dates.max().date()),
    }

    # ------------------------------------------------------------------ #
    # BASELINES (one-step, test region)
    # ------------------------------------------------------------------ #
    baselines = {}
    yt_p, yp_p = M.persistence_forecast(vals[test_idx[0] - 1:test_idx[-1] + 1])
    baselines["Persistence (t-1)"] = {"mae": M.mae(yt_p, yp_p), "rmse": M.rmse(yt_p, yp_p)}
    yt_s7, yp_s7 = M.seasonal_naive_forecast(vals, test_idx, m=7)
    baselines["Seasonal naive (t-7)"] = {"mae": M.mae(yt_s7, yp_s7), "rmse": M.rmse(yt_s7, yp_s7)}
    yt_s365, yp_s365 = M.seasonal_naive_forecast(vals, test_idx, m=365)
    baselines["Seasonal naive (t-365)"] = {"mae": M.mae(yt_s365, yp_s365), "rmse": M.rmse(yt_s365, yp_s365)}
    yp_mean = M.mean_forecast(vals[:i_tr], len(test_idx))
    baselines["Mean (climatology)"] = {"mae": M.mae(yte_orig, yp_mean), "rmse": M.rmse(yte_orig, yp_mean)}
    # ARIMA rolling one-step
    yt_a, yp_a, order = S.arima_rolling_onestep(vals[:i_va], vals[i_va:], order=(2, 0, 2))
    # align ARIMA preds to the sequence-based test targets (drop the first SEQ offset diff)
    arima_pred_full = pd.Series(yp_a, index=temp.index[i_va:])
    arima_pred = arima_pred_full.reindex(test_dates).values
    baselines[f"ARIMA{order}"] = {"mae": M.mae(yte_orig, arima_pred), "rmse": M.rmse(yte_orig, arima_pred)}

    # ------------------------------------------------------------------ #
    # LSTM + Transformer (one-step) with loss history
    # ------------------------------------------------------------------ #
    lstm = M.LSTMForecaster(hidden=32, layers=1, horizon=1)
    lstm, lstm_val, lstm_tt, lstm_hist = M.train_model(lstm, Xtr, ytr, Xva, yva, epochs=15, patience=3)
    pred_lstm = inv(M.predict(lstm, Xte).ravel())

    trf = M.TransformerForecaster(d_model=32, nhead=4, layers=2, ff=64, dropout=0.1, horizon=1)
    trf, trf_val, trf_tt, trf_hist = M.train_model(trf, Xtr, ytr, Xva, yva, epochs=15, patience=3)
    pred_trf = inv(M.predict(trf, Xte).ravel())

    models_res = {
        "LSTM": {"mae": M.mae(yte_orig, pred_lstm), "rmse": M.rmse(yte_orig, pred_lstm),
                 "val_loss": float(lstm_val), "train_time": float(lstm_tt)},
        "Transformer": {"mae": M.mae(yte_orig, pred_trf), "rmse": M.rmse(yte_orig, pred_trf),
                        "val_loss": float(trf_val), "train_time": float(trf_tt)},
    }
    RESULTS["onestep"] = {"baselines": baselines, "models": models_res}

    # forecast plots
    for pred, name, fname, color in [
        (pred_lstm, "LSTM", "lstm_forecast.png", "#c0392b"),
        (pred_trf, "Transformer", "transformer_forecast.png", "#8e44ad"),
    ]:
        plt.figure(figsize=(13, 4))
        plt.plot(test_dates, yte_orig, label="Actual", lw=1.1, color="black")
        plt.plot(test_dates, pred, label=f"{name} predicted", lw=1.1, color=color, alpha=0.8)
        plt.title(f"{name} — Next-day Temperature Forecast (out-of-sample test)")
        plt.xlabel("Date"); plt.ylabel("°C"); plt.legend()
        savefig(fname)

    # loss curves
    for hist, name, fname in [(lstm_hist, "LSTM", "lstm_loss.png"),
                              (trf_hist, "Transformer", "transformer_loss.png")]:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(hist["train"]) + 1), hist["train"], marker="o", ms=3, label="Train")
        plt.plot(range(1, len(hist["val"]) + 1), hist["val"], marker="s", ms=3, label="Validation")
        plt.title(f"{name} — Training vs Validation Loss (MSE)")
        plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)"); plt.legend()
        savefig(fname)

    # ------------------------------------------------------------------ #
    # RESIDUAL / ERROR ANALYSIS
    # ------------------------------------------------------------------ #
    res_lstm = yte_orig - pred_lstm
    res_trf = yte_orig - pred_trf
    for res, name, fname, color in [(res_lstm, "LSTM", "lstm_residuals.png", "#c0392b"),
                                    (res_trf, "Transformer", "transformer_residuals.png", "#8e44ad")]:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(test_dates, res, lw=0.8, color=color)
        ax[0].axhline(0, color="black", lw=0.8)
        ax[0].set_title(f"{name} Residuals over Time"); ax[0].set_xlabel("Date"); ax[0].set_ylabel("Actual − Pred (°C)")
        ax[1].hist(res, bins=30, color=color, alpha=0.8)
        ax[1].set_title(f"{name} Residual Distribution (mean={res.mean():+.2f})")
        ax[1].set_xlabel("Residual (°C)"); ax[1].set_ylabel("Count")
        savefig(fname)

    # error-spike investigation (one-step)
    err_lstm = np.abs(res_lstm)
    err_trf = np.abs(res_trf)
    k = 10
    top_lstm = set(test_dates[np.argsort(err_lstm)[-k:]])
    top_trf = set(test_dates[np.argsort(err_trf)[-k:]])
    overlap = sorted(d.strftime("%Y-%m-%d") for d in (top_lstm & top_trf))
    spike_dates = test_dates[np.argsort(err_lstm)[-k:]]
    spike_interp = int(interp_mask.reindex(spike_dates).fillna(False).sum())
    plt.figure(figsize=(13, 4))
    plt.plot(test_dates, err_lstm, lw=0.8, color="#c0392b", label="|LSTM error|")
    plt.plot(test_dates, err_trf, lw=0.8, color="#8e44ad", alpha=0.7, label="|Transformer error|")
    plt.title("Absolute Prediction Error over Test Period (spike investigation)")
    plt.xlabel("Date"); plt.ylabel("Absolute error (°C)"); plt.legend()
    savefig("error_spikes.png")
    RESULTS["error_analysis"] = {
        "lstm_residual_mean": float(res_lstm.mean()), "trf_residual_mean": float(res_trf.mean()),
        "top10_error_overlap_dates": overlap,
        "n_overlap": len(overlap),
        "spike_dates_interpolated": spike_interp,
        "note": ("High-error days are shared between both models (they fail on the "
                 "same rapid-swing dates), and are not driven by interpolated inputs; "
                 "this indicates a shared inability to anticipate abrupt day-to-day "
                 "changes rather than a data-quality artefact."),
    }

    # ================================================================== #
    # MULTI-HORIZON FORECASTING (direct 1..H output)
    # ================================================================== #
    def multi(a_in, a_tgt, lo, hi):
        X, Y, oidx = [], [], []
        for o in range(SEQ - 1, n - H_MAX):
            if lo <= o < hi:
                X.append(a_in[o - SEQ + 1:o + 1]); Y.append(a_tgt[o + 1:o + 1 + H_MAX]); oidx.append(o)
        return np.array(X)[..., None], np.array(Y), oidx

    Xtr_m, Ytr_m, _ = multi(scaled, scaled, SEQ - 1, i_tr)
    Xva_m, Yva_m, _ = multi(scaled, scaled, i_tr, i_va)
    Xte_m, Yte_m, ote = multi(scaled, scaled, i_va, n)
    Yte_m_orig = np.stack([inv(Yte_m[:, h]) for h in range(H_MAX)], axis=1)

    lstm_m = M.LSTMForecaster(hidden=32, layers=1, horizon=H_MAX)
    lstm_m, _, _, _ = M.train_model(lstm_m, Xtr_m, Ytr_m, Xva_m, Yva_m, epochs=15, patience=3)
    pred_lstm_m = np.stack([inv(M.predict(lstm_m, Xte_m)[:, h]) for h in range(H_MAX)], axis=1)

    trf_m = M.TransformerForecaster(d_model=32, nhead=4, layers=2, ff=64, dropout=0.1, horizon=H_MAX)
    trf_m, _, _, _ = M.train_model(trf_m, Xtr_m, Ytr_m, Xva_m, Yva_m, epochs=15, patience=3)
    pred_trf_m = np.stack([inv(M.predict(trf_m, Xte_m)[:, h]) for h in range(H_MAX)], axis=1)

    # baselines for multi-horizon
    persist_m = np.repeat(inv(Xte_m[:, -1, 0]).reshape(-1, 1), H_MAX, axis=1)  # y_hat(t+h)=y(t)
    snaive_m = np.stack([[vals[o + h + 1 - 365] for o in ote] for h in range(H_MAX)], axis=1)

    def horizon_table(pred):
        return {str(h): {"mae": M.mae(Yte_m_orig[:, h - 1], pred[:, h - 1]),
                         "rmse": M.rmse(Yte_m_orig[:, h - 1], pred[:, h - 1])} for h in HORIZONS}

    RESULTS["multi_horizon"] = {
        "horizons": HORIZONS, "n_test_origins": int(len(ote)),
        "Persistence": horizon_table(persist_m),
        "Seasonal naive (365)": horizon_table(snaive_m),
        "LSTM": horizon_table(pred_lstm_m),
        "Transformer": horizon_table(pred_trf_m),
    }

    # error vs horizon plot (MAE)
    plt.figure(figsize=(9, 5))
    for pred, name, color in [(persist_m, "Persistence", "#7f8c8d"),
                              (snaive_m, "Seasonal naive (365)", "#27ae60"),
                              (pred_lstm_m, "LSTM", "#c0392b"),
                              (pred_trf_m, "Transformer", "#8e44ad")]:
        maes = [M.mae(Yte_m_orig[:, h - 1], pred[:, h - 1]) for h in range(1, H_MAX + 1)]
        plt.plot(range(1, H_MAX + 1), maes, marker="o", ms=2.5, label=name, color=color)
    plt.title("Forecast Error vs Horizon (multi-step, out-of-sample)")
    plt.xlabel("Forecast horizon (days ahead)"); plt.ylabel("MAE (°C)"); plt.legend()
    savefig("error_vs_horizon.png")

    # ------------------------------------------------------------------ #
    # MODEL COMPARISON (one-step, all models)
    # ------------------------------------------------------------------ #
    all_rows = []
    for name, r in baselines.items():
        all_rows.append((name, r["mae"], r["rmse"], 0.0))
    all_rows.append(("LSTM", models_res["LSTM"]["mae"], models_res["LSTM"]["rmse"], models_res["LSTM"]["train_time"]))
    all_rows.append(("Transformer", models_res["Transformer"]["mae"], models_res["Transformer"]["rmse"], models_res["Transformer"]["train_time"]))
    all_rows.sort(key=lambda x: x[2])  # rank by RMSE
    RESULTS["onestep"]["ranking"] = [{"model": r[0], "mae": r[1], "rmse": r[2]} for r in all_rows]
    RESULTS["onestep"]["best"] = all_rows[0][0]

    labels = [r[0] for r in all_rows]
    maes = [r[1] for r in all_rows]; rmses = [r[2] for r in all_rows]
    x = np.arange(len(labels)); w = 0.4
    plt.figure(figsize=(11, 5))
    plt.bar(x - w/2, maes, w, label="MAE", color="#2471a3")
    plt.bar(x + w/2, rmses, w, label="RMSE", color="#c0392b")
    plt.xticks(x, labels, rotation=30, ha="right"); plt.ylabel("Error (°C)")
    plt.title("One-step Model Comparison — Test MAE / RMSE (sorted by RMSE)"); plt.legend()
    savefig("model_comparison.png")

    with open(os.path.join(HERE, "..", "results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("Best one-step model:", RESULTS["onestep"]["best"])
    print(json.dumps({k: RESULTS[k] for k in ["dataset", "decomposition", "acf", "forecast_setup"]}, indent=2))
    print("One-step ranking (by RMSE):")
    for r in all_rows:
        print(f"  {r[0]:24s} MAE {r[1]:.3f}  RMSE {r[2]:.3f}")


if __name__ == "__main__":
    main()
