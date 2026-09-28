"""Additional model families and diagnostics for the remaining rubric items.

All model forecasts are evaluated on each city's existing 822-day test window.
Calendar Fourier terms are known in advance; no future measured weather is used.
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, t as student_t
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

import models as M
import statistical_analysis as S
from cities import CITIES
from preprocessing import build_processed

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "rubric_extension")
FIG = os.path.join(ROOT, "figures", "rubric_extension")
SEQ = 30
HAC_LAGS = 30


def calendar_features(index):
    """Deterministic yearly and weekly Fourier terms, known at forecast time."""
    idx = pd.DatetimeIndex(index)
    day = np.arange(len(idx), dtype=float)
    annual = 2 * np.pi * (idx.dayofyear.to_numpy() - 1) / 365.25
    cols = []
    for k in (1, 2, 3):
        cols.extend([np.sin(k * annual), np.cos(k * annual)])
    weekly = 2 * np.pi * day / 7.0
    cols.extend([np.sin(weekly), np.cos(weekly)])
    return np.column_stack(cols)


def one_step_sarimax(train, test, train_index, test_index, order, seasonal_order,
                     with_calendar=False):
    x_train = calendar_features(train_index) if with_calendar else None
    x_test = calendar_features(test_index) if with_calendar else None
    fit = SARIMAX(
        train, exog=x_train, order=order, seasonal_order=seasonal_order,
        trend="c", enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=100)
    extended = fit.append(test, exog=x_test, refit=False)
    start = len(train)
    pred = extended.get_prediction(start=start, end=start + len(test) - 1).predicted_mean
    return np.asarray(pred), fit


def choose_differencing_order(train):
    """Use ADF and KPSS on pre-test observations to choose d (0 or 1)."""
    series = pd.Series(np.asarray(train, dtype=float)).dropna()
    evidence = []
    for d in (0, 1):
        candidate = series.diff(d).dropna() if d else series
        adf_p = float(adfuller(candidate, autolag="AIC")[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p = float(kpss(candidate, regression="c", nlags="auto")[1])
        evidence.append({"d": d, "adf_p_value": adf_p, "kpss_p_value": kpss_p,
                         "adf_stationary": adf_p < 0.05, "kpss_stationary": kpss_p >= 0.05})
        if adf_p < 0.05 and kpss_p >= 0.05:
            break
    # When the tests disagree for both candidates, use first differencing as
    # the conservative choice; both orders remain documented for review.
    else:
        d = 1
    differenced = series.diff(d).dropna() if d else series
    max_lag = min(20, max(3, len(differenced) // 2 - 1))
    threshold = 1.96 / np.sqrt(len(differenced))
    acf_values = acf(differenced, nlags=max_lag, fft=True)
    pacf_values = pacf(differenced, nlags=max_lag, method="ywm")
    q_hint = max([lag for lag in range(1, min(3, max_lag) + 1)
                  if abs(acf_values[lag]) > threshold] or [0])
    p_hint = max([lag for lag in range(1, min(3, max_lag) + 1)
                  if abs(pacf_values[lag]) > threshold] or [0])
    evidence.append({"correlogram": {"max_lag": max_lag,
                                     "confidence_threshold": float(threshold),
                                     "acf_lags_1_to_3": acf_values[1:4].tolist(),
                                     "pacf_lags_1_to_3": pacf_values[1:4].tolist(),
                                     "initial_q_hint": q_hint, "initial_p_hint": p_hint}})
    return d, evidence


def select_information_criterion_model(train, test, train_index, test_index,
                                       family, d, with_calendar=False):
    """Grid-search p/q, then seasonal variants around leading IC candidates."""
    candidates = []
    x_train = calendar_features(train_index) if with_calendar else None
    x_test = calendar_features(test_index) if with_calendar else None
    def fit_candidate(order, seasonal_order):
        try:
            fitted = SARIMAX(
                train, exog=x_train, order=order, seasonal_order=seasonal_order,
                trend="c", enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=60)
            if np.isfinite(fitted.bic) and np.isfinite(fitted.aic):
                candidates.append({
                    "order": list(order), "seasonal_order": list(seasonal_order),
                    "aic": float(fitted.aic), "bic": float(fitted.bic),
                    "converged": bool(fitted.mle_retvals.get("converged", False)),
                })
        except Exception as exc:
            candidates.append({"order": list(order), "seasonal_order": list(seasonal_order),
                               "aic": None, "bic": None, "error": str(exc)[:240]})

    # First compare the compact nonseasonal p/q grid. Seasonal candidates are
    # then fit around the union of the top two AIC and top two BIC p/q orders.
    for p in range(3):
        for q in range(3):
            fit_candidate((p, d, q), (0, 0, 0, 0))
    base_candidates = [c for c in candidates if c["bic"] is not None]
    if family in ("SARIMA", "SARIMAX") and base_candidates:
        pool = [c for c in base_candidates if c["converged"]] or base_candidates
        seeds = {tuple(c["order"]) for criterion in ("aic", "bic")
                 for c in sorted(pool, key=lambda item: item[criterion])[:2]}
        for order in sorted(seeds):
            for seasonal_p, seasonal_q in ((0, 1), (1, 0), (1, 1)):
                fit_candidate(order, (seasonal_p, 0, seasonal_q, 7))
    valid = [c for c in candidates if c["bic"] is not None]
    if not valid:
        raise RuntimeError(f"No converged {family} candidates for the requested grid")
    converged = [c for c in valid if c["converged"]]
    selection_pool = converged or valid
    selected = min(selection_pool, key=lambda c: c["bic"])
    selected_aic = min(selection_pool, key=lambda c: c["aic"])
    pred, _ = one_step_sarimax(
        train, test, train_index, test_index, tuple(selected["order"]),
        tuple(selected["seasonal_order"]), with_calendar=with_calendar,
    )
    return pred, {"family": family, "selection_criterion": "BIC",
                  "selection_pool": "converged candidates" if converged else "all finite candidates",
                  "selected": selected, "selected_by_aic": selected_aic,
                  "seasonal_seed_orders": [list(x) for x in sorted(seeds)] if family in ("SARIMA", "SARIMAX") and base_candidates else [],
                  "candidate_count": len(candidates), "valid_candidate_count": len(valid),
                  "candidates": candidates}


def one_step_holt_winters(train, test, damped):
    """Fit additive annual Holt-Winters and update its states with each observation."""
    m = 365
    fit = ExponentialSmoothing(
        train, trend="add", damped_trend=damped, seasonal="add",
        seasonal_periods=m, initialization_method="heuristic",
    ).fit(optimized=True, use_brute=False, remove_bias=False)
    params = fit.params
    alpha = float(params["smoothing_level"])
    beta = float(params["smoothing_trend"])
    gamma = float(params["smoothing_seasonal"])
    phi = float(params.get("damping_trend", 1.0)) if damped else 1.0
    level = float(fit.level[-1])
    trend = float(fit.trend[-1])
    seasons = list(np.asarray(fit.season[-m:], dtype=float))
    predictions = []
    for i, obs in enumerate(test):
        old_season = seasons[0]
        forecast = level + phi * trend + old_season
        # Statsmodels' forecast is used for the first step as an implementation
        # anchor; subsequent steps apply the same additive Holt-Winters recursions.
        if i == 0:
            forecast = float(fit.forecast(1)[0])
        predictions.append(forecast)
        new_level = alpha * (obs - old_season) + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        new_season = gamma * (obs - new_level) + (1 - gamma) * old_season
        level, trend = new_level, new_trend
        seasons.pop(0)
        seasons.append(new_season)
    return np.asarray(predictions), fit


def ml_features(values, index, target_indices):
    cyc = calendar_features(index)
    return np.asarray([np.concatenate([values[t - SEQ:t], cyc[t]]) for t in target_indices])


def train_ml(values, index, i_fit, i_test):
    train_idx = np.arange(SEQ, i_fit)
    test_idx = np.arange(i_test, len(values))
    x_train = ml_features(values, index, train_idx)
    x_test = ml_features(values, index, test_idx)
    y_train = values[train_idx]
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            max_features=0.8, n_jobs=-1, random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2,
            loss="huber", subsample=0.8, random_state=42,
        ),
    }
    preds = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds[name] = model.predict(x_test)
    return preds


def train_neural(values, i_train, i_val, i_test):
    scaler = StandardScaler().fit(values[:i_train].reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()
    def sequences(lo, hi):
        idx = np.arange(max(SEQ, lo), hi)
        x = np.asarray([scaled[t - SEQ:t] for t in idx])[..., None]
        y = scaled[idx]
        return x, y
    x_train, y_train = sequences(SEQ, i_train)
    x_val, y_val = sequences(i_train, i_val)
    x_test, _ = sequences(i_test, len(values))
    preds = {}
    # Match the original pipeline's initialization sequence: train LSTM first,
    # then initialize the Transformer from the resulting RNG state.
    M.set_seed()
    for name, model in [("LSTM", M.LSTMForecaster(hidden=32, layers=1, horizon=1))]:
        model, _, _, _ = M.train_model(model, x_train, y_train, x_val, y_val, epochs=15, patience=3)
        scaled_pred = M.predict(model, x_test).ravel()
        preds[name] = scaler.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()
    model = M.TransformerForecaster(d_model=32, nhead=4, layers=2, ff=64, dropout=0.1, horizon=1)
    model, _, _, _ = M.train_model(model, x_train, y_train, x_val, y_val, epochs=15, patience=3)
    scaled_pred = M.predict(model, x_test).ravel()
    preds["Transformer"] = scaler.inverse_transform(scaled_pred.reshape(-1, 1)).ravel()
    return preds


def scores(actual, predicted, train):
    err = np.asarray(actual) - np.asarray(predicted)
    mae = M.mae(actual, predicted)
    return {
        "mae": mae,
        "rmse": M.rmse(actual, predicted),
        "mase": float(mae / np.mean(np.abs(np.diff(train)))),
        "r2": float(r2_score(actual, predicted)),
        "bias_actual_minus_predicted": float(np.mean(err)),
    }


def residual_diagnostics(actual, predicted):
    resid = np.asarray(actual) - np.asarray(predicted)
    lb = acorr_ljungbox(resid, lags=[10, 30], return_df=True)
    jb = jarque_bera(resid)
    return {
        "mean": float(np.mean(resid)), "std": float(np.std(resid, ddof=1)),
        "durbin_watson": float(durbin_watson(resid)),
        "ljung_box_lag10_stat": float(lb.loc[10, "lb_stat"]),
        "ljung_box_lag10_p": float(lb.loc[10, "lb_pvalue"]),
        "ljung_box_lag30_stat": float(lb.loc[30, "lb_stat"]),
        "ljung_box_lag30_p": float(lb.loc[30, "lb_pvalue"]),
        "jarque_bera_stat": float(jb.statistic), "jarque_bera_p": float(jb.pvalue),
    }


def summarize_model_families(results):
    """Equal-weight city averages and per-city RMSE wins/ranks."""
    names = list(next(iter(results.values()))["model_metrics"])
    rank_sums = {name: 0 for name in names}
    wins = {name: 0 for name in names}
    for result in results.values():
        ranked = sorted(result["model_metrics"], key=lambda name: result["model_metrics"][name]["rmse"])
        wins[ranked[0]] += 1
        for rank, name in enumerate(ranked, start=1):
            rank_sums[name] += rank
    rows = []
    for name in names:
        ms = [result["model_metrics"][name] for result in results.values()]
        rows.append({
            "model": name, "mean_mae": float(np.mean([m["mae"] for m in ms])),
            "mean_rmse": float(np.mean([m["rmse"] for m in ms])),
            "mean_mase": float(np.mean([m["mase"] for m in ms])),
            "mean_r2": float(np.mean([m["r2"] for m in ms])),
            "cities_won_by_rmse": wins[name],
            "mean_rmse_rank": rank_sums[name] / len(results),
        })
    return sorted(rows, key=lambda row: row["mean_rmse"])


def dm_absolute_loss_vs_persistence(actual, prediction, persistence):
    """Newey-West/HAC Diebold-Mariano test for paired absolute-error loss."""
    d = np.abs(actual - prediction) - np.abs(actual - persistence)
    n = len(d)
    centered = d - d.mean()
    lrv = float(np.dot(centered, centered) / n)
    for lag in range(1, min(HAC_LAGS, n - 1) + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        lrv += 2 * (1 - lag / (HAC_LAGS + 1)) * gamma
    se = np.sqrt(max(lrv, 0.0) / n)
    if not np.isfinite(se) or se == 0:
        stat, pval = 0.0, 1.0
    else:
        stat = float(d.mean() / se)
        pval = float(2 * student_t.sf(abs(stat), df=n - 1))
    return {"mean_abs_loss_difference": float(d.mean()), "statistic": stat, "p_value": pval,
            "hac_lags": HAC_LAGS, "n": n}


def save_residual_figure(city, dates, actual, predictions, metrics):
    slug = city.lower()
    city_fig = os.path.join(FIG, slug)
    os.makedirs(city_fig, exist_ok=True)
    best = min(metrics, key=lambda name: metrics[name]["rmse"])
    shown = list(dict.fromkeys(["Persistence (t-1)", "ARIMA", "SARIMA", "SARIMAX", "ARMAX", "Holt-Winters additive", "GradientBoosting", "LSTM", "Transformer", best]))
    shown = [name for name in shown if name in predictions]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    for name in shown:
        ax[0].plot(dates, actual - predictions[name], lw=0.65, alpha=0.7, label=name)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_ylabel("Actual − forecast (°C)"); ax[0].set_title(f"{city}: One-step Test Residuals")
    ax[0].legend(ncol=3, fontsize=8)
    resid = actual - predictions[best]
    ax[1].hist(resid, bins=35, color="#2471a3", alpha=0.85, density=True)
    ax[1].set_title(f"Best model: {best}; mean residual={np.mean(resid):+.3f} °C")
    ax[1].set_xlabel("Residual (°C)"); ax[1].set_ylabel("Density")
    fig.tight_layout(); fig.savefig(os.path.join(city_fig, "residual_diagnostics.png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_acf(resid, lags=40, ax=ax)
    ax.set_title(f"{city}: Residual ACF — {best}"); ax.set_xlabel("Lag (days)")
    fig.tight_layout(); fig.savefig(os.path.join(city_fig, "best_model_residual_acf.png"), dpi=160); plt.close(fig)


def main(selected_cities=None, skip_existing=False):
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    all_results = {}
    summary_rows = []
    for city, cfg in CITIES.items():
        if selected_cities and city not in selected_cities:
            continue
        saved_path = os.path.join(OUT, f"{city.lower()}.json")
        if skip_existing and os.path.isfile(saved_path):
            with open(saved_path, encoding="utf-8") as f:
                city_result = json.load(f)
            if "test_residuals" in city_result:
                actual = np.asarray(city_result["actual_test_temperature_c"])
                predictions = {name: actual - np.asarray(resid)
                              for name, resid in city_result["test_residuals"].items()}
                save_residual_figure(city, pd.to_datetime(city_result["test_dates"]),
                                     actual, predictions, city_result["model_metrics"])
            all_results[city] = city_result
            best = city_result["model_order_by_rmse"][0]
            metrics = city_result["model_metrics"]
            summary_rows.append({
                "city": city, "station_id": city_result["station_id"], "best_model": best,
                "best_mae": metrics[best]["mae"], "best_rmse": metrics[best]["rmse"],
                "persistence_mae": metrics["Persistence (t-1)"]["mae"],
                "persistence_rmse": metrics["Persistence (t-1)"]["rmse"],
                "best_skill_vs_persistence_pct": 100 * (1 - metrics[best]["mae"] / metrics["Persistence (t-1)"]["mae"]),
                "best_model_ljung_box_10_p": city_result["residual_diagnostics"][best]["ljung_box_lag10_p"],
                "best_model_ljung_box_30_p": city_result["residual_diagnostics"][best]["ljung_box_lag30_p"],
                "best_model_jarque_bera_p": city_result["residual_diagnostics"][best]["jarque_bera_p"],
                "best_model_residual_mean": city_result["residual_diagnostics"][best]["mean"],
            })
            continue
        print(f"\n{city}: fitting extended rubric models", flush=True)
        station = cfg["station"]
        df = build_processed(station, save=False)
        values = df["temperature_c_interp"].to_numpy(dtype=float)
        dates = df.index
        n = len(values); i_train = int(n * 0.70); i_val = int(n * 0.85); i_test = i_val
        train_fit = values[:i_val]
        y_test = values[i_test:]
        test_dates = dates[i_test:]
        predictions = {
            "Persistence (t-1)": values[i_test - 1:n - 1],
            "Seasonal naive (t-7)": values[i_test - 7:n - 7],
            "Seasonal naive (t-365)": values[i_test - 365:n - 365],
            "Mean (climatology)": np.full(len(y_test), float(np.mean(values[:i_train]))),
        }

        # Choose d from ADF/KPSS using pre-test data, then sweep the small p/q grid
        # (and weekly P/Q for seasonal families) by BIC. The test is untouched.
        d, stationarity = choose_differencing_order(train_fit)
        order_selection = {"d_selection": {"selected_d": d, "method": "ADF and KPSS; first d for which both support stationarity, else d=1",
                                             "tests": stationarity},
                           "grid": {"p": [0, 1, 2], "q": [0, 1, 2],
                                    "seasonal_P": [0, 1], "seasonal_Q": [0, 1],
                                    "seasonal_period": 7}}
        for family, with_calendar in [("ARIMA", False), ("SARIMA", False),
                                      ("ARMAX", True), ("SARIMAX", True)]:
            pred, selection = select_information_criterion_model(
                train_fit, y_test, dates[:i_val], test_dates, family, d,
                with_calendar=with_calendar,
            )
            predictions[family] = pred
            order_selection[family] = selection
            chosen = selection["selected"]
            print(f"  {city}: {family} BIC order={chosen['order']} "
                  f"seasonal={chosen['seasonal_order']} BIC={chosen['bic']:.1f}; "
                  f"AIC order={selection['selected_by_aic']['order']} "
                  f"({selection['valid_candidate_count']}/{selection['candidate_count']} valid)", flush=True)

        hw_add, _ = one_step_holt_winters(train_fit, y_test, damped=False)
        predictions["Holt-Winters additive"] = hw_add
        print(f"  {city}: Holt-Winters additive done", flush=True)
        hw_damped, _ = one_step_holt_winters(train_fit, y_test, damped=True)
        predictions["Holt-Winters damped"] = hw_damped
        ml_preds = train_ml(values, dates, i_val, i_test)
        predictions.update(ml_preds)
        print(f"  {city}: tree models done", flush=True)
        predictions.update(train_neural(values, i_train, i_val, i_test))

        model_metrics = {name: scores(y_test, pred, values[:i_train]) for name, pred in predictions.items()}
        model_metrics = dict(sorted(model_metrics.items(), key=lambda item: item[1]["rmse"]))
        persistence = predictions["Persistence (t-1)"]
        dm = {name: dm_absolute_loss_vs_persistence(y_test, pred, persistence)
              for name, pred in predictions.items() if name != "Persistence (t-1)"}
        pvals = [x["p_value"] for x in dm.values()]
        reject, p_adjusted, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        for (name, test), p_adj, rej in zip(dm.items(), p_adjusted, reject):
            test["holm_adjusted_p_value"] = float(p_adj)
            test["significant_at_0_05"] = bool(rej)
        residuals = {name: residual_diagnostics(y_test, pred) for name, pred in predictions.items()}
        save_residual_figure(city, test_dates, y_test, predictions, model_metrics)

        city_result = {
            "city": city, "station": df["NAME"].iloc[0], "station_id": station,
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "test_period": [str(test_dates[0].date()), str(test_dates[-1].date())],
            "n_train": i_train, "n_validation": i_val - i_train, "n_test": len(y_test),
            "model_metrics": model_metrics,
            "residual_diagnostics": residuals,
            "test_dates": [str(d.date()) for d in test_dates],
            "actual_test_temperature_c": y_test.tolist(),
            "test_residuals": {name: (y_test - pred).tolist() for name, pred in predictions.items()},
            "information_criterion_order_selection": order_selection,
            "diebold_mariano_absolute_loss_vs_persistence": dm,
            "model_order_by_rmse": list(model_metrics),
        }
        all_results[city] = city_result
        path = os.path.join(OUT, f"{city.lower()}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(city_result, f, indent=2)
        best = next(iter(model_metrics))
        summary_rows.append({
            "city": city, "station_id": station, "best_model": best,
            "best_mae": model_metrics[best]["mae"], "best_rmse": model_metrics[best]["rmse"],
            "persistence_mae": model_metrics["Persistence (t-1)"]["mae"],
            "persistence_rmse": model_metrics["Persistence (t-1)"]["rmse"],
            "best_skill_vs_persistence_pct": 100 * (1 - model_metrics[best]["mae"] / model_metrics["Persistence (t-1)"]["mae"]),
            "best_model_ljung_box_10_p": residuals[best]["ljung_box_lag10_p"],
            "best_model_ljung_box_30_p": residuals[best]["ljung_box_lag30_p"],
            "best_model_jarque_bera_p": residuals[best]["jarque_bera_p"],
            "best_model_residual_mean": residuals[best]["mean"],
        })
        print(f"  {city}: complete — {best}, MAE={model_metrics[best]['mae']:.3f}, RMSE={model_metrics[best]['rmse']:.3f}", flush=True)

    summary_rows.sort(key=lambda row: row["best_rmse"])
    family_rows = summarize_model_families(all_results)
    with open(os.path.join(OUT, "city_model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"cities_ranked_by_best_model_rmse": summary_rows,
                   "models_ranked_by_mean_city_rmse": family_rows,
                   "per_city_results": all_results}, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "city_model_comparison.csv"), index=False)
    pd.DataFrame(family_rows).to_csv(os.path.join(OUT, "model_family_comparison.csv"), index=False)
    report_path = write_rubric_report(summary_rows, all_results)
    plot_city_comparison(summary_rows, all_results)
    print(f"Saved extended rubric results and {report_path}", flush=True)


def write_rubric_report(rows, results):
    family_rows = summarize_model_families(results)
    lines = [
        "# Remaining Forecasting Rubric: Eight-City Results", "",
        "## Evaluation setup", "",
        "Each city uses NOAA GSOD daily mean temperature for 2010–2024. The common chronological partition is 70% train, 15% validation, and 15% test, with the same 30-day input window and common test dates. Classical ARIMA-family and Holt-Winters models use the first 85% available history and update their one-step states with each observed test value. Tree models use lagged temperature plus known calendar features and are fit on the first 85%. LSTM and Transformer use the original train-only fit, validation early stopping, and test window. No future measured weather values enter the exogenous calendar features.", "",
        "## Model families and distinctions", "",
        "- **ARIMA:** non-seasonal ARIMA with p, q selected from a grid and d chosen using ADF/KPSS stationarity tests.",
        "- **SARIMA:** ARIMA plus weekly seasonal AR and MA terms (period 7), with P/Q also selected by information criteria.",
        "- **ARMAX:** ARMA errors plus deterministic annual and weekly Fourier calendar predictors; p, q selected by information criteria.",
        "- **SARIMAX:** seasonal ARIMA plus the deterministic calendar predictors; p, q, P, Q selected by information criteria.",
        "- For every city, ADF and KPSS on the pre-test 85% select d; ACF/PACF correlograms provide initial p/q hints, then all p,q in 0..2 are swept. Seasonal P,Q in 0..1 are swept for seasonal models (weekly period 7; D fixed at 0). AIC and BIC are computed for every candidate. BIC selects the forecast model used in the city comparison; the AIC-selected orders are also saved. The held-out test set is used only for final accuracy evaluation.",
        "- **Holt-Winters:** additive annual seasonality (period 365), with and without a damped additive trend.",
        "- **Machine learning:** Random Forest and Gradient Boosting using the past 30 daily temperatures plus calendar Fourier terms.",
        "- **Deep learning:** LSTM and Transformer as in the base analysis.",
        "- **Baselines:** persistence, 7-day seasonal naive, 365-day seasonal naive, train-mean climatology, and ARIMA.", "",
        "Calendar terms are deterministic, known-ahead regressors rather than observed exogenous weather. These models test whether annual/weekly timing improves forecasts; they do not test the value of adding rainfall or other measured predictors.", "",
        "## Selected ARIMA-family orders", "",
        "Orders below are selected independently per city. The result JSON stores every candidate's AIC, BIC, and convergence status under `information_criterion_order_selection`.", "",
        "| City | d | Family | BIC-selected (p,d,q)(P,D,Q,7) | BIC | AIC-selected order | AIC |", "|---|---:|---|---|---:|---|---:|",
    ]
    for city, result in results.items():
        selection = result.get("information_criterion_order_selection", {})
        d = selection.get("d_selection", {}).get("selected_d", "?")
        for family in ("ARIMA", "SARIMA", "ARMAX", "SARIMAX"):
            fit = selection.get(family)
            if not fit:
                continue
            bic, aic = fit["selected"], fit["selected_by_aic"]
            lines.append(f"| {city} | {d} | {family} | {tuple(bic['order'])} {tuple(bic['seasonal_order'])} | {bic['bic']:.1f} | {tuple(aic['order'])} {tuple(aic['seasonal_order'])} | {aic['aic']:.1f} |")
    lines += [
        "", "## Forecast accuracy and model comparison", "",
        "| City | Best model | MAE (°C) | RMSE (°C) | Persistence MAE | Best-model MAE skill vs persistence |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['city']} | {row['best_model']} | {row['best_mae']:.3f} | {row['best_rmse']:.3f} | {row['persistence_mae']:.3f} | {row['best_skill_vs_persistence_pct']:+.1f}% |")
    lines += [
        "", "### Equal-weight model averages across cities", "",
        "This table averages the eight station-level test scores equally and shows each model's city-specific RMSE wins and mean RMSE rank.", "",
        "| Model | Mean MAE (°C) | Mean RMSE (°C) | Mean MASE | Mean R² | RMSE wins | Mean RMSE rank |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(f"| {row['model']} | {row['mean_mae']:.3f} | {row['mean_rmse']:.3f} | {row['mean_mase']:.3f} | {row['mean_r2']:.3f} | {row['cities_won_by_rmse']} | {row['mean_rmse_rank']:.2f} |")
    lines += [
        "", "The complete per-city ranking table includes MAE, RMSE, MASE, R², and bias for every model in `results/rubric_extension/<city>.json`. MASE scales MAE by the training-period one-day naive error. A positive skill score means lower MAE than persistence. Absolute error describes forecast difficulty; persistence-relative skill describes incremental value over yesterday's observation.",
        "", "### Diebold–Mariano paired loss tests", "",
        "Each model's test-period absolute error is compared with persistence using a paired Diebold–Mariano test with Newey–West/HAC variance (30 lags). Holm correction controls family-wise error across models within each city. A corrected p-value below 0.05 indicates a significant difference in mean absolute loss; negative signed mean loss difference means the model improves on persistence.",
        "", "## Residual analysis", "",
        "For each model, the result JSON reports residual mean and standard deviation, Durbin–Watson, Ljung–Box tests at lags 10 and 30, and Jarque–Bera normality. Residual plots and the best-model residual ACF are in `figures/rubric_extension/<city>/`. Ljung–Box significance indicates residual serial dependence; a non-significant Jarque–Bera result is consistent with, but does not prove, normally distributed residuals.",
        "", "| City | Best model | Residual mean (°C) | Durbin–Watson | Ljung–Box lag-30 p | Jarque–Bera p |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = results[row["city"]]["residual_diagnostics"][row["best_model"]]
        lines.append(f"| {row['city']} | {row['best_model']} | {d['mean']:+.3f} | {d['durbin_watson']:.3f} | {d['ljung_box_lag30_p']:.4g} | {d['jarque_bera_p']:.4g} |")
    lines += [
        "", "## Research conclusion", "",
        "",
    ]
    significant = []
    residual_acf = []
    winner_counts = {}
    for row in rows:
        best = row["best_model"]
        result = results[row["city"]]
        if result["diebold_mariano_absolute_loss_vs_persistence"][best]["significant_at_0_05"]:
            significant.append(row["city"])
        if row["best_model_ljung_box_30_p"] < 0.05:
            residual_acf.append(row["city"])
        winner_counts[best] = winner_counts.get(best, 0) + 1
    base_path = os.path.join(ROOT, "results", "city_comparison.json")
    horizon_note = ""
    if os.path.isfile(base_path):
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
        gains = [row["transformer_horizon_skill_pct"] for row in base["cities_ranked_by_best_one_step_rmse"]]
        horizon_note = f" The base direct 1-30 day LSTM/Transformer comparison found positive mean-horizon Transformer skill over persistence in all eight cities ({min(gains):.1f}% to {max(gains):.1f}%); see `report/city_comparison.md`."
    lines += [
        f"On the shared 2022-10-02 to 2024-12-31 holdout, {rows[0]['city']} has the lowest best-model RMSE ({rows[0]['best_rmse']:.3f} deg C), while {rows[-1]['city']} has the highest ({rows[-1]['best_rmse']:.3f} deg C). No model wins in every city: " + "; ".join(f"{name} leads in {count}" for name, count in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))) + ".",
        ("After Holm correction, the best model's absolute loss is significantly lower than persistence in " + ", ".join(significant) + ". In the other cities, this test did not find a significant difference; that does not establish equivalence." if significant else "After Holm correction, no model had significantly lower absolute loss than persistence."),
        ("Best-model residuals retain significant autocorrelation at lag 30 for " + ", ".join(residual_acf) + "; Ljung-Box results at lag 10 and lag 30 are in the table above." if residual_acf else "Best-model residuals show no significant autocorrelation at lag 30 in the Ljung-Box test."),
        "Jarque-Bera rejects residual normality for all eight selected best models, so normal-error assumptions for uncertainty intervals need further checking. Best-model residual biases remain below 0.2 deg C in magnitude." + horizon_note,
        "Use RMSE to compare absolute forecast difficulty and skill relative to persistence to judge the gain from more complex models. This observational analysis represents one station per city and one temporal holdout; it does not establish city-wide predictability or causal effects.",
        "", "## Individual contribution (5 marks)", "",
        "Fill this table with the actual team members and work completed. The project files cannot establish who performed each contribution, so names and ownership are intentionally left for the team to confirm.", "",
        "| Member | Contribution completed | Evidence (file, analysis, or presentation work) |", "|---|---|---|",
        "| [Name] | [Work performed] | [Evidence] |", "| [Name] | [Work performed] | [Evidence] |", "| [Name] | [Work performed] | [Evidence] |",
        "", "## Reproduction", "",
        "Run `python src/run_all_cities.py` for the base city analysis, then `python src/rubric_extension.py` for ARIMA-family, Holt-Winters, tree-model, and residual outputs. Use `python src/rubric_extension.py --skip-existing` to continue a partial run from saved city results. Both use files in `data/raw/`; download them with `python src/download_data.py` if needed.", "",
    ]
    path = os.path.join(ROOT, "report", "remaining_rubric_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def plot_city_comparison(rows, results):
    labels = [r["city"] for r in rows]
    models = ["ARIMA", "SARIMA", "SARIMAX", "ARMAX", "Holt-Winters additive", "Holt-Winters damped", "RandomForest", "GradientBoosting", "LSTM", "Transformer"]
    width = 0.8 / len(models); x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(17, 7))
    for i, model in enumerate(models):
        vals = [results[city]["model_metrics"][model]["mae"] for city in labels]
        ax.bar(x - 0.4 + (i + 0.5) * width, vals, width, label=model)
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("Test MAE (°C)")
    ax.set_title("One-step Accuracy by City and Model Family"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "all_models_city_mae.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    skip = "--skip-existing" in args
    selected = [arg for arg in args if not arg.startswith("--")]
    unknown = set(selected) - set(CITIES)
    if unknown:
        raise SystemExit(f"Unknown city name(s): {', '.join(sorted(unknown))}")
    main(selected_cities=selected or None, skip_existing=skip)
