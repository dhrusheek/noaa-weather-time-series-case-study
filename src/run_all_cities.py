"""Run the complete analysis for all configured cities and compare forecasting."""
import csv
import copy
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_analysis
from cities import CITIES

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures", "city_comparison")


def write_report(rows, results):
    """Write a concise cross-city report from the complete result objects."""
    lines = [
        "# NOAA Weather Forecasting Across Major Indian Cities", "",
        "## Scope and method", "",
        "Daily NOAA GSOD data (2010–2024) were analyzed for the eight configured airport/observatory stations. Each city uses the same cleaning rules, 70/15/15 chronological split, 30-day lookback, five one-step baselines (including rolling ARIMA), LSTM, Transformer, and direct 1–30 day LSTM/Transformer forecasts. All scores below are from each station's held-out test period.", "",
        "Absolute MAE/RMSE compare errors in °C. Skill vs persistence is `100 × (1 − model MAE / persistence MAE)`; positive values improve on simply predicting yesterday's temperature. City ranking by raw error is influenced by each location's temperature variability, so read it alongside relative skill.", "",
        "## Stations and data coverage", "",
        "| City | GSOD station | Observations | Date range | Test period | Missing daily mean temperature |", "|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        ds = results[row["city"]]["dataset"]
        lines.append(f"| {row['city']} | `{row['station_id']}` ({row['station_name']}) | {row['n_obs']:,} | {row['start']}–{row['end']} | {row['test_start']}–{row['test_end']} | {row['missing_temperature']} |")
    lines += ["", "## One-day-ahead results", "", "Lower MAE/RMSE is better. The best model is ranked by RMSE across the five baselines and two neural models.", "", "| City | Persistence MAE | LSTM MAE | Transformer MAE | Best model | Best MAE | Best RMSE | Best-model skill vs persistence |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['city']} | {row['persistence_mae']:.3f} | {row['lstm_mae']:.3f} | {row['transformer_mae']:.3f} | {row['best_one_step_model']} | {row['best_one_step_mae']:.3f} | {row['best_one_step_rmse']:.3f} | {row['best_skill_vs_persistence_mae_pct']:+.1f}% |")
    lines += ["", "## Transformer forecast errors by lead time", "", "| City | Day 1 | Day 7 | Day 14 | Day 21 | Day 30 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        horizon = results[row["city"]]["multi_horizon"]
        scores = " | ".join(f"{horizon['Transformer'][str(h)]['mae']:.3f}" for h in horizon["horizons"])
        lines.append(f"| {row['city']} | {scores} |")
    lines += ["", "## Time-series characteristics", "", "| City | Seasonal strength | Linear trend (°C/decade) | Lag-1 ACF | Lag-365 ACF |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['city']} | {row['seasonal_strength']:.3f} | {row['trend_c_per_decade']:+.3f} | {row['acf_lag1']:.3f} | {row['acf_lag365']:.3f} |")
    lines += ["", "## Interpretation limits", "", "Each location is represented by one airport/observatory station. Station microclimate and measurement coverage differ, even though the analysis settings and 2010–2024 calendar window are held consistent. Skill vs persistence is a useful relative measure, but it does not remove all station-to-station differences. This is a single chronological holdout, not rolling-origin validation; precipitation is analyzed but is not a forecast input. Full stationarity tests, ACF/PACF values, figures, residual analyses, and all model metrics are in `results/cities/<city>.json` and `figures/cities/<city>/`.", ""]
    path = os.path.join(ROOT, "report", "city_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main(aggregate_only=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    all_results = {}
    if aggregate_only:
        for city in CITIES:
            path = os.path.join(RESULTS_DIR, "cities", f"{city.lower()}.json")
            with open(path, encoding="utf-8") as f:
                all_results[city] = json.load(f)
    else:
        for city, info in CITIES.items():
            print(f"\n{'=' * 16} {city} {'=' * 16}")
            raw_pattern = os.path.join(ROOT, "data", "raw", f"{info['station']}_*.csv")
            if not glob.glob(raw_pattern):
                raise FileNotFoundError(f"No downloaded NOAA data for {city}; run src/download_data.py first")
            all_results[city] = copy.deepcopy(run_analysis.main(city, info["station"]))

    rows = []
    for city, result in all_results.items():
        persistence = result["onestep"]["baselines"]["Persistence (t-1)"]
        models = {**result["onestep"]["baselines"], **result["onestep"]["models"]}
        best = min(models, key=lambda name: models[name]["rmse"])
        mult = result["multi_horizon"]
        horizons = mult["horizons"]
        transformer_mae = [mult["Transformer"][str(h)]["mae"] for h in horizons]
        persistence_mae = [mult["Persistence"][str(h)]["mae"] for h in horizons]
        row = {
            "city": city, "station_id": result["station"]["id"],
            "station_name": result["station"]["name"], "n_obs": result["dataset"]["n_obs"],
            "start": result["dataset"]["start"], "end": result["dataset"]["end"],
            "missing_temperature": result["dataset"]["missing_temperature_c"],
            "test_start": result["forecast_setup"]["test_start"], "test_end": result["forecast_setup"]["test_end"],
            "persistence_mae": persistence["mae"], "persistence_rmse": persistence["rmse"],
            "lstm_mae": result["onestep"]["models"]["LSTM"]["mae"],
            "lstm_rmse": result["onestep"]["models"]["LSTM"]["rmse"],
            "transformer_mae": result["onestep"]["models"]["Transformer"]["mae"],
            "transformer_rmse": result["onestep"]["models"]["Transformer"]["rmse"],
            "best_one_step_model": best, "best_one_step_mae": models[best]["mae"],
            "best_one_step_rmse": models[best]["rmse"],
            "best_skill_vs_persistence_mae_pct": 100 * (1 - models[best]["mae"] / persistence["mae"]),
            "transformer_skill_vs_persistence_mae_pct": 100 * (1 - result["onestep"]["models"]["Transformer"]["mae"] / persistence["mae"]),
            "transformer_mean_horizon_mae": float(np.mean(transformer_mae)),
            "persistence_mean_horizon_mae": float(np.mean(persistence_mae)),
            "transformer_horizon_skill_pct": 100 * (1 - np.mean(transformer_mae) / np.mean(persistence_mae)),
            "seasonal_strength": result["decomposition"]["seasonal_strength"],
            "trend_c_per_decade": result["decomposition"]["linear_slope_per_decade"],
            "acf_lag1": result["acf"]["lag1_raw"], "acf_lag365": result["acf"]["acf_lag365_raw"],
        }
        for model_name, metrics in result["onestep"]["baselines"].items():
            slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
            row[f"{slug}_mae"] = metrics["mae"]
            row[f"{slug}_rmse"] = metrics["rmse"]
        rows.append(row)
    rows.sort(key=lambda row: row["best_one_step_rmse"])
    payload = {
        "scope": "NOAA GSOD daily mean temperature; same 2010-2024 period, preprocessing, chronological splits, models, and settings per city",
        "cities": list(CITIES),
        "ranking_note": "Lower test MAE/RMSE means more accurate forecasts. Skill percentages compare each model against that city's persistence baseline.",
        "cities_ranked_by_best_one_step_rmse": rows,
        "per_city_results": all_results,
    }
    out_json = os.path.join(RESULTS_DIR, "city_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    out_csv = os.path.join(RESULTS_DIR, "city_comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = write_report(rows, all_results)

    labels = [row["city"] for row in rows]
    x = np.arange(len(labels)); width = 0.25
    plt.figure(figsize=(12, 6))
    for offset, key, label in [(-1.5 * width, "persistence_mae", "Persistence"), (-0.5 * width, "arima_2_0_2_mae", "ARIMA(2,0,2)"), (0.5 * width, "lstm_mae", "LSTM"), (1.5 * width, "transformer_mae", "Transformer")]:
        plt.bar(x + offset, [row[key] for row in rows], width, label=label)
    plt.xticks(x, labels, rotation=25, ha="right"); plt.ylabel("Test MAE (°C)")
    plt.title("City Comparison: One-step Forecast Error"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "onestep_mae.png"), dpi=160); plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(labels, [row["best_skill_vs_persistence_mae_pct"] for row in rows], color="#2471a3")
    plt.axhline(0, color="black", lw=0.8); plt.xticks(rotation=25, ha="right")
    plt.ylabel("Best model MAE improvement over persistence (%)")
    plt.title("Relative Forecasting Gain by City"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "skill_vs_persistence.png"), dpi=160); plt.close()

    plt.figure(figsize=(11, 5))
    for city, result in all_results.items():
        horizon = result["multi_horizon"]
        plt.plot(horizon["horizons"], [horizon["Transformer"][str(h)]["mae"] for h in horizon["horizons"]], marker="o", label=city)
    plt.xlabel("Forecast horizon (days)"); plt.ylabel("Transformer test MAE (°C)")
    plt.title("Transformer Forecast Error by City and Horizon"); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "transformer_horizon_mae.png"), dpi=160); plt.close()

    print(f"\nSaved {out_json}, {out_csv}, and {report_path}")
    print("Ranked by best one-step test RMSE:")
    for row in rows:
        print(f"  {row['city']:12s} {row['best_one_step_model']:20s} RMSE={row['best_one_step_rmse']:.3f} C; gain vs persistence={row['best_skill_vs_persistence_mae_pct']:+.1f}%")
    return payload


if __name__ == "__main__":
    main(aggregate_only="--aggregate-only" in sys.argv[1:])
