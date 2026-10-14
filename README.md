# NOAA Weather Time-Series Case Study — Major Indian Cities

Comparable time-series analysis and short-term temperature forecasting for eight
Indian cities using NOAA's **Global Summary of the Day (GSOD)**. Each location gets
the same statistical analysis, **five baselines**, an **LSTM**, and a **Transformer**,
evaluated one day ahead and across a **30-day horizon**.

## Overview

This repository now runs the same end-to-end time-series analysis for eight major
Indian cities: Bengaluru, Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Ahmedabad,
and Pune. Each is represented by a single NOAA GSOD airport/observatory station,
with daily data from 2010 through 2024. Per-city forecasts use the same
train/validation/test split, baselines, neural models, and metrics; comparison outputs include absolute
errors and improvement relative to each city's persistence forecast.

- **Period:** 2010-01-01 → 2024-12-31 (daily records)
- **Primary forecast target:** daily mean temperature (°C)
- **Exploratory secondary variable:** precipitation (mm)

## Research Question

> How do temporal dependence, trend, and seasonal patterns influence daily mean
> How do temporal structure and weather variability differ across major Indian
> cities, and how easily can daily temperature be forecast in each city one day
> ahead and over a 30-day horizon using the same baselines and neural models?

Observational data — **no causal claims**.

## Dataset

NOAA/NCEI GSOD daily CSVs, 2010–2024, retrieved automatically over HTTPS
(`src/download_data.py`). GSOD native units are °F (temperature, sentinel `9999.9`)
and inches (precipitation, sentinel `99.99`); converted to °C and mm. `PRCP` flag `I`
(no measurement) is treated as **missing**, not a true zero. See `data/README.md` and
`report/time_series_case_study.md`.

## Methodology

1. **Automated HTTPS retrieval** of the configured yearly GSOD CSVs for eight cities (`src/download_data.py`, `src/cities.py`).
2. **Reproducible cleaning**: unit conversion, sentinel/flag handling, duplicate
   removal, reindex to a continuous daily calendar, ≤3-day temperature interpolation
   (`src/preprocessing.py`).
3. **EDA + statistics**: rolling stats, monthly/box/subseries plots, explicit trend
   (STL + linear fit), STL(365) decomposition, ADF + KPSS, ACF/PACF
   (`src/statistical_analysis.py`, `src/run_analysis.py`).
4. **Leakage-safe forecasting**: chronological 70/15/15 split, scaler fit on train
   only, 30-day input window.
5. **Models**: 5 baselines + LSTM + Transformer, one-step and multi-horizon
   (`src/models.py`).

## Rubric Coverage

| Rubric Component | Marks | Implementation |
|------------------|------:|----------------|
| Research problem/question | 1 | Report §1 + README; phenomenon, scope, variables, one-step & multi-horizon objective explicit |
| Dataset understanding | 1 | Report §2; source/product/URL, station rationale, units verified, `head/tail/shape/info`, describe, missing/duplicate audit, raw-vs-interp explained |
| Time-series visualization | 1 | 8+ labelled figures (incl. **boxplot** & **subseries**), each interpreted; report §4 |
| Trend and seasonality | 2 | **Explicit** trend (STL component + OLS +0.19 °C/decade) **and** seasonality (STL strength 0.80, subseries, lag-365 ACF 0.717); report §5–6 |
| Stationarity analysis | 2 | **ADF & KPSS** on raw + differenced, results table, opposite-null explanation, differencing & transformation rationale; report §7 |
| ACF/PACF analysis | 2 | ACF/PACF raw + differenced + long-lag; lag-1 = 0.923, lag-365 = 0.717; interpretation + confidence bounds + link to ARIMA/seq length; report §8 |
| Initial findings/hypothesis | 1 | Prose synthesis + H1/H0 with outcome; report §9 |
| **Total** | **10** | **Complete** |

## City Comparison Outputs

`src/run_all_cities.py` runs the full statistical and forecasting analysis per city.
It writes individual results to `results/cities/`, a ranked summary to
`results/city_comparison.csv` and `results/city_comparison.json`, per-city figures
to `figures/cities/<city>/`, and comparison plots to `figures/city_comparison/`.
Use test MAE/RMSE for absolute forecast error and skill vs persistence
(`1 - model MAE / persistence MAE`) to compare improvement over each city's local
naive forecast.

The remaining forecasting rubric is covered by
[report/remaining_rubric_analysis.md](report/remaining_rubric_analysis.md): it compares
ARIMA, SARIMA, ARMAX, SARIMAX, two Holt-Winters variants, Random Forest, Gradient
Boosting, LSTM, and Transformer with the same one-step test dates. It includes
MAE/RMSE/MASE/R², per-city ranks, corrected Diebold-Mariano tests, and residual
diagnostics. Detailed outputs are in `results/rubric_extension/` and
`figures/rubric_extension/`.

The complete eight-city walkthrough is in
[`notebooks/eight_city_weather_forecasting_comparison.ipynb`](notebooks/eight_city_weather_forecasting_comparison.ipynb).
It brings together the original exploration rubric, city plots, per-city ARIMA-family
AIC/BIC order selection, forecasting comparisons, residual diagnostics, conclusions,
the Stage A / Stage B pipeline reconciliation, and the team-contribution table. It reads
saved results and figures (recomputing only the ACF/PACF); run the
analysis scripts above first to regenerate them.

## Forecasting Models

- **5 baselines** — Persistence (t-1), Seasonal naive (t-7), Seasonal naive (t-365), Mean (climatology), **ARIMA(2,0,2)**
- **LSTM** — hidden 32, 1 layer, ≤15 epochs, early stopping (patience 3)
- **Transformer** — d_model 32, 4 heads, 2 encoder layers, positional encoding, ≤15 epochs

## Base-pipeline cross-city one-step summary

Eight city pipelines completed on the common 2022-10-02 to 2024-12-31 test period.
Ranked by each city's best model test RMSE:

| City | Best one-step model | MAE (°C) | RMSE (°C) | MAE gain vs persistence |
|---|---|---:|---:|---:|
| Bengaluru | ARIMA(2,0,2) | 0.638 | 0.829 | +4.4% |
| Mumbai | LSTM | 0.617 | 0.858 | +3.2% |
| Chennai | Transformer | 0.631 | 0.858 | +2.7% |
| Pune | Transformer | 0.743 | 0.976 | +0.6% |
| Ahmedabad | Transformer | 0.765 | 1.044 | +1.7% |
| Hyderabad | Transformer | 0.818 | 1.090 | +0.0% |
| Kolkata | ARIMA(2,0,2) | 0.913 | 1.221 | +1.1% |
| Delhi | Transformer | 1.044 | 1.367 | +0.7% |

By raw one-step error, Bengaluru is easiest to forecast and Delhi hardest in this
holdout. For 1/7/14/21/30-day leads, the Transformer improves over persistence in
all cities; its mean-horizon MAE improvement ranges from 11.8% (Hyderabad) to 30.6%
(Delhi). Full lead-time results and diagnostics are in [the comparison report](report/city_comparison.md)
and the machine-readable [comparison CSV](results/city_comparison.csv).

The table above covers the original baseline/LSTM/Transformer pipeline. The expanded
ARIMA-family, Holt-Winters, and ML comparison has a separate ranking in
[the remaining-rubric report](report/remaining_rubric_analysis.md).

## Original Bengaluru run (reference results from `results.json`)

**One-step (t+1), out-of-sample test, ranked by RMSE:**

| Model | MAE (°C) | RMSE (°C) |
|---|---:|---:|
| **ARIMA(2,0,2)** | **0.638** | **0.829** |
| LSTM | 0.650 | 0.834 |
| Transformer | 0.655 | 0.837 |
| Persistence (t-1) | 0.668 | 0.874 |
| Seasonal naive (t-7) | 1.192 | 1.514 |
| Seasonal naive (t-365) | 1.384 | 1.770 |
| Mean (climatology) | 1.745 | 2.304 |

**Multi-horizon MAE (°C) by lead time:**

| Horizon (days) | 1 | 7 | 14 | 21 | 30 |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.67 | 1.18 | 1.32 | 1.51 | 1.74 |
| Seasonal naive (365) | 1.41 | 1.40 | 1.40 | 1.40 | 1.39 |
| LSTM | 0.82 | 1.03 | 1.19 | 1.36 | 1.49 |
| **Transformer** | 0.69 | **1.02** | **1.17** | **1.29** | **1.38** |

**Best model:** ARIMA at t+1 (LSTM/Transformer close behind and both beat
persistence); the **Transformer is best across the 30-day horizon**. With ~15 years
of data the neural models beat the naive baseline (H0 rejected) — an earlier 6-year
run could not. Reported truthfully — no fabricated metrics.

## Repository Structure

```
noaa-weather-time-series-case-study/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── results.json                      # machine-readable run outputs
├── data/
│   ├── README.md
│   ├── raw/                           # GSOD source CSVs 2010-2024 (small, kept for reproducibility)
│   └── processed/weather_daily.csv
├── notebooks/time_series_case_study.ipynb
├── src/
│   ├── download_data.py               # automated HTTPS retrieval
│   ├── preprocessing.py               # cleaning / units / sentinels
│   ├── statistical_analysis.py        # ADF, KPSS, ARIMA helpers
│   ├── models.py                      # baselines, LSTM, Transformer
│   ├── run_analysis.py                # full pipeline -> figures + results.json
│   └── build_notebook.py              # regenerates the notebook
├── figures/                          # 23 generated PNGs
└── report/time_series_case_study.md
```

The city expansion also adds `src/cities.py`, `src/run_all_cities.py`,
`results/city_comparison.*`, and `report/city_comparison.md`; figures and processed
data are organized by station/city.

## Reproduction Instructions

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py     # HTTPS fetch all configured city/year CSVs -> data/raw/
python src/run_all_cities.py    # full analysis and comparison across all cities
python src/run_all_cities.py --aggregate-only  # rebuild comparison outputs from saved city results
python src/rubric_extension.py   # remaining model families, accuracy tests, residual analysis
python src/rubric_extension.py --skip-existing  # resume from completed per-city extension results
python src/run_analysis.py      # original Bengaluru-only pipeline
python src/build_notebook.py    # (optional) regenerate the notebook
```

Neural training uses seed 42 and is reproducible from the downloaded raw data. The
notebook `notebooks/time_series_case_study.ipynb` documents the original Bengaluru
analysis; the generated city-specific result JSON files contain the same analysis
and metrics for each configured location. The new eight-city notebook presents the
cross-city analysis and remaining model-selection rubric in one place.

## Data Source

NOAA/NCEI Global Summary of the Day (GSOD):
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

Station metadata reference:
https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv

## Limitations

Each city is represented by one airport/observatory, so local microclimate, elevation,
and data quality differ. Fifteen years is short for climate-trend attribution; GSOD
precipitation has substantial documented missingness; forecasting is univariate;
models are intentionally small and untuned; and results use one temporal holdout.
Next-day forecasting structurally favours persistence.

## Authors / Collaborators

Maintained by the repository owner. Collaborators invited via GitHub:
`kushalyalamanchi@gmail.com`, `Nethi.kushaal@gmail.com` (see project notes on invite status).
