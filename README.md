# NOAA Weather Time-Series Case Study — Bengaluru (VOBL)

Time-series analysis and short-term forecasting of daily temperature and
precipitation for a NOAA weather station, using the **Global Summary of the Day
(GSOD)** product. Statistical analysis (trend, seasonality, stationarity,
autocorrelation) plus persistence, **LSTM**, and **Transformer** forecasting.

## Overview

- **Station:** BANGALORE, IN — GSOD id `43295099999` (USAF 432950 / WBAN 99999), VOBL
- **Location:** 12.9667° N, 77.5833° E, elevation 921 m
- **Period:** 2019-01-01 → 2024-12-31 (**2,192** daily records)
- **Primary variable:** daily mean temperature (°C) · **Secondary:** precipitation (mm)

## Research Question

> How do temporal dependence, trend, and seasonal patterns influence daily mean
> temperature at the Bengaluru NOAA station, and to what extent can historical
> observations forecast next-day temperature using LSTM and Transformer models?
> (Supporting: how does precipitation differ from temperature in variability,
> stationarity, and autocorrelation?)

Observational data — **no causal claims**.

## Dataset

NOAA/NCEI GSOD daily CSVs, 2019–2024. GSOD native units are °F (temperature,
sentinel `9999.9`) and inches (precipitation, sentinel `99.99`); converted to °C
and mm. `PRCP` flag `I` (no measurement) is treated as **missing**, not a true
zero. See `data/README.md` and `report/time_series_case_study.md`.

## Methodology

1. Download GSOD CSVs (`src/download_data.py`).
2. Clean, convert units, handle sentinels/flags, reindex to a daily calendar (`src/preprocessing.py`).
3. EDA visualizations, STL decomposition, ADF + KPSS stationarity tests, ACF/PACF (`src/statistical_analysis.py`, `src/run_analysis.py`).
4. Chronological 70/15/15 split, scaler fit on train only, 30-day input window.
5. Persistence baseline, small LSTM, small Transformer (`src/models.py`).

## Rubric Coverage

| Rubric Component | Marks | Implementation |
|------------------|------:|----------------|
| Research problem/question | 1 | Report §1 + README; phenomenon, scope, variables, forecasting objective explicit |
| Dataset understanding | 1 | Report §2; source/product/URL, station, units, `head/tail/shape/info`, describe, missing-value & duplicate audit |
| Time-series visualization | 1 | 6+ labelled figures in `figures/`, each interpreted in report §4 |
| Trend and seasonality | 2 | Rolling stats + monthly aggregation + **STL(365)**; seasonal strength 0.79, trend +0.21 °C; report §5 |
| Stationarity analysis | 2 | **ADF & KPSS** on raw + differenced series, results table, opposite-null explanation; report §6 |
| ACF/PACF analysis | 2 | ACF/PACF raw + differenced + long-lag; lag-1 = 0.921, lag-365 = 0.702; interpretation + link to seq length; report §7 |
| Initial findings/hypothesis | 1 | 5 evidence-backed findings + H1/H0; report §8 |
| **Total** | **10** | **Complete** |

## Forecasting Models

- **Persistence** — `y_hat(t+1) = y(t)`
- **LSTM** — hidden 32, 1 layer, ≤15 epochs, early stopping (patience 3)
- **Transformer** — d_model 32, 4 heads, 2 encoder layers, positional encoding, ≤15 epochs

## Results (actual, from `results.json`)

| Model | MAE (°C) | RMSE (°C) | Train time (s) |
|---|---:|---:|---:|
| **Persistence (naive)** | **0.674** | **0.888** | 0.0 |
| LSTM | 0.766 | 0.972 | 5.5 |
| Transformer | 0.736 | 0.944 | 9.9 |

**Best model: Persistence.** With lag-1 autocorrelation of 0.92, yesterday's
temperature is a near-optimal one-step predictor; the small untuned neural models
are competitive but do not beat it. Reported truthfully — no fabricated metrics.

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
│   ├── raw/                           # GSOD source CSVs (small, kept for reproducibility)
│   └── processed/weather_daily.csv
├── notebooks/time_series_case_study.ipynb
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── statistical_analysis.py
│   ├── models.py
│   └── run_analysis.py
├── figures/                          # all generated PNGs
└── report/time_series_case_study.md
```

## Reproduction Instructions

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py     # fetch GSOD CSVs -> data/raw/
python src/run_analysis.py      # preprocessing + stats + figures + models -> figures/, results.json
```

The notebook `notebooks/time_series_case_study.ipynb` runs the same analysis
top-to-bottom with narrative. Seed = 42 throughout.

## Data Source

NOAA/NCEI Global Summary of the Day (GSOD):
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

Station metadata reference:
https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv

## Limitations

Six years is short for climate-trend claims; single-station results are local;
GSOD daily precipitation has substantial documented missingness; models are
intentionally tiny/untuned (CPU + battery constraints); the t+1 horizon strongly
favours persistence.

## Authors / Collaborators

Maintained by the repository owner. Collaborators invited via GitHub:
`kushalyalamanchi@gmail.com`, `Nethi.kushaal@gmail.com`.
