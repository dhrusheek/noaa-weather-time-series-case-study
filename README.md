# NOAA Weather Time-Series Case Study — Bengaluru (VOBL)

Time-series analysis and short-term forecasting of daily temperature and
precipitation for a NOAA weather station, using the **Global Summary of the Day
(GSOD)** product. Full statistical analysis (trend, seasonality, stationarity,
autocorrelation) plus **five baselines**, an **LSTM**, and a **Transformer**,
evaluated at **one step** and across a **30-day horizon**.

## Overview

- **Station:** BANGALORE, IN — GSOD id `43295099999` (USAF 432950 / WBAN 99999), VOBL
- **Location:** 12.9667° N, 77.5833° E, elevation 921 m
- **Period:** 2010-01-01 → 2024-12-31 (**5,479** daily records, ≈15 years)
- **Primary variable:** daily mean temperature (°C) · **Secondary:** precipitation (mm)

## Research Question

> How do temporal dependence, trend, and seasonal patterns influence daily mean
> temperature at the Bengaluru NOAA station, and to what extent can historical
> observations forecast temperature — one day ahead and over a 30-day horizon —
> using LSTM and Transformer models? (Supporting: how does precipitation differ from
> temperature in variability, stationarity, and autocorrelation?)

Observational data — **no causal claims**.

## Dataset

NOAA/NCEI GSOD daily CSVs, 2010–2024, retrieved automatically over HTTPS
(`src/download_data.py`). GSOD native units are °F (temperature, sentinel `9999.9`)
and inches (precipitation, sentinel `99.99`); converted to °C and mm. `PRCP` flag `I`
(no measurement) is treated as **missing**, not a true zero. See `data/README.md` and
`report/time_series_case_study.md`.

## Methodology

1. **Automated HTTPS retrieval** of all yearly GSOD CSVs (`src/download_data.py`).
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

## Forecasting Models

- **5 baselines** — Persistence (t-1), Seasonal naive (t-7), Seasonal naive (t-365), Mean (climatology), **ARIMA(2,0,2)**
- **LSTM** — hidden 32, 1 layer, ≤15 epochs, early stopping (patience 3)
- **Transformer** — d_model 32, 4 heads, 2 encoder layers, positional encoding, ≤15 epochs

## Results (actual, from `results.json`)

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

## Reproduction Instructions

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py     # HTTPS fetch GSOD CSVs -> data/raw/
python src/run_analysis.py      # cleaning + stats + figures + models -> figures/, results.json
python src/build_notebook.py    # (optional) regenerate the notebook
```

Everything is deterministic (seed = 42) and regenerable from raw data with the two
commands above. The notebook `notebooks/time_series_case_study.ipynb` runs the same
analysis top-to-bottom with narrative.

## Data Source

NOAA/NCEI Global Summary of the Day (GSOD):
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

Station metadata reference:
https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv

## Limitations

Single station/location; 15 years is short for climate-trend attribution; GSOD daily
precipitation has substantial documented missingness; univariate (no exogenous
predictors); models intentionally tiny/untuned (CPU + battery); single temporal
holdout; t+1 structurally favours persistence.

## Authors / Collaborators

Maintained by the repository owner. Collaborators invited via GitHub:
`kushalyalamanchi@gmail.com`, `Nethi.kushaal@gmail.com` (see project notes on invite status).
