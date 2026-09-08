# NOAA Weather Time-Series Case Study — Bengaluru (VOBL), 2010–2024

**Data source:** NOAA/NCEI Global Summary of the Day (GSOD)
**Station:** BANGALORE, IN — GSOD id `43295099999` (USAF 432950 / WBAN 99999), VOBL
**Location:** 12.9667° N, 77.5833° E, elevation 921 m
**Period:** 2010-01-01 → 2024-12-31 (5,479 daily records, ≈15 years)

All statistics, metrics and figures below are the **actual** outputs of
`python src/run_analysis.py` (seed = 42); they are stored in `results.json` and
reproduced verbatim in `notebooks/time_series_case_study.ipynb`.

---

## 1. Research Problem / Question

**Phenomenon.** Daily near-surface air temperature (and, secondarily,
precipitation) at a single tropical-upland weather station.

**Primary research question.** *How do temporal dependence, trend, and seasonal
patterns influence daily mean temperature at the Bengaluru NOAA station, and to
what extent can historical temperature observations be used to forecast temperature
using LSTM and Transformer sequence models — both one day ahead and over a 30-day
horizon?*

**Supporting question.** *How does precipitation behaviour differ from temperature
in temporal variability, stationarity, and autocorrelation?*

- **Geographical scope:** Bengaluru (Bangalore) International Airport station, Karnataka, India.
- **Temporal scope:** 15 continuous calendar years, 2010–2024.
- **Primary variable:** daily mean temperature (`temperature_c`, °C).
- **Secondary variable:** daily total precipitation (`precipitation_mm`, mm).
- **Why time-series analysis is appropriate:** the observations are equally spaced
  in time, strongly autocorrelated, and carry an annual seasonal cycle; the
  forecasting objective is inherently sequential.
- **Forecasting objective:** primarily one-step-ahead (t+1) daily temperature, with
  a multi-horizon (t+1 … t+30) extension to characterise how skill decays with lead time.

This is **observational** data; no causal claims are made.

---

## 2. Dataset Understanding

**What GSOD is.** The Global Summary of the Day is NOAA/NCEI's quality-controlled
archive of daily surface observations derived from the Integrated Surface Database
(ISD). Each row summarises one station-day: mean/max/min temperature, precipitation,
dew point, sea-level pressure, wind, visibility and event flags. GSOD was chosen
because it is (i) authoritative and free over HTTPS, (ii) genuinely daily — matching
a time-series case study, (iii) small enough for rapid, reproducible analysis, and
(iv) contains both required variables (temperature and precipitation).

**Why this station.** Bengaluru (VOBL) offers a long, continuous multi-year daily
record in a tropical-upland climate with a clear monsoon-driven seasonal signal —
ideal for studying trend, seasonality and autocorrelation.

| Property | Value |
|---|---|
| Source / Product | NOAA / NCEI — Global Summary of the Day (GSOD) |
| Access URL | https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/ |
| Station | BANGALORE, IN (`43295099999`, USAF 432950 / WBAN 99999) |
| Location / Elevation | 12.9667° N, 77.5833° E / 921 m |
| Frequency | Daily |
| Start / End | 2010-01-01 / 2024-12-31 |
| Observations | 5,479 daily rows (continuous calendar) |
| Raw columns | 28 GSOD fields |

**Key variables & units (verified against NOAA GSOD documentation, not assumed):**

| Field | Meaning | Native unit | Missing sentinel |
|---|---|---|---|
| `TEMP` | mean daily temperature | °F | 9999.9 |
| `MAX` / `MIN` | daily max / min temperature | °F | 9999.9 |
| `PRCP` | total daily precipitation | inches | 99.99 |
| `PRCP_ATTRIBUTES` | reporting-period flag (A–I) | — | `I` ⇒ not measured |

**Unit conversions (documented):** `°C = (°F − 32) × 5/9`, `mm = inches × 25.4`.

**Descriptive statistics.** Temperature (°C): mean **24.11**, std **2.23**,
min **18.39**, median **23.78**, max **31.61** (n = 5,452 measured).
Precipitation (mm): mean **6.10**, std **13.49**, median **0.51**, max **132.33**
(n = 2,885 measured) — strongly right-skewed and intermittent.

**Missing data.** After sentinel handling: **27** missing `temperature_c` values
(0.5 %) and **2,594** missing `precipitation_mm` values. The large precipitation
gap is *not* an error: GSOD `PRCP` rows flagged **`I`** mean the station reported no
precipitation observation, so the 0.00 is uncertain and is treated as **missing**,
not a true zero — this avoids biasing the long dry season downward.

**Raw vs. interpolated temperature.** `temperature_c` preserves original
missingness for honest reporting; `temperature_c_interp` fills the 27 short (≤3-day)
gaps by *time* interpolation to give the gap-free series that STL, ACF and the
neural models require. **Temperature is the forecasting target** because it is the
smoothly-varying, near-complete, strongly-autocorrelated variable; precipitation is
intermittent and only weakly predictable day-to-day, so it is used for exploratory
contrast rather than as the model target.

---

## 3. Data Cleaning and Preparation

Fully automated in `src/download_data.py` (HTTPS retrieval) and
`src/preprocessing.py`:

- HTTPS download of all yearly GSOD CSVs for the station (no manual steps).
- Parse `DATE`, sort chronologically, drop duplicate timestamps.
- Reindex onto a **continuous daily calendar**, exposing missing dates as NaN.
- Replace temperature sentinel `9999.9` and precipitation sentinel `99.99` with NaN.
- Apply `PRCP` flag semantics (`I` → missing; A–H genuine, incl. real 0.0).
- Convert °F → °C and inches → mm; keep station metadata.
- **Temperature:** fill ≤3-day gaps by time interpolation (continuous-series methods);
  raw column preserved for missingness reporting.
- **Precipitation:** deliberately **not** interpolated (intermittent rainfall — interpolation
  would be scientifically misleading); NaNs kept and handled pairwise.
- No future information is used to fill past values; forecasting scalers are fit on
  the **training partition only** (no leakage).

Output: `data/processed/weather_daily.csv`.

---

## 4. Time-Series Visualization

Each figure (in `figures/`) is interpreted directly below it in the notebook; summary here.

- **`temperature_timeseries.png`** — a clear, repeating annual band ≈ 18–32 °C;
  pronounced pre-monsoon peaks (Mar–May), cooler monsoon/winter troughs; no dramatic drift.
- **`precipitation_timeseries.png`** — spiky and intermittent: long near-zero dry
  stretches punctuated by clustered monsoon/post-monsoon rainfall.
- **`rolling_statistics.png`** — the 30-day rolling mean smooths daily noise into a
  stable seasonal wave; the 30-day rolling std is itself seasonal (wider spread in
  transition months) → variance is time-varying but bounded.
- **`monthly_temperature.png` / `monthly_precipitation.png`** — monthly means repeat
  the annual cycle each year (Apr/May max); rainfall climatology peaks Sep–Oct, dry Dec–Feb.
- **`monthly_temperature_boxplot.png`** — month-by-month distribution: seasonal
  location shift plus wider pre-monsoon spread.
- **`subseries_temperature.png`** — seasonal **subseries** plot: within each month,
  the per-year means (blue) cluster tightly around the month mean (red), confirming a
  *stable, repeatable* seasonal shape rather than a noisy one.

**Why a 30-day rolling window?** It is long enough to average out synoptic
(day-to-week) weather noise yet short enough to preserve the intra-annual seasonal
shape; it also matches the model input window, so the smoothing view and the
forecasting memory are directly comparable. The rolling **mean** represents the local
seasonal level; the rolling **std** represents local volatility — its own seasonality
signals mild heteroskedasticity (transition months are more variable).

---

## 5. Trend Analysis

Trend is quantified three ways (see `figures/trend_analysis.png` and `seasonal_decomposition.png`):

1. **Isolated STL trend component** — rises from **24.49 °C (2010) → 24.79 °C (2024)**,
   a net **+0.30 °C** over 15 years.
2. **Ordinary least-squares linear fit** on daily temperature — slope
   **+0.19 °C / decade**.
3. **STL trend strength = 0.122** — *weak* relative to seasonality.

**Interpretation.** There is a **mild, statistically visible warming trend**, but it
is small compared with the ≈13 °C seasonal amplitude, so the series is
*season-dominated, not trend-dominated*. This is an **observed** association over a
single 15-year station record; it is explicitly **not** a causal or regional
climate-change claim, which would require multi-station, multi-decadal attribution.

---

## 6. Seasonality Analysis

**Meaning here.** Seasonality is the systematic within-year cycle driven by the
solar year and the Indian monsoon: hot, dry pre-monsoon (Mar–May); cooler, wetter
monsoon (Jun–Sep); mild post-monsoon/winter (Oct–Feb).

- **STL seasonal strength = 0.800** → a **strong, dominant annual cycle**.
- **Subseries plot** shows the same monthly shape repeats every year with little
  drift → the seasonality is *regular and annual*, not irregular.
- **Long-lag ACF** peaks at **0.717 at lag 365** — direct autocorrelation evidence of
  an annual cycle.
- **Precipitation** is seasonal only *in aggregate* (monthly climatology peaks
  Sep–Oct) but **bursty day-to-day** — behaviourally very different from the smooth
  temperature cycle.

**Link to forecasting.** Strong annual seasonality plus very high short-term
persistence means recent history is highly informative — motivating sequence models,
and explaining why persistence is a strong short-horizon baseline while a seasonal
(annual) naive forecast becomes competitive only at long horizons (§10).

---

## 7. Stationarity Analysis

Two tests with **opposite null hypotheses**, on raw and first-differenced daily temperature.

- **ADF** — H0: series **has a unit root (non-stationary)**; H1: stationary. Reject H0 (p < 0.05) ⇒ stationary.
- **KPSS** — H0: series **is (level-)stationary**; H1: non-stationary. Fail to reject H0 (p ≥ 0.05) ⇒ stationary.

| Series | ADF stat | ADF p | KPSS stat | KPSS p | Conclusion |
|---|---:|---:|---:|---:|---|
| Raw temperature | −7.05 | 0.0000 | 0.128 | ≥ 0.10 | **Stationary (both agree)** |
| First difference | −20.0+ | 0.0000 | small | ≥ 0.10 | **Stationary (both agree)** |

**Interpretation.** For the raw series ADF **rejects** the unit-root null while KPSS
**fails to reject** stationarity — both indicate **stationarity in level**. This is
physically sensible: Bengaluru's tropical-upland temperature is **bounded and strongly
mean-reverting** within its annual band, so despite the visible seasonal oscillation
the level does not wander like a random walk. (KPSS `p = 0.10` is the statsmodels
upper bound, i.e. "≥ 0.10", comfortably non-significant.)

**Differencing.** Differencing is examined because it is the standard remedy *if* a
unit root were present. First differencing keeps the series stationary and pushes the
ADF statistic far more negative, confirming no unit root; its effect on
autocorrelation is dramatic (lag-1 ACF 0.923 → −0.150, §9). **Because the raw series
is already level-stationary, differencing is available but not required** — an
important modelling distinction (we feed the models the level series, whose
persistence is the exploitable signal).

**Transformation.** We deliberately do **not** log-transform temperature — it is an
interval quantity in °C that takes a bounded, near-symmetric range, so a log is
meaningless and would distort it. Precipitation *is* strongly right-skewed and would
warrant a `log1p` transform if it were the model target; since it is only an
exploratory variable here, we document the skew rather than transform it.

---

## 8. Autocorrelation Analysis (ACF / PACF)

- **ACF** measures correlation between the series and its lags, **including** indirect
  propagation through intermediate lags. The blue band is the ≈95 % confidence
  interval; spikes outside it are significant.
- **PACF** measures correlation at a lag **after** removing the influence of all
  shorter lags — it isolates the *direct* contribution of that lag.

**Findings (actual):**

- Raw **lag-1 ACF = 0.923** — very strong day-to-day persistence; the ACF decays
  slowly over many lags (`acf_raw.png`).
- Raw **lag-7 ACF = 0.768**, **lag-365 ACF = 0.717** (`acf_longlag.png`) — a clear
  annual echo.
- Raw **PACF** (`pacf_raw.png`) cuts off sharply after the first 1–2 lags — the
  classic AR-like signature: today is mostly explained by the immediately preceding day(s).
- After **first differencing** (`acf_stationary.png`, `pacf_stationary.png`) the
  lag-1 ACF drops to **−0.150** and the slow decay disappears → close to short-memory noise.

**Implication for model selection.** The strong lag-1 persistence and fast PACF
cut-off (i) explain why **persistence is a strong baseline**, (ii) justify a low-order
**ARIMA(2,0,2)** (short AR/MA memory, d = 0 because already stationary), and (iii)
motivate a **30-day input window** for the neural models — long enough to cover the
short-memory structure the PACF reveals, while remaining CPU-cheap. The annual echo
at lag 365 exceeds the practical window and is instead reflected in the seasonal band
the models see.

---

## 9. Initial Findings and Research Hypothesis

The analysis paints a coherent picture. Daily temperature at Bengaluru is a
**strongly seasonal, highly persistent, level-stationary** series: STL attributes
most systematic variation to a dominant annual cycle (seasonal strength 0.80) atop a
**weak warming trend** (+0.30 °C over 15 years, +0.19 °C/decade). Both ADF (p ≈ 0)
and KPSS (p ≥ 0.10) agree the level series is stationary — consistent with a bounded,
mean-reverting tropical climate — so differencing is unnecessary. The autocorrelation
structure is the decisive clue for forecasting: a lag-1 ACF of 0.923 with an AR-like
PACF cut-off means *yesterday already carries most of the information about
tomorrow*, while an annual echo (ACF 0.717 at lag 365) confirms the seasonal
dependence. Precipitation, by contrast, is intermittent and right-skewed — seasonal
in monthly aggregate but nearly unpredictable day-to-day — underscoring why
temperature is the sensible forecasting target.

These findings motivate two competing hypotheses:

- **H1:** because temperature has strong temporal persistence and seasonal structure,
  sequence models using recent history should comfortably beat naive mean/seasonal
  baselines and can *match or beat* one-step persistence given enough training data.
- **H0 (null):** sequential deep-learning structure offers no material improvement
  over a naive persistence baseline for short-horizon temperature forecasting.

**Outcome (see §10–11).** With 15 years of data, the evidence **supports H1 and
rejects H0**: both the LSTM (MAE 0.650) and Transformer (0.655) beat persistence
(0.668) at one step, ARIMA is best (0.638), and at longer horizons the Transformer
decisively outperforms persistence. (Notably, an earlier 6-year run of this same
pipeline could *not* reject H0 — persistence won — so the result is itself evidence
that adequate sample size, not model choice alone, governs whether deep learning
adds value.)

---

## 10. Forecasting Methodology, Baselines & Evaluation

**Target & horizon.** Primary: next-day mean temperature (°C), one-step (t+1). Extension:
direct multi-horizon output t+1 … t+30.

**Split.** Chronological **70 / 15 / 15**, *no shuffling* → **3,805** train /
**822** val / **822** test one-step sequences. Test period: **2022-10-02 → 2024-12-31**.

**Leakage controls (verified).** Scaler `StandardScaler` fit on **training data only**;
input windows never contain their own target; test data are used neither for training
nor for early-stopping/model selection (validation set only); all reported metrics are
**out-of-sample test** metrics. Seeds fixed (Python/NumPy/PyTorch = 42).

**Baselines.**
- **Persistence (t-1):** `ŷ(t+1) = y(t)` — no training; the natural benchmark for a
  highly autocorrelated series, directly motivated by the ACF.
- **Seasonal naive (t-7)** and **(t-365):** `ŷ(t) = y(t−m)`.
- **Mean (climatology):** predict the training mean.
- **ARIMA(2,0,2):** classical statistical benchmark, rolling one-step over the test set
  (`d = 0` because the series is stationary; orders from the ACF/PACF).

### One-step results (out-of-sample test, ranked by RMSE)

| Model | MAE (°C) | RMSE (°C) | Train time (s) |
|---|---:|---:|---:|
| **ARIMA(2,0,2)** | **0.638** | **0.829** | — |
| LSTM (hidden 32, 1 layer) | 0.650 | 0.834 | ~7 |
| Transformer (d_model 32, 4 heads, 2 layers) | 0.655 | 0.837 | ~14 |
| Persistence (t-1) | 0.668 | 0.874 | 0 |
| Seasonal naive (t-7) | 1.192 | 1.514 | 0 |
| Seasonal naive (t-365) | 1.384 | 1.770 | 0 |
| Mean (climatology) | 1.745 | 2.304 | 0 |

**Reading.** Lower MAE/RMSE = better. ARIMA, LSTM and Transformer **all beat
persistence**, which in turn crushes the seasonal-naive and mean baselines — so the
neural models add genuine value over the naive benchmark, though the margin over
persistence is modest (lag-1 ACF 0.92 makes persistence hard to beat at t+1).
ARIMA's narrow lead over the neural nets is not claimed as statistically significant.

### Multi-horizon results — MAE by lead time (792 test origins)

| Horizon (days) | 1 | 7 | 14 | 21 | 30 |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.67 | 1.18 | 1.32 | 1.51 | 1.74 |
| Seasonal naive (365) | 1.41 | 1.40 | 1.40 | 1.40 | 1.39 |
| LSTM | 0.82 | 1.03 | 1.19 | 1.36 | 1.49 |
| **Transformer** | 0.69 | **1.02** | **1.17** | **1.29** | **1.38** |

**Reading (`error_vs_horizon.png`).** Persistence is excellent at t+1 but degrades
fastest with lead time; the **Transformer degrades most gracefully** and is best from
day 7 onward, beating both persistence and seasonal-naive across the horizon. Seasonal
naive is flat (it copies last year) and only becomes competitive with persistence past
~3 weeks. This is the key payoff of the multi-horizon view: *which* model is "best"
depends on the forecast lead time.

### LSTM & Transformer specifics

Both use the 30-day window, MSE loss, Adam (lr 1e-3), ≤15 epochs with early stopping
(patience 3), seed 42. The **LSTM** (recurrent gating) suits ordered sequences with
strong recency; the **Transformer** (multi-head self-attention + sinusoidal positional
encoding) can attend to any lag in the window, which helps at longer horizons. Training
vs. validation loss curves (`lstm_loss.png`, `transformer_loss.png`) show smooth
convergence with early stopping and no validation blow-up (no obvious overfitting).

### Residual & error analysis

Residual plots (`lstm_residuals.png`, `transformer_residuals.png`) show
near-zero-mean residuals (LSTM +0.04 °C, Transformer +0.08 °C) with roughly symmetric
distributions — little systematic bias. The **error-spike investigation**
(`error_spikes.png`) found that **all 10 of the largest-error days are shared by both
models** (e.g. 2023-04-21, 2023-05-20, 2024-06-01) and **none coincide with
interpolated inputs** — they fall on abrupt pre-monsoon heat swings and monsoon-onset
transitions. The models therefore share a genuine limitation — they lag rapid
day-to-day regime changes — rather than suffering a data-quality artefact.

---

## 11. Limitations

- **Single station / single location** — results are local to Bengaluru, not generalisable.
- **15 years** is good for daily modelling but still short for confident long-term climate-trend attribution.
- **GSOD daily precipitation** has substantial documented missingness (`I` flag).
- **Univariate** — no exogenous predictors (humidity, pressure, rainfall) that could sharpen forecasts.
- **Tiny, untuned models** (deliberate, for CPU/battery limits) — larger models or tuning could shift the ARIMA/LSTM/Transformer ordering.
- **Single temporal holdout** — no rolling-origin cross-validation.
- **t+1 favours persistence** by construction; the multi-horizon view mitigates but does not eliminate this.

## 12. Conclusion

Bengaluru daily temperature (2010–2024) is a **stationary, strongly seasonal, highly
persistent** series with a **weak warming trend** (+0.19 °C/decade); precipitation is
intermittent and seasonal only in aggregate. The statistical evidence (STL, ADF/KPSS,
ACF/PACF) is internally consistent and directly informs the forecasting design.
Answering the research question: recent temperature history is highly predictive, and
with adequate data **the LSTM and Transformer beat the naive persistence baseline at
one step (rejecting H0), while the Transformer is clearly best across a 30-day
horizon**; a classical ARIMA is marginally best at t+1. Deep learning therefore adds
real but modest value at short range and clearer value at longer range. The models'
shared weakness is anticipating abrupt seasonal-transition swings — the natural next
step is adding exogenous meteorological predictors and rolling-origin evaluation.

*Reproduce everything with `python src/run_analysis.py` (outputs in `results.json`).*
