# NOAA Weather Time-Series Case Study — Bengaluru (VOBL), 2019–2024

**Data source:** NOAA/NCEI Global Summary of the Day (GSOD)
**Station:** BANGALORE, IN — GSOD id `43295099999` (USAF 432950 / WBAN 99999), VOBL
**Location:** 12.9667° N, 77.5833° E, elevation 921 m
**Period:** 2019-01-01 → 2024-12-31 (2,192 daily records)

---

## 1. Research Problem / Question

**Phenomenon.** Daily near-surface air temperature and precipitation at a single
tropical-upland weather station.

**Primary research question.** *How do temporal dependence, trend, and seasonal
patterns influence daily mean temperature at the Bengaluru NOAA station, and to
what extent can historical temperature observations be used to forecast next-day
temperature using LSTM and Transformer sequence models?*

**Supporting question.** *How does precipitation behaviour differ from
temperature in terms of temporal variability, stationarity, and autocorrelation?*

- **Geographical scope:** Bengaluru (Bangalore) International Airport station, Karnataka, India.
- **Temporal scope:** six continuous calendar years, 2019–2024.
- **Primary variable:** daily mean temperature (`temperature_c`, °C).
- **Secondary variable:** daily total precipitation (`precipitation_mm`, mm).
- **Why time-series analysis is appropriate:** the observations are equally
  spaced in time, exhibit autocorrelation and an annual seasonal cycle, and the
  forecasting objective is inherently sequential.
- **Forecasting objective:** one-step-ahead (t+1) prediction of daily mean
  temperature in °C.

This is **observational** weather data; no causal claims are made. All statistics
below are the actual outputs of `src/run_analysis.py`.

---

## 2. Dataset Understanding

| Property | Value |
|---|---|
| Source | NOAA / NCEI |
| Product | Global Summary of the Day (GSOD) |
| Access URL | https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/ |
| Station | BANGALORE, IN (`43295099999`) |
| Location | 12.9667° N, 77.5833° E, 921 m |
| Frequency | Daily |
| Start / End | 2019-01-01 / 2024-12-31 |
| Observations | 2,192 daily rows (continuous calendar) |
| Raw columns | 28 GSOD fields |

**Key GSOD variables and units (verified against NOAA GSOD documentation):**

| Field | Meaning | Native unit | Sentinel (missing) |
|---|---|---|---|
| `TEMP` | mean daily temperature | °F | 9999.9 |
| `MAX` / `MIN` | daily max / min temperature | °F | 9999.9 |
| `PRCP` | total daily precipitation | inches | 99.99 |
| `PRCP_ATTRIBUTES` | reporting-period flag (A–I) | — | — |

**Unit conversions (documented, not assumed):**
`°C = (°F − 32) × 5/9`  and  `mm = inches × 25.4`.

**Missing values.** After sentinel handling: **22** missing `temperature_c`
values and **991** missing `precipitation_mm` values. The large precipitation gap
is *not* an error: GSOD `PRCP` rows flagged **`I`** mean the station reported no
precipitation observation for the day and the 0.00 value is uncertain — these are
correctly treated as **missing**, not as true zeros, to avoid biasing the dry
season downward.

**Descriptive statistics (temperature, °C):** mean **24.21**, std **2.20**,
min **18.39**, median **23.89**, max **31.61** (n = 2,170 non-missing).
**Precipitation (mm):** mean **6.32**, std **13.75**, median **0.51**,
max **131.57** (n = 1,201 measured days) — a strongly right-skewed,
intermittent variable.

The analysis dataframe (`data/processed/weather_daily.csv`) retains
`date, temperature_c, precipitation_mm` plus `tmax_c`, `tmin_c`, station
metadata, and a small-gap-interpolated `temperature_c_interp` column used for
continuous-series methods (STL, ACF, modelling).

---

## 3. Data Cleaning and Preparation

- Parsed `DATE` to datetime, sorted chronologically, dropped duplicate timestamps.
- Reindexed onto a **continuous daily calendar**, which exposes missing dates as NaN.
- Replaced temperature sentinel `9999.9` and precipitation sentinel `99.99` with NaN.
- Applied `PRCP` flag semantics (`I` → missing; A–H retained as genuine, incl. real 0.0).
- Converted temperature °F → °C and precipitation inches → mm.
- **Temperature:** small gaps (≤ 3 days) filled by *time* interpolation for the
  continuous-series methods; raw column preserved for reporting missingness.
- **Precipitation:** deliberately **not** interpolated — rainfall is intermittent
  and interpolating it would be scientifically misleading; NaNs are kept and
  excluded pairwise.
- No future information is used to fill past values; scalers for forecasting are
  fit on the training partition only (no leakage).

---

## 4. Time-Series Visualization

Figures in `figures/`, each interpreted below.

- **`temperature_timeseries.png`** — Daily temperature oscillates in a clear
  repeating annual band roughly 18–32 °C, with pronounced pre-monsoon peaks
  (Mar–May) and cooler monsoon/winter troughs. No dramatic long-term drift is
  visible to the eye.
- **`precipitation_timeseries.png`** — Precipitation is spiky and intermittent:
  long near-zero stretches (Dec–Mar) punctuated by clustered high-rainfall events
  concentrated in the monsoon/post-monsoon months.
- **`rolling_statistics.png`** — The 30-day rolling mean smooths daily noise into
  a stable seasonal wave; the 30-day rolling std is itself seasonal (higher spread
  in transition months), an early hint that variance is time-varying but bounded.
- **`monthly_temperature.png`** — Monthly means trace the same annual cycle every
  year with a consistent April/May maximum and a monsoon/winter minimum.
- **`monthly_precipitation.png`** — Monthly climatology confirms a wet season
  peaking around **September–October** and a dry **December–February** window.
- **`monthly_temperature_boxplot.png`** — Month-by-month boxplots show the seasonal
  location shift plus wider spread in pre-monsoon months.

---

## 5. Trend and Seasonality Analysis

Three complementary methods were used: rolling statistics, calendar aggregation,
and **STL decomposition** (`period = 365`, robust) on the continuous daily series.

**STL results (actual):**

- **Seasonal strength = 0.789** → a **strong, dominant annual cycle**.
- **Trend strength = 0.161** → a **weak** long-term trend.
- STL trend endpoints: **24.65 °C (2019) → 24.86 °C (2024)**, i.e. a mild net
  warming of ≈ **+0.21 °C** over the six years — real but small relative to the
  seasonal amplitude, so it should not be over-interpreted from six years alone.

**Interpretation.**
- *Trend:* slightly upward but weak; the series is dominated by seasonality, not drift.
- *Seasonality:* strong and regular — highest temperatures in the **March–May**
  pre-monsoon, lowest in the **monsoon-to-winter** window, repeating each year.
- *Precipitation seasonality:* regular at the **monthly-climatology** level (wet
  Sep–Oct, dry Dec–Feb) but **intermittent day-to-day** — seasonal in aggregate,
  bursty in detail — contrasting with temperature's smooth cycle.
- *Residuals:* the STL remainder is small and centred near zero with no obvious
  structure, indicating trend + annual season capture most systematic variation.

See `figures/seasonal_decomposition.png`.

---

## 6. Stationarity Analysis

Two tests with **opposite null hypotheses** were applied to daily temperature,
raw and first-differenced.

- **ADF** — H0: the series **contains a unit root (non-stationary)**. Reject H0
  (p < 0.05) ⇒ stationary.
- **KPSS** — H0: the series **is stationary** around a deterministic component.
  Fail to reject H0 (p ≥ 0.05) ⇒ stationary.

| Series | ADF stat | ADF p | KPSS stat | KPSS p | Conclusion |
|---|---:|---:|---:|---:|---|
| Raw temperature | −4.451 | 0.00024 | 0.128 | 0.10 | **Stationary (both agree)** |
| First difference | −23.500 | 0.0000 | 0.039 | 0.10 | **Stationary (both agree)** |

**Interpretation.** For the raw series, ADF **rejects** the unit-root null and
KPSS **fails to reject** the stationarity null — both point to **stationarity**.
This is physically sensible: Bengaluru's tropical-upland temperature is
**bounded and strongly mean-reverting** within its annual band, so despite the
visible seasonal oscillation the level does not wander like a random walk.
(KPSS `p = 0.10` is the statsmodels upper bound, meaning "≥ 0.10", i.e.
comfortably non-significant.) First differencing keeps the series stationary and
sharply increases the ADF statistic magnitude, confirming no unit root. Because
the raw series is already stationary in level, differencing is available as a
transformation but is **not required** to achieve stationarity — an important
distinction for model input design. ADF and KPSS agree here, so no reconciliation
of conflicting nulls is needed.

---

## 7. Autocorrelation Analysis (ACF / PACF)

- **ACF** measures correlation between the series and its lags, **including**
  indirect propagation through intermediate lags.
- **PACF** measures correlation at a given lag **after** removing the influence of
  all shorter lags.

**Findings (actual):**

- Raw **lag-1 autocorrelation = 0.921** — very strong day-to-day persistence
  (`figures/acf_raw.png`); the ACF decays slowly over many lags.
- Raw **PACF** (`pacf_raw.png`) cuts off sharply after the first one or two lags —
  the classic AR-like signature: today's temperature is largely explained by the
  immediately preceding day(s).
- The long-lag ACF (`acf_longlag.png`) shows **ACF ≈ 0.702 at lag 365**, a clear
  **annual seasonal echo**.
- After **first differencing** (`acf_stationary.png`, `pacf_stationary.png`),
  the lag-1 autocorrelation drops to **−0.148** and the slow decay disappears,
  confirming the differenced series is close to short-memory noise.

**Implication for forecasting.** The strong lag-1 persistence and fast PACF
cut-off explain why a **persistence baseline is a very strong competitor** and
motivate a modest input window: a **30-day sequence length** comfortably covers
the short-memory structure the PACF reveals, while remaining CPU-cheap. Seasonal
information at lag 365 exceeds the six-year window's practical modelling range and
is instead captured implicitly by the seasonal band the models see.

---

## 8. Initial Findings and Research Hypothesis

**Findings (evidence-backed):**

1. Daily temperature is dominated by a **strong annual seasonal cycle** (STL
   seasonal strength 0.79) with a weak upward trend (≈ +0.21 °C over 2019–2024).
2. Temperature is **stationary in level** by both ADF (p = 0.00024) and
   KPSS (p = 0.10) — bounded and mean-reverting, not a random walk.
3. Temperature shows **very high short-term persistence** (lag-1 ACF = 0.921) with
   an AR-like PACF cut-off after 1–2 lags.
4. Precipitation is **intermittent and right-skewed** (median 0.51 mm, max
   131.6 mm), seasonal in monthly aggregate but bursty daily — behaviourally very
   different from the smooth temperature cycle.
5. A clear **annual echo** appears at lag 365 (ACF = 0.702).

**Hypotheses.**

- **H1:** Daily temperature exhibits statistically meaningful temporal persistence
  and seasonal structure, so sequence models using recent history should easily beat
  a mean forecast and *may* approach or beat a one-step persistence baseline.
- **H0 (null):** Historical sequential structure offers no material improvement over
  a naive persistence baseline for short-horizon (t+1) temperature forecasting.

**Result vs. hypothesis (see §9–10).** The evidence **fails to reject H0**: with a
lag-1 autocorrelation of 0.92, persistence is extremely hard to beat at the one-day
horizon, and the small LSTM/Transformer did not improve on it. This is a truthful,
expected outcome for strongly-persistent daily temperature — reported as-is.

---

## 9. Forecasting Methodology

- **Target:** next-day mean temperature (°C), one-step (t+1).
- **Split:** chronological 70 / 15 / 15 (no shuffling) →
  **1,504** train / **329** val / **329** test sequences.
- **Sequence length:** 30 days (justified by ACF/PACF above).
- **Scaling:** `StandardScaler` fit on **training data only**, applied to all
  partitions — no leakage.
- **Reproducibility:** seed = 42 for Python, NumPy, and PyTorch.

## 10. Baseline, LSTM, Transformer, Comparison

| Model | MAE (°C) | RMSE (°C) | Train time (s) |
|---|---:|---:|---:|
| **Persistence (naive)** | **0.674** | **0.888** | 0.0 |
| LSTM (hidden 32, 1 layer) | 0.766 | 0.972 | 5.5 |
| Transformer (d_model 32, 4 heads, 2 layers) | 0.736 | 0.944 | 9.9 |

- The **persistence baseline is the best model** on both MAE and RMSE.
- The **Transformer edges out the LSTM**, but both trail persistence.
- Cause: lag-1 autocorrelation of 0.92 means yesterday's value is already a
  near-optimal one-step predictor; small neural models trained for ≤ 15 epochs on
  ~1,500 sequences cannot extract enough extra signal to overcome that at t+1.
- Differences among neural models are small and are **not** claimed as
  statistically significant.

Figures: `lstm_forecast.png`, `transformer_forecast.png`, `model_comparison.png`.

---

## 11. Limitations

- Six years is short for confident long-term climate-trend claims.
- Single station — results are local to Bengaluru, not generalisable.
- GSOD daily precipitation has substantial documented missingness (`I` flag).
- Models are intentionally tiny and untuned (CPU/battery constraints); a longer
  horizon or exogenous inputs (humidity, rainfall) could change the LSTM/Transformer
  vs. persistence comparison.
- t+1 horizon strongly favours persistence; multi-step forecasting was out of scope.

## 12. Conclusion

Bengaluru daily temperature (2019–2024) is a **stationary, strongly seasonal,
highly persistent** series with a weak warming trend, while precipitation is
intermittent and seasonal only in aggregate. Statistical evidence (STL, ADF/KPSS,
ACF/PACF) is internally consistent. For **one-step** forecasting the naive
persistence baseline is the strongest model; the small LSTM and Transformer are
competitive but do not beat it — an honest, well-understood result given the
0.92 lag-1 autocorrelation.

*All numbers above are reproduced by `python src/run_analysis.py` (see `results.json`).*
