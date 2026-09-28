# NOAA Weather Forecasting Across Major Indian Cities

## Scope and method

Daily NOAA GSOD data (2010–2024) were analyzed for the eight configured airport/observatory stations. Each city uses the same cleaning rules, 70/15/15 chronological split, 30-day lookback, five one-step baselines (including rolling ARIMA), LSTM, Transformer, and direct 1–30 day LSTM/Transformer forecasts. All scores below are from each station's held-out test period.

Absolute MAE/RMSE compare errors in °C. Skill vs persistence is `100 × (1 − model MAE / persistence MAE)`; positive values improve on simply predicting yesterday's temperature. City ranking by raw error is influenced by each location's temperature variability, so read it alongside relative skill.

## Stations and data coverage

| City | GSOD station | Observations | Date range | Test period | Missing daily mean temperature |
|---|---:|---:|---|---|---:|
| Bengaluru | `43295099999` (BANGALORE, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 27 |
| Mumbai | `43003099999` (CHHATRAPATI SHIVAJI INTERNATIONAL, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 13 |
| Chennai | `43279099999` (CHENNAI INTERNATIONAL, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 12 |
| Pune | `43063099999` (PUNE, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 25 |
| Ahmedabad | `42647099999` (AHMEDABAD, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 66 |
| Hyderabad | `43128099999` (BEGUMPET AIRPORT, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 15 |
| Kolkata | `42809099999` (NETAJI SUBHASH CHANDRA BOSE INTERNATIONAL, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 20 |
| Delhi | `42182099999` (SAFDARJUNG, IN) | 5,479 | 2010-01-01–2024-12-31 | 2022-10-02–2024-12-31 | 37 |

## One-day-ahead results

Lower MAE/RMSE is better. The best model is ranked by RMSE across the five baselines and two neural models.

| City | Persistence MAE | LSTM MAE | Transformer MAE | Best model | Best MAE | Best RMSE | Best-model skill vs persistence |
|---|---:|---:|---:|---|---:|---:|---:|
| Bengaluru | 0.668 | 0.650 | 0.655 | ARIMA(2, 0, 2) | 0.638 | 0.829 | +4.4% |
| Mumbai | 0.637 | 0.617 | 0.637 | LSTM | 0.617 | 0.858 | +3.2% |
| Chennai | 0.649 | 0.630 | 0.631 | Transformer | 0.631 | 0.858 | +2.7% |
| Pune | 0.748 | 0.741 | 0.743 | Transformer | 0.743 | 0.976 | +0.6% |
| Ahmedabad | 0.778 | 0.792 | 0.765 | Transformer | 0.765 | 1.044 | +1.7% |
| Hyderabad | 0.818 | 0.826 | 0.818 | Transformer | 0.818 | 1.090 | +0.0% |
| Kolkata | 0.922 | 0.952 | 0.956 | ARIMA(2, 0, 2) | 0.913 | 1.221 | +1.1% |
| Delhi | 1.051 | 1.102 | 1.044 | Transformer | 1.044 | 1.367 | +0.7% |

## Transformer forecast errors by lead time

| City | Day 1 | Day 7 | Day 14 | Day 21 | Day 30 |
|---|---:|---:|---:|---:|---:|
| Bengaluru | 0.688 | 1.019 | 1.169 | 1.290 | 1.376 |
| Mumbai | 0.761 | 1.182 | 1.315 | 1.406 | 1.542 |
| Chennai | 0.689 | 1.057 | 1.079 | 1.075 | 1.197 |
| Pune | 0.910 | 1.354 | 1.535 | 1.591 | 1.754 |
| Ahmedabad | 0.912 | 1.517 | 1.701 | 1.912 | 2.235 |
| Hyderabad | 0.953 | 1.481 | 1.650 | 1.748 | 2.000 |
| Kolkata | 1.185 | 1.716 | 1.810 | 1.907 | 1.985 |
| Delhi | 1.184 | 1.900 | 2.104 | 2.123 | 2.333 |

## Time-series characteristics

| City | Seasonal strength | Linear trend (°C/decade) | Lag-1 ACF | Lag-365 ACF |
|---|---:|---:|---:|---:|
| Bengaluru | 0.800 | +0.187 | 0.923 | 0.717 |
| Mumbai | 0.636 | +0.581 | 0.902 | 0.577 |
| Chennai | 0.836 | +0.772 | 0.940 | 0.791 |
| Pune | 0.851 | +0.186 | 0.954 | 0.802 |
| Ahmedabad | 0.907 | -0.245 | 0.973 | 0.872 |
| Hyderabad | 0.806 | -0.398 | 0.940 | 0.743 |
| Kolkata | 0.883 | +0.453 | 0.962 | 0.846 |
| Delhi | 0.933 | +0.056 | 0.979 | 0.909 |

## Interpretation limits

Each location is represented by one airport/observatory station. Station microclimate and measurement coverage differ, even though the analysis settings and 2010–2024 calendar window are held consistent. Skill vs persistence is a useful relative measure, but it does not remove all station-to-station differences. This is a single chronological holdout, not rolling-origin validation; precipitation is analyzed but is not a forecast input. Full stationarity tests, ACF/PACF values, figures, residual analyses, and all model metrics are in `results/cities/<city>.json` and `figures/cities/<city>/`.
