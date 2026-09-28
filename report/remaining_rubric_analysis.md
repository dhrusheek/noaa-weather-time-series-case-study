# Remaining Forecasting Rubric: Eight-City Results

## Evaluation setup

Each city uses NOAA GSOD daily mean temperature for 2010–2024. The common chronological partition is 70% train, 15% validation, and 15% test, with the same 30-day input window and common test dates. Classical ARIMA-family and Holt-Winters models use the first 85% available history and update their one-step states with each observed test value. Tree models use lagged temperature plus known calendar features and are fit on the first 85%. LSTM and Transformer use the original train-only fit, validation early stopping, and test window. No future measured weather values enter the exogenous calendar features.

## Model families and distinctions

- **ARIMA:** non-seasonal ARIMA with p, q selected from a grid and d chosen using ADF/KPSS stationarity tests.
- **SARIMA:** ARIMA plus weekly seasonal AR and MA terms (period 7), with P/Q also selected by information criteria.
- **ARMAX:** ARMA errors plus deterministic annual and weekly Fourier calendar predictors; p, q selected by information criteria.
- **SARIMAX:** seasonal ARIMA plus the deterministic calendar predictors; p, q, P, Q selected by information criteria.
- For every city, ADF and KPSS on the pre-test 85% select d; ACF/PACF correlograms provide initial p/q hints, then all p,q in 0..2 are swept. Seasonal P,Q in 0..1 are swept for seasonal models (weekly period 7; D fixed at 0). AIC and BIC are computed for every candidate. BIC selects the forecast model used in the city comparison; the AIC-selected orders are also saved. The held-out test set is used only for final accuracy evaluation.
- **Holt-Winters:** additive annual seasonality (period 365), with and without a damped additive trend.
- **Machine learning:** Random Forest and Gradient Boosting using the past 30 daily temperatures plus calendar Fourier terms.
- **Deep learning:** LSTM and Transformer as in the base analysis.
- **Baselines:** persistence, 7-day seasonal naive, 365-day seasonal naive, train-mean climatology, and ARIMA.

Calendar terms are deterministic, known-ahead regressors rather than observed exogenous weather. These models test whether annual/weekly timing improves forecasts; they do not test the value of adding rainfall or other measured predictors.

## Selected ARIMA-family orders

Orders below are selected independently per city. The result JSON stores every candidate's AIC, BIC, and convergence status under `information_criterion_order_selection`.

| City | d | Family | BIC-selected (p,d,q)(P,D,Q,7) | BIC | AIC-selected order | AIC |
|---|---:|---|---|---:|---|---:|
| Bengaluru | 0 | ARIMA | (1, 0, 2) (0, 0, 0, 0) | 11699.9 | (1, 0, 2) (0, 0, 0, 0) | 11667.7 |
| Bengaluru | 0 | SARIMA | (1, 0, 2) (0, 0, 1, 7) | 11686.7 | (1, 0, 2) (0, 0, 1, 7) | 11648.0 |
| Bengaluru | 0 | ARMAX | (1, 0, 0) (0, 0, 0, 0) | 11392.4 | (1, 0, 0) (0, 0, 0, 0) | 11321.5 |
| Bengaluru | 0 | SARIMAX | (1, 0, 0) (0, 0, 1, 7) | 11383.8 | (1, 0, 1) (0, 0, 1, 7) | 11305.3 |
| Delhi | 0 | ARIMA | (2, 0, 1) (0, 0, 0, 0) | 17009.0 | (2, 0, 1) (0, 0, 0, 0) | 16976.8 |
| Delhi | 0 | SARIMA | (2, 0, 1) (1, 0, 0, 7) | 16991.9 | (2, 0, 1) (1, 0, 0, 7) | 16953.2 |
| Delhi | 0 | ARMAX | (1, 0, 1) (0, 0, 0, 0) | 16673.7 | (1, 0, 1) (0, 0, 0, 0) | 16596.3 |
| Delhi | 0 | SARIMAX | (1, 0, 1) (0, 0, 1, 7) | 16652.0 | (1, 0, 1) (0, 0, 1, 7) | 16568.2 |
| Mumbai | 0 | ARIMA | (1, 0, 2) (0, 0, 0, 0) | 11917.1 | (1, 0, 2) (0, 0, 0, 0) | 11884.8 |
| Mumbai | 0 | SARIMA | (1, 0, 2) (0, 0, 1, 7) | 11901.6 | (1, 0, 2) (0, 0, 1, 7) | 11862.9 |
| Mumbai | 0 | ARMAX | (2, 0, 0) (0, 0, 0, 0) | 11619.0 | (2, 0, 2) (0, 0, 0, 0) | 11538.5 |
| Mumbai | 0 | SARIMAX | (2, 0, 0) (0, 0, 0, 0) | 11619.0 | (2, 0, 2) (0, 0, 0, 0) | 11538.5 |
| Chennai | 0 | ARIMA | (2, 0, 1) (0, 0, 0, 0) | 11375.2 | (2, 0, 1) (0, 0, 0, 0) | 11343.0 |
| Chennai | 0 | SARIMA | (2, 0, 1) (0, 0, 0, 0) | 11375.2 | (2, 0, 1) (0, 0, 0, 0) | 11343.0 |
| Chennai | 0 | ARMAX | (2, 0, 2) (0, 0, 0, 0) | 11239.4 | (2, 0, 2) (0, 0, 0, 0) | 11149.2 |
| Chennai | 0 | SARIMAX | (2, 0, 2) (0, 0, 1, 7) | 11221.5 | (2, 0, 2) (0, 0, 1, 7) | 11124.8 |
| Kolkata | 0 | ARIMA | (2, 0, 2) (0, 0, 0, 0) | 14928.7 | (2, 0, 2) (0, 0, 0, 0) | 14890.0 |
| Kolkata | 0 | SARIMA | (2, 0, 2) (1, 0, 0, 7) | 14922.1 | (2, 0, 2) (1, 0, 0, 7) | 14877.0 |
| Kolkata | 0 | ARMAX | (1, 0, 1) (0, 0, 0, 0) | 14625.1 | (1, 0, 1) (0, 0, 0, 0) | 14547.8 |
| Kolkata | 0 | SARIMAX | (1, 0, 1) (1, 0, 0, 7) | 14616.2 | (1, 0, 1) (1, 0, 1, 7) | 14532.0 |
| Hyderabad | 0 | ARIMA | (2, 0, 1) (0, 0, 0, 0) | 15248.4 | (2, 0, 1) (0, 0, 0, 0) | 15216.2 |
| Hyderabad | 0 | SARIMA | (2, 0, 1) (0, 0, 1, 7) | 15232.2 | (2, 0, 1) (0, 0, 1, 7) | 15193.6 |
| Hyderabad | 0 | ARMAX | (1, 0, 2) (0, 0, 0, 0) | 15127.4 | (1, 0, 2) (0, 0, 0, 0) | 15043.6 |
| Hyderabad | 0 | SARIMAX | (1, 0, 2) (0, 0, 1, 7) | 15110.5 | (1, 0, 2) (0, 0, 1, 7) | 15020.3 |
| Ahmedabad | 0 | ARIMA | (2, 0, 2) (0, 0, 0, 0) | 14442.7 | (2, 0, 2) (0, 0, 0, 0) | 14404.0 |
| Ahmedabad | 0 | SARIMA | (2, 0, 2) (0, 0, 1, 7) | 14436.2 | (2, 0, 2) (0, 0, 1, 7) | 14391.0 |
| Ahmedabad | 0 | ARMAX | (1, 0, 1) (0, 0, 0, 0) | 14170.5 | (2, 0, 2) (0, 0, 0, 0) | 14086.0 |
| Ahmedabad | 0 | SARIMAX | (2, 0, 0) (0, 0, 1, 7) | 14165.1 | (2, 0, 0) (0, 0, 1, 7) | 14081.3 |
| Pune | 0 | ARIMA | (2, 0, 2) (0, 0, 0, 0) | 13027.5 | (2, 0, 2) (0, 0, 0, 0) | 12988.9 |
| Pune | 0 | SARIMA | (2, 0, 2) (0, 0, 1, 7) | 13019.5 | (2, 0, 2) (0, 0, 1, 7) | 12974.4 |
| Pune | 0 | ARMAX | (2, 0, 0) (0, 0, 0, 0) | 12736.6 | (2, 0, 0) (0, 0, 0, 0) | 12659.2 |
| Pune | 0 | SARIMAX | (2, 0, 0) (0, 0, 0, 0) | 12736.6 | (2, 0, 0) (0, 0, 0, 0) | 12659.2 |

## Forecast accuracy and model comparison

| City | Best model | MAE (°C) | RMSE (°C) | Persistence MAE | Best-model MAE skill vs persistence |
|---|---|---:|---:|---:|---:|
| Bengaluru | SARIMAX | 0.656 | 0.826 | 0.668 | +1.7% |
| Chennai | GradientBoosting | 0.621 | 0.845 | 0.649 | +4.4% |
| Mumbai | RandomForest | 0.631 | 0.850 | 0.637 | +1.0% |
| Pune | ARMAX | 0.734 | 0.963 | 0.748 | +1.9% |
| Ahmedabad | SARIMAX | 0.724 | 1.015 | 0.778 | +6.9% |
| Hyderabad | RandomForest | 0.802 | 1.065 | 0.818 | +2.0% |
| Kolkata | ARMAX | 0.919 | 1.208 | 0.922 | +0.3% |
| Delhi | SARIMAX | 1.005 | 1.338 | 1.051 | +4.5% |

### Equal-weight model averages across cities

This table averages the eight station-level test scores equally and shows each model's city-specific RMSE wins and mean RMSE rank.

| Model | Mean MAE (°C) | Mean RMSE (°C) | Mean MASE | Mean R² | RMSE wins | Mean RMSE rank |
|---|---:|---:|---:|---:|---:|---:|
| ARMAX | 0.763 | 1.016 | 0.935 | 0.894 | 2 | 1.75 |
| SARIMAX | 0.763 | 1.016 | 0.935 | 0.894 | 3 | 2.00 |
| RandomForest | 0.777 | 1.027 | 0.952 | 0.893 | 2 | 3.88 |
| GradientBoosting | 0.776 | 1.029 | 0.949 | 0.893 | 1 | 3.75 |
| Transformer | 0.781 | 1.037 | 0.955 | 0.890 | 0 | 5.88 |
| ARIMA | 0.774 | 1.037 | 0.946 | 0.890 | 0 | 5.75 |
| SARIMA | 0.775 | 1.038 | 0.947 | 0.890 | 0 | 6.38 |
| LSTM | 0.789 | 1.051 | 0.960 | 0.889 | 0 | 6.75 |
| Persistence (t-1) | 0.784 | 1.071 | 0.960 | 0.882 | 0 | 8.88 |
| Holt-Winters damped | 0.859 | 1.146 | 1.053 | 0.865 | 0 | 10.12 |
| Holt-Winters additive | 0.859 | 1.146 | 1.053 | 0.865 | 0 | 10.88 |
| Seasonal naive (t-7) | 1.627 | 2.108 | 1.979 | 0.557 | 0 | 12.12 |
| Seasonal naive (t-365) | 1.835 | 2.372 | 2.219 | 0.455 | 0 | 13.00 |
| Mean (climatology) | 2.933 | 3.618 | 3.412 | -0.019 | 0 | 13.88 |

The complete per-city ranking table includes MAE, RMSE, MASE, R², and bias for every model in `results/rubric_extension/<city>.json`. MASE scales MAE by the training-period one-day naive error. A positive skill score means lower MAE than persistence. Absolute error describes forecast difficulty; persistence-relative skill describes incremental value over yesterday's observation.

### Diebold–Mariano paired loss tests

Each model's test-period absolute error is compared with persistence using a paired Diebold–Mariano test with Newey–West/HAC variance (30 lags). Holm correction controls family-wise error across models within each city. A corrected p-value below 0.05 indicates a significant difference in mean absolute loss; negative signed mean loss difference means the model improves on persistence.

## Residual analysis

For each model, the result JSON reports residual mean and standard deviation, Durbin–Watson, Ljung–Box tests at lags 10 and 30, and Jarque–Bera normality. Residual plots and the best-model residual ACF are in `figures/rubric_extension/<city>/`. Ljung–Box significance indicates residual serial dependence; a non-significant Jarque–Bera result is consistent with, but does not prove, normally distributed residuals.

| City | Best model | Residual mean (°C) | Durbin–Watson | Ljung–Box lag-30 p | Jarque–Bera p |
|---|---|---:|---:|---:|---:|
| Bengaluru | SARIMAX | +0.144 | 1.941 | 0.02009 | 1.925e-11 |
| Chennai | GradientBoosting | +0.047 | 1.900 | 0.05074 | 1.813e-51 |
| Mumbai | RandomForest | +0.069 | 1.661 | 0.00524 | 3.407e-34 |
| Pune | ARMAX | +0.059 | 1.933 | 0.006088 | 5.507e-21 |
| Ahmedabad | SARIMAX | -0.077 | 1.886 | 0.301 | 3.006e-116 |
| Hyderabad | RandomForest | +0.015 | 1.789 | 0.03682 | 2.743e-34 |
| Kolkata | ARMAX | +0.205 | 1.742 | 0.07966 | 5.27e-84 |
| Delhi | SARIMAX | -0.112 | 1.927 | 0.04806 | 1.33e-25 |

## Research conclusion


On the shared 2022-10-02 to 2024-12-31 holdout, Bengaluru has the lowest best-model RMSE (0.826 deg C), while Delhi has the highest (1.338 deg C). No model wins in every city: SARIMAX leads in 3; ARMAX leads in 2; RandomForest leads in 2; GradientBoosting leads in 1.
After Holm correction, the best model's absolute loss is significantly lower than persistence in Chennai, Ahmedabad, Delhi. In the other cities, this test did not find a significant difference; that does not establish equivalence.
Best-model residuals retain significant autocorrelation at lag 30 for Bengaluru, Mumbai, Pune, Hyderabad, Delhi; Ljung-Box results at lag 10 and lag 30 are in the table above.
Jarque-Bera rejects residual normality for all eight selected best models, so normal-error assumptions for uncertainty intervals need further checking. Best-model residual biases remain below 0.2 deg C in magnitude. The base direct 1-30 day LSTM/Transformer comparison found positive mean-horizon Transformer skill over persistence in all eight cities (11.8% to 30.6%); see `report/city_comparison.md`.
Use RMSE to compare absolute forecast difficulty and skill relative to persistence to judge the gain from more complex models. This observational analysis represents one station per city and one temporal holdout; it does not establish city-wide predictability or causal effects.

## Individual contribution (5 marks)

Fill this table with the actual team members and work completed. The project files cannot establish who performed each contribution, so names and ownership are intentionally left for the team to confirm.

| Member | Contribution completed | Evidence (file, analysis, or presentation work) |
|---|---|---|
| [Name] | [Work performed] | [Evidence] |
| [Name] | [Work performed] | [Evidence] |
| [Name] | [Work performed] | [Evidence] |

## Reproduction

Run `python src/run_all_cities.py` for the base city analysis, then `python src/rubric_extension.py` for ARIMA-family, Holt-Winters, tree-model, and residual outputs. Use `python src/rubric_extension.py --skip-existing` to continue a partial run from saved city results. Both use files in `data/raw/`; download them with `python src/download_data.py` if needed.
