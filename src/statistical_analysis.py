"""Statistical time-series analysis: decomposition, stationarity, ACF/PACF."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller. H0: series has a unit root (NON-stationary)."""
    s = series.dropna()
    stat, p, lags, nobs, crit, _ = adfuller(s, autolag="AIC")
    return {
        "test": "ADF",
        "statistic": float(stat),
        "p_value": float(p),
        "lags": int(lags),
        "nobs": int(nobs),
        "crit": {k: float(v) for k, v in crit.items()},
        # reject H0 (p<0.05) => stationary
        "stationary": bool(p < 0.05),
    }


def kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """KPSS. H0: series is (trend-)stationary around a deterministic component."""
    s = series.dropna()
    stat, p, lags, crit = kpss(s, regression=regression, nlags="auto")
    return {
        "test": "KPSS",
        "statistic": float(stat),
        "p_value": float(p),  # note: p is bounded/interpolated by statsmodels
        "lags": int(lags),
        "crit": {k: float(v) for k, v in crit.items()},
        # fail to reject H0 (p>=0.05) => stationary
        "stationary": bool(p >= 0.05),
    }


def stationarity_conclusion(adf: dict, kpss_: dict) -> str:
    a, k = adf["stationary"], kpss_["stationary"]
    if a and k:
        return "Stationary (ADF & KPSS agree)"
    if not a and not k:
        return "Non-stationary (ADF & KPSS agree)"
    if a and not k:
        return "Difference-stationary / trend present (ADF rejects unit root, KPSS rejects stationarity)"
    return "Trend-stationary (ADF cannot reject unit root, KPSS cannot reject stationarity)"
