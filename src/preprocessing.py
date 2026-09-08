"""Load, clean and standardise NOAA GSOD daily data for one station.

GSOD units / sentinels (per NOAA GSOD documentation):
  TEMP, MAX, MIN : mean/max/min temperature in degrees Fahrenheit; missing = 9999.9
  PRCP           : total precipitation in INCHES; missing = 99.99
  PRCP_ATTRIBUTES: single-letter flag describing the reporting period.
                   Flag 'I' => station did not report precipitation and no
                   occurrence was noted in hourly obs => amount is UNCERTAIN,
                   so we treat 'I'-flagged 0.00 values as MISSING (not a true 0).
                   All other flags (A-H) are genuine measurements (0.00 = real 0).

Conversions:
  degC = (degF - 32) * 5/9
  mm   = inches * 25.4
"""
import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "weather_daily.csv"
)

TEMP_MISSING = 9999.9
PRCP_MISSING = 99.99


def f_to_c(series: pd.Series) -> pd.Series:
    return (series - 32.0) * 5.0 / 9.0


def load_raw(station: str = "43295099999") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, f"{station}_*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw files for station {station} in {RAW_DIR}")
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # --- date parsing & ordering -------------------------------------------
    df["date"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("date").reset_index(drop=True)
    # drop duplicate timestamps (keep first)
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    # --- temperature: sentinel -> NaN, F -> C ------------------------------
    for col in ["TEMP", "MAX", "MIN"]:
        df.loc[df[col] >= TEMP_MISSING, col] = np.nan
    df["temperature_c"] = f_to_c(df["TEMP"])
    df["tmax_c"] = f_to_c(df["MAX"])
    df["tmin_c"] = f_to_c(df["MIN"])

    # --- precipitation: sentinel + flag semantics --------------------------
    prcp = df["PRCP"].copy()
    prcp[prcp >= PRCP_MISSING] = np.nan
    # Flag 'I' => not actually measured -> treat as missing, not a real zero.
    flag_i = df["PRCP_ATTRIBUTES"].astype(str).str.strip() == "I"
    prcp[flag_i] = np.nan
    df["precipitation_mm"] = prcp * 25.4  # inches -> mm

    # --- reindex to a continuous daily calendar (exposes missing dates) ----
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full)
    df.index.name = "date"

    keep = ["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION",
            "temperature_c", "tmax_c", "tmin_c", "precipitation_mm"]
    out = df[keep].copy()
    # station metadata is constant -> forward/back fill the descriptors only
    for c in ["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION"]:
        out[c] = out[c].ffill().bfill()

    # --- small-gap temperature interpolation (time-based, <=3 days) --------
    out["temperature_c_interp"] = out["temperature_c"].interpolate(
        method="time", limit=3, limit_direction="both"
    )
    return out


def build_processed(station: str = "43295099999", save: bool = True) -> pd.DataFrame:
    raw = load_raw(station)
    out = clean(raw)
    if save:
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        out.reset_index().to_csv(PROCESSED_PATH, index=False)
    return out


if __name__ == "__main__":
    d = build_processed()
    print("Processed shape:", d.shape)
    print("Date range:", d.index.min().date(), "->", d.index.max().date())
    print("Missing temperature_c:", int(d["temperature_c"].isna().sum()))
    print("Missing precipitation_mm:", int(d["precipitation_mm"].isna().sum()))
    print("Saved ->", PROCESSED_PATH)
