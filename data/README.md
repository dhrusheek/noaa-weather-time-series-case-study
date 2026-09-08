# Data

## Source
NOAA / NCEI **Global Summary of the Day (GSOD)**
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

URL pattern: `.../access/{YEAR}/{STATION}.csv`

## Station
- **BANGALORE, IN** — GSOD id `43295099999` (USAF 432950 / WBAN 99999), VOBL
- 12.9667° N, 77.5833° E, elevation 921 m
- Years: 2019–2024

## Files
- `raw/43295099999_YYYY.csv` — original GSOD CSVs, one per year (small; kept for reproducibility).
- `processed/weather_daily.csv` — cleaned daily series produced by `src/preprocessing.py`.

## GSOD units & sentinels (per NOAA documentation)
| Field | Meaning | Native unit | Missing sentinel |
|---|---|---|---|
| `TEMP`, `MAX`, `MIN` | mean/max/min temperature | °F | 9999.9 |
| `PRCP` | total precipitation | inches | 99.99 |
| `PRCP_ATTRIBUTES` | reporting-period flag A–I | — | `I` = not measured |

## Conversions applied
- `temperature_c = (TEMP − 32) × 5/9`
- `precipitation_mm = PRCP × 25.4`
- `PRCP` rows flagged `I` are set to missing (uncertain 0.00), **not** treated as real zeros.

## Processed columns
`date, STATION, NAME, LATITUDE, LONGITUDE, ELEVATION, temperature_c, tmax_c,
tmin_c, precipitation_mm, temperature_c_interp`

`temperature_c_interp` fills ≤3-day gaps by time interpolation for continuous-series
methods (STL, ACF, modelling); `temperature_c` preserves original missingness.
