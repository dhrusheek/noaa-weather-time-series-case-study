# Data

## Source
NOAA / NCEI **Global Summary of the Day (GSOD)**
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

URL pattern: `.../access/{YEAR}/{STATION}.csv`

## Stations
The city comparison uses one fixed GSOD station per city, configured in
`src/cities.py`. Each station has yearly CSVs downloaded for 2010–2024.

| City | NOAA station | GSOD station name |
|---|---:|---|
| Bengaluru | `43295099999` | BANGALORE |
| Delhi | `42182099999` | SAFDARJUNG |
| Mumbai | `43003099999` | CHHATRAPATI SHIVAJI MAHARAJ INTL |
| Chennai | `43279099999` | CHENNAI INTL |
| Kolkata | `42809099999` | NETAJI SUBHASH CHANDRA BOSE INTL |
| Hyderabad | `43128099999` | BEGUMPET AIRPORT / HYDERABAD INTL |
| Ahmedabad | `42647099999` | AHMEDABAD / SARDAR VALLABHBHAI PATEL INTL |
| Pune | `43063099999` | PUNE |

## Files
- `raw/<STATION>_YYYY.csv` — original GSOD CSVs, by station and year (small; kept for reproducibility).
- `processed/weather_daily_<STATION>.csv` — cleaned daily series produced by the city analysis.

## Bengaluru missing data (after cleaning; other cities vary)
- `temperature_c`: 27 missing (0.5 %), filled by ≤3-day time interpolation into `temperature_c_interp`.
- `precipitation_mm`: 2,594 missing — mostly `PRCP` flag `I` (not measured); kept as NaN, **not** interpolated.

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
