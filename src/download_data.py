"""Download NOAA GSOD daily data for a single station.

Source: NOAA/NCEI Global Summary of the Day (GSOD)
URL pattern: https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{YEAR}/{STATION}.csv

Default station: Bengaluru / Bangalore Int'l Airport (VOBL), STATION id 43295099999
(USAF 432950, WBAN 99999).
"""
import io
import os
import sys
import urllib.request

BASE = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
STATION = "43295099999"          # Bengaluru (VOBL)
# Extended coverage: the retrieval is fully automated over HTTPS, so we pull the
# widest continuous window the station offers. Years that 404 are skipped with a
# warning (see download_year), keeping the pipeline reproducible from scratch.
YEARS = list(range(2010, 2025))  # 2010-2024 inclusive (extend for more samples)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_year(station: str, year: int) -> bytes | None:
    url = f"{BASE}/{year}/{station}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] {year}: {exc}")
        return None


def main(station: str = STATION) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    ok = 0
    for year in YEARS:
        data = download_year(station, year)
        if data:
            path = os.path.join(RAW_DIR, f"{station}_{year}.csv")
            with open(path, "wb") as fh:
                fh.write(data)
            print(f"  [OK]  {year}: {len(data)} bytes -> {path}")
            ok += 1
    print(f"Downloaded {ok}/{len(YEARS)} years for station {station}")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else STATION
    main(st)
