"""NOAA GSOD station choices for the Indian city comparison.

Station IDs are USAF + WBAN as listed in NOAA/NCEI isd-history.csv. These
airport/observatory stations provide a common 2010-2024 daily comparison window.
"""
CITIES = {
    "Bengaluru": {"station": "43295099999"},
    "Delhi": {"station": "42182099999"},
    "Mumbai": {"station": "43003099999"},
    "Chennai": {"station": "43279099999"},
    "Kolkata": {"station": "42809099999"},
    "Hyderabad": {"station": "43128099999"},
    "Ahmedabad": {"station": "42647099999"},
    "Pune": {"station": "43063099999"},
}

YEARS = range(2010, 2025)
