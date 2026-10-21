# Submission: NOAA GSOD Time-Series Case Study (Review 1 and Review 2)

Team: Kushal Yalamanchi, Kushaal K N, Dhrusheek

| File | Contents |
|---|---|
| `Review_1_Report.pdf` / `review_1_report.tex` | Review 1, revised after the review feedback. Covers the expansion from one station to eight, the data and provenance, exploratory analysis, trend, seasonality, stationarity, ACF/PACF, the forecasting design and the hypotheses |
| `Review_2_Report.pdf` / `review_2_report.tex` | Review 2. Covers ARIMA-family order selection, Holt–Winters, tree ensembles, LSTM and Transformer, baselines, accuracy, paired tests, multi-horizon results, residual analysis, conclusions and individual contributions |
| `references.bib` | Bibliography for both reports |
| `figures/` | Every figure the two reports include |
| `tables/` | LaTeX table bodies generated from the saved results |
| `scripts/build_assets.py` | Rebuilds `figures/` and `tables/` from the repository's saved outputs |
| `scripts/key_numbers.json` | Derived values quoted in the prose, kept for cross-checking |

## Dataset

NOAA National Centers for Environmental Information, *Global Surface Summary of the Day (GSOD)*, version 1.0, dataset identifier `gov.noaa.ncdc:C00516`.

- Download location (one CSV per year and station): <https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/>, for example `2024/43295099999.csv`
- Dataset landing page: <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516>
- Format documentation: <https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt>
- Station history: <https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv>

| City | GSOD station |
|---|---|
| Bengaluru | 43295099999 |
| Delhi | 42182099999 |
| Mumbai | 43003099999 |
| Chennai | 43279099999 |
| Kolkata | 42809099999 |
| Hyderabad | 43128099999 |
| Ahmedabad | 42647099999 |
| Pune | 43063099999 |

Period: 2010-01-01 to 2024-12-31, daily (5,479 days per station). The primary variable is daily mean temperature (`TEMP`, converted from °F to °C); precipitation (`PRCP`, converted from inches to mm) is used for description only.

## Compiling the reports

Both documents compile with XeLaTeX, LuaLaTeX or pdfLaTeX; the preamble selects fonts according to the engine. Run the commands from this folder.

With Tectonic (used to produce the included PDFs):

```bash
tectonic review_1_report.tex
```

```bash
tectonic review_2_report.tex
```

With a TeX Live or MacTeX installation:

```bash
latexmk -xelatex review_1_report.tex
```

```bash
latexmk -xelatex review_2_report.tex
```

Tectonic and latexmk write `review_1_report.pdf` and `review_2_report.pdf`; the submitted copies are those files renamed to `Review_1_Report.pdf` and `Review_2_Report.pdf`. Course name, institution and instructor can be set in the three `\newcommand*` lines at the top of each `.tex` file; when left empty they are omitted from the title page.

## Regenerating figures and tables

The reports do not refit any model. `scripts/build_assets.py` reads the saved outputs from the project repository and does three things:

- copies the existing figures from `figures/`;
- extracts four figures embedded in the executed notebook `notebooks/eight_city_weather_forecasting_comparison.ipynb`;
- draws two summary plots (`rmse_skill_heatmap.png`, `difficulty_vs_variability.png`) and writes the table bodies, using `results/city_comparison.json`, `results/rubric_extension/city_model_comparison.json` and `data/processed/`.

From the repository root, with numpy, pandas, matplotlib, scipy and statsmodels installed:

```bash
python submission/scripts/build_assets.py
```

To regenerate the underlying results from the raw NOAA files, follow the repository README: `src/download_data.py`, then `src/run_all_cities.py`, then `src/rubric_extension.py`.
