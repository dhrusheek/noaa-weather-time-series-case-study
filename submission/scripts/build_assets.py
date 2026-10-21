"""Build the figures and LaTeX tables used by the Review 1 and Review 2 reports.

Every number in the generated tables is read from the saved project outputs
(results/*.json, data/processed/*.csv) or from the executed notebook. Nothing is
refitted: the only computations here are descriptive statistics on the processed
series (monthly means, mean absolute daily change, correlogram values) that the
notebook itself also computes, plus ratios of saved metrics.

Run from the repository root:
    python submission/scripts/build_assets.py
"""
import base64
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submission"
FIG = SUB / "figures"
TAB = SUB / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "src"))
from cities import CITIES  # noqa: E402

base = json.load(open(ROOT / "results" / "city_comparison.json", encoding="utf-8"))
ext = json.load(open(ROOT / "results" / "rubric_extension" / "city_model_comparison.json", encoding="utf-8"))
CITY = base["cities"]
A = base["per_city_results"]
B = ext["per_city_results"]
KEY = {}  # numbers quoted in the prose, dumped for cross-checking


def series(city):
    st = CITIES[city]["station"]
    return pd.read_csv(ROOT / "data" / "processed" / f"weather_daily_{st}.csv",
                       parse_dates=["date"]).set_index("date")


def write(name, text):
    (TAB / name).write_text(text, encoding="utf-8")


def f(x, nd=3):
    out = f"{x:.{nd}f}"
    if float(out) == 0:
        out = out.lstrip("-")
    return out.replace("-", "$-$")


def sgn(x, nd=3):
    out = f"{x:+.{nd}f}"
    if float(out) == 0:
        out = f"{0:.{nd}f}"
    return out.replace("-", "$-$")


def p_fmt(p):
    if p >= 0.0995:
        return r"$\geq$0.10"
    if p < 1e-4:
        return r"$<$0.0001"
    return f"{p:.4f}"


# --------------------------------------------------------------------------- #
# 1. Figures copied from the repository (unchanged)
# --------------------------------------------------------------------------- #
copies = {
    "bengaluru_rolling_statistics.png": "figures/cities/bengaluru/rolling_statistics.png",
    "bengaluru_subseries.png": "figures/cities/bengaluru/subseries_temperature.png",
    "bengaluru_precipitation.png": "figures/cities/bengaluru/precipitation_timeseries.png",
    "bengaluru_monthly_precipitation.png": "figures/cities/bengaluru/monthly_precipitation.png",
    "bengaluru_acf_raw.png": "figures/cities/bengaluru/acf_raw.png",
    "bengaluru_pacf_raw.png": "figures/cities/bengaluru/pacf_raw.png",
    "bengaluru_acf_diff.png": "figures/cities/bengaluru/acf_stationary.png",
    "bengaluru_pacf_diff.png": "figures/cities/bengaluru/pacf_stationary.png",
    "delhi_stl.png": "figures/cities/delhi/seasonal_decomposition.png",
    "bengaluru_error_vs_horizon.png": "figures/cities/bengaluru/error_vs_horizon.png",
    "bengaluru_error_spikes.png": "figures/cities/bengaluru/error_spikes.png",
    "bengaluru_lstm_loss.png": "figures/cities/bengaluru/lstm_loss.png",
    "bengaluru_transformer_loss.png": "figures/cities/bengaluru/transformer_loss.png",
    "transformer_horizon_mae.png": "figures/city_comparison/transformer_horizon_mae.png",
    "bengaluru_residual_diagnostics.png": "figures/rubric_extension/bengaluru/residual_diagnostics.png",
}
for city in CITY:
    copies[f"acf_longlag_{city.lower()}.png"] = f"figures/cities/{city.lower()}/acf_longlag.png"
for dst, src in copies.items():
    shutil.copyfile(ROOT / src, FIG / dst)

# --------------------------------------------------------------------------- #
# 2. Figures embedded in the executed eight-city notebook (extracted, unchanged)
# --------------------------------------------------------------------------- #
nb = json.load(open(ROOT / "notebooks" / "eight_city_weather_forecasting_comparison.ipynb", encoding="utf-8"))
signatures = {
    "eight_city_daily_temperature.png": 'fig.suptitle("Daily mean temperature by city',
    "eight_city_monthly_climatology.png": 'fig.suptitle("Mean temperature by calendar month"',
    "eight_city_acf_pacf_pretest.png": "calendar-adjusted",
    "eight_city_residual_acf.png": "selected-model residual ACF",
}
for dst, sig in signatures.items():
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and sig in "".join(c["source"]))
    png = next(o["data"]["image/png"] for o in cell["outputs"] if "image/png" in o.get("data", {}))
    (FIG / dst).write_bytes(base64.b64decode(png))

# --------------------------------------------------------------------------- #
# 3. Descriptive statistics from the processed series
# --------------------------------------------------------------------------- #
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
desc = {}
for city in CITY:
    df = series(city)
    t = df["temperature_c_interp"]
    clim = t.groupby(t.index.month).mean()
    p = df["precipitation_mm"]
    full_pacf = sm_pacf(t.values, nlags=3, method="ywm")
    desc[city] = {
        "mean": float(df["temperature_c"].mean()), "sd": float(df["temperature_c"].std()),
        "min": float(df["temperature_c"].min()), "max": float(df["temperature_c"].max()),
        "warm_m": MONTHS[int(clim.idxmax()) - 1], "warm": float(clim.max()),
        "cool_m": MONTHS[int(clim.idxmin()) - 1], "cool": float(clim.min()),
        "amp": float(clim.max() - clim.min()),
        "abs_change": float(t.diff().abs().mean()),
        "still_nan": int(t.isna().sum()),
        "p_n": int(p.notna().sum()), "p_missing_pct": 100 * float(p.isna().mean()),
        "p_wet_pct": 100 * float((p[p.notna()] > 0).mean()),
        "p_median": float(p.median()), "p_mean": float(p.mean()), "p_max": float(p.max()),
        "pacf1": float(full_pacf[1]), "pacf2": float(full_pacf[2]), "pacf3": float(full_pacf[3]),
    }
KEY["descriptive"] = desc

# Station table
rows = []
for city in CITY:
    st, ds = A[city]["station"], A[city]["dataset"]
    rows.append(
        f"{city} & \\texttt{{{st['id']}}} & {st['name'].replace(', IN', '').title()} & "
        f"{st['lat']:.3f} & {st['lon']:.3f} & {st['elevation_m']:.0f} & "
        f"{ds['missing_temperature_c']} ({100 * ds['missing_temperature_c'] / ds['n_obs']:.1f}) & "
        f"{ds['missing_precipitation_mm']:,} ({100 * ds['missing_precipitation_mm'] / ds['n_obs']:.0f}) \\\\")
write("r1_stations.tex", "\n".join(rows) + "\n")

# Temperature descriptive table
rows = []
for city in CITY:
    d = desc[city]
    rows.append(f"{city} & {f(d['mean'], 2)} & {f(d['sd'], 2)} & {f(d['min'], 1)} & {f(d['max'], 1)} & "
                f"{d['warm_m']} ({f(d['warm'], 1)}) & {d['cool_m']} ({f(d['cool'], 1)}) & "
                f"{f(d['amp'], 1)} & {f(d['abs_change'], 2)} \\\\")
write("r1_descriptive.tex", "\n".join(rows) + "\n")

# Trend and seasonality
rows = []
for city in CITY:
    dec, ac = A[city]["decomposition"], A[city]["acf"]
    agree = np.sign(dec["linear_slope_per_decade"]) == np.sign(dec["stl_trend_delta"])
    rows.append(f"{city} & {f(dec['seasonal_strength'])} & {f(dec['trend_strength'])} & "
                f"{sgn(dec['linear_slope_per_decade'], 2)} & {sgn(dec['stl_trend_delta'], 2)} & "
                f"{'yes' if agree else 'no'} & {f(ac['acf_lag365_raw'])} \\\\")
write("r1_trend_season.tex", "\n".join(rows) + "\n")
KEY["trend_sign_disagree"] = [c for c in CITY if np.sign(A[c]["decomposition"]["linear_slope_per_decade"])
                              != np.sign(A[c]["decomposition"]["stl_trend_delta"])]

# Stationarity
rows = []
for city in CITY:
    s = A[city]["stationarity"]
    r, d = s["raw"], s["diff"]
    rows.append(f"{city} & {f(r['adf']['statistic'], 2)} & {p_fmt(r['adf']['p_value'])} & "
                f"{f(r['kpss']['statistic'], 3)} & {p_fmt(r['kpss']['p_value'])} & "
                f"{f(d['adf']['statistic'], 2)} & {p_fmt(d['adf']['p_value'])} & "
                f"{f(d['kpss']['statistic'], 3)} & {p_fmt(d['kpss']['p_value'])} \\\\")
write("r1_stationarity.tex", "\n".join(rows) + "\n")

# Autocorrelation
rows = []
for city in CITY:
    ac, d = A[city]["acf"], desc[city]
    rows.append(f"{city} & {f(ac['lag1_raw'])} & {f(ac['acf_lag7_raw'])} & {f(ac['acf_lag365_raw'])} & "
                f"{f(d['pacf1'])} & {sgn(d['pacf2'])} & {sgn(d['pacf3'])} & {sgn(ac['lag1_diff'])} \\\\")
write("r1_acf.tex", "\n".join(rows) + "\n")

# Precipitation
rows = []
for city in CITY:
    d = desc[city]
    rows.append(f"{city} & {d['p_n']:,} & {f(d['p_missing_pct'], 1)} & {f(d['p_wet_pct'], 1)} & "
                f"{f(d['p_median'], 2)} & {f(d['p_mean'], 2)} & {f(d['p_max'], 1)} \\\\")
write("r1_precip.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- #
# 4. Review 2 tables
# --------------------------------------------------------------------------- #
def calendar_features(index):
    """Identical to src/rubric_extension.py (annual k = 1-3 and weekly Fourier terms)."""
    idx = pd.DatetimeIndex(index)
    annual = 2 * np.pi * (idx.dayofyear.to_numpy() - 1) / 365.25
    weekly = 2 * np.pi * np.arange(len(idx), dtype=float) / 7.0
    cols = [fn(k * annual) for k in (1, 2, 3) for fn in (np.sin, np.cos)]
    return np.column_stack(cols + [np.sin(weekly), np.cos(weekly)])


def order_str(o, so):
    s = f"({o[0]},{o[1]},{o[2]})"
    if any(so[:3]):
        s += f"$\\times$({so[0]},{so[1]},{so[2]})$_{{{so[3]}}}$"
    return s


rows, corr = [], {}
for city in CITY:
    df = series(city)
    y = df["temperature_c_interp"].to_numpy(dtype=float)
    y = y[: int(len(y) * 0.85)]
    X = np.column_stack([np.ones(len(y)), calendar_features(df.index[: len(y)])])
    an = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ra, rp = sm_acf(y, nlags=30, fft=True), sm_pacf(y, nlags=30, method="ywm")
    aa, ap = sm_acf(an, nlags=30, fft=True), sm_pacf(an, nlags=30, method="ywm")
    corr[city] = {"raw_pacf1": rp[1], "raw_pacf_max27": float(np.abs(rp[2:8]).max()),
                  "adj_acf1": aa[1], "adj_acf7": aa[7], "adj_acf30": aa[30],
                  "adj_pacf1": ap[1], "adj_pacf2": ap[2], "adj_pacf7": ap[7],
                  "band": 1.96 / np.sqrt(len(y))}
    sel = B[city]["information_criterion_order_selection"]
    fam = {k: sel[k]["selected"] for k in ("ARIMA", "SARIMA", "ARMAX", "SARIMAX")}
    c = corr[city]
    rows.append(f"{city} & {f(c['raw_pacf1'], 2)} & {f(c['adj_pacf1'], 2)} & {sgn(c['adj_pacf2'], 2)} & "
                f"{sgn(c['adj_pacf7'], 2)} & {f(c['adj_acf7'], 2)} & "
                + " & ".join(order_str(fam[k]["order"], fam[k]["seasonal_order"]) for k in fam)
                + f" & {fam['ARIMA']['bic'] - fam['ARMAX']['bic']:.0f} \\\\")
write("r2_orders.tex", "\n".join(rows) + "\n")
KEY["correlogram"] = {c: {k: float(v) for k, v in d.items()} for c, d in corr.items()}
agree = sum(tuple(B[c]["information_criterion_order_selection"][k]["selected"]["order"])
            == tuple(B[c]["information_criterion_order_selection"][k]["selected_by_aic"]["order"])
            and tuple(B[c]["information_criterion_order_selection"][k]["selected"]["seasonal_order"])
            == tuple(B[c]["information_criterion_order_selection"][k]["selected_by_aic"]["seasonal_order"])
            for c in CITY for k in ("ARIMA", "SARIMA", "ARMAX", "SARIMAX"))
KEY["aic_bic_agree_of_32"] = int(agree)
KEY["candidates_total"] = int(sum(B[c]["information_criterion_order_selection"][k]["candidate_count"]
                                  for c in CITY for k in ("ARIMA", "SARIMA", "ARMAX", "SARIMAX")))
KEY["candidates_valid"] = int(sum(B[c]["information_criterion_order_selection"][k]["valid_candidate_count"]
                                  for c in CITY for k in ("ARIMA", "SARIMA", "ARMAX", "SARIMAX")))

LABEL = {"Persistence (t-1)": "Persistence", "Seasonal naive (t-7)": "Seasonal naive ($m{=}7$)",
         "Seasonal naive (t-365)": "Seasonal naive ($m{=}365$)", "Mean (climatology)": "Climatology mean",
         "Holt-Winters additive": "Holt--Winters", "Holt-Winters damped": "Holt--Winters (damped)",
         "RandomForest": "Random forest", "GradientBoosting": "Gradient boosting"}
lab = lambda m: LABEL.get(m, m)  # noqa: E731

# Per-city one-step leaders
ranked = sorted(CITY, key=lambda c: min(v["rmse"] for v in B[c]["model_metrics"].values()))
rows = []
for city in ranked:
    mm = B[city]["model_metrics"]
    best = min(mm, key=lambda m: mm[m]["rmse"])
    per = mm["Persistence (t-1)"]
    dm = B[city]["diebold_mariano_absolute_loss_vs_persistence"][best]
    rows.append(f"{city} & {lab(best)} & {f(mm[best]['mae'])} & {f(mm[best]['rmse'])} & "
                f"{f(per['mae'])} & {f(per['rmse'])} & {f(100 * (1 - mm[best]['rmse'] / per['rmse']), 1)} & "
                f"{f(100 * (1 - mm[best]['mae'] / per['mae']), 1)} & {sgn(dm['mean_abs_loss_difference'], 3)} & "
                f"{p_fmt(dm['holm_adjusted_p_value'])}{'$^{*}$' if dm['significant_at_0_05'] else ''} \\\\")
write("r2_onestep_best.tex", "\n".join(rows) + "\n")

# Family averages
rows = []
per_mean = next(r["mean_rmse"] for r in ext["models_ranked_by_mean_city_rmse"] if r["model"] == "Persistence (t-1)")
for r in ext["models_ranked_by_mean_city_rmse"]:
    rows.append(f"{lab(r['model'])} & {f(r['mean_mae'])} & {f(r['mean_rmse'])} & "
                f"{sgn(100 * (1 - r['mean_rmse'] / per_mean), 1)} & {f(r['mean_mase'])} & {f(r['mean_r2'])} & "
                f"{r['cities_won_by_rmse']} & {f(r['mean_rmse_rank'], 2)} \\\\")
write("r2_family.tex", "\n".join(rows) + "\n")

# Full appendix matrices (RMSE and MAE)
MODELS = [r["model"] for r in ext["models_ranked_by_mean_city_rmse"]]
for metric in ("rmse", "mae"):
    rows = []
    for m in MODELS:
        vals = [B[c]["model_metrics"][m][metric] for c in CITY]
        best_each = [min(B[c]["model_metrics"][k][metric] for k in MODELS) for c in CITY]
        cells = [(f"\\textbf{{{f(v)}}}" if abs(v - b) < 1e-12 else f(v)) for v, b in zip(vals, best_each)]
        rows.append(f"{lab(m)} & " + " & ".join(cells) + " \\\\")
    write(f"r2_full_{metric}.tex", "\n".join(rows) + "\n")

# Stage A one-step (fixed ARIMA, neural)
rows = []
for city in CITY:
    o = A[city]["onestep"]
    s = {**o["baselines"], **o["models"]}
    p = s["Persistence (t-1)"]["mae"]
    rows.append(f"{city} & {f(p)} & {f(s['ARIMA(2, 0, 2)']['mae'])} & {f(s['LSTM']['mae'])} & "
                f"{f(s['Transformer']['mae'])} & {sgn(100 * (1 - s['LSTM']['mae'] / p), 1)} & "
                f"{sgn(100 * (1 - s['Transformer']['mae'] / p), 1)} \\\\")
write("r2_stageA.tex", "\n".join(rows) + "\n")

# Multi-horizon
rows, mh_key = [], {}
for city in CITY:
    mh = A[city]["multi_horizon"]
    H = mh["horizons"]
    mean = {m: float(np.mean([mh[m][str(h)]["mae"] for h in H]))
            for m in ("Persistence", "Seasonal naive (365)", "LSTM", "Transformer")}
    tr = [mh["Transformer"][str(h)]["mae"] for h in H]
    sn30, tr30 = mh["Seasonal naive (365)"]["30"]["mae"], mh["Transformer"]["30"]["mae"]
    mh_key[city] = {"mean": mean, "sn30": sn30, "tr30": tr30,
                    "lstm_h1": mh["LSTM"]["1"]["mae"], "pers_h1": mh["Persistence"]["1"]["mae"],
                    "tr_h1": mh["Transformer"]["1"]["mae"]}
    rows.append(f"{city} & " + " & ".join(f(x, 2) for x in tr) + f" & {f(mean['Persistence'], 2)} & "
                f"{f(mean['Seasonal naive (365)'], 2)} & {f(mean['LSTM'], 2)} & {f(mean['Transformer'], 2)} & "
                f"{f(100 * (1 - mean['Transformer'] / mean['Persistence']), 1)} & "
                f"{f(100 * (1 - mean['Transformer'] / mean['Seasonal naive (365)']), 1)} \\\\")
write("r2_horizon.tex", "\n".join(rows) + "\n")
KEY["multi_horizon"] = mh_key

# Residual diagnostics of the selected model
rows, res_key = [], {}
for city in CITY:
    r = B[city]
    best = min(r["model_metrics"], key=lambda m: r["model_metrics"][m]["rmse"])
    d = r["residual_diagnostics"][best]
    e = np.asarray(r["test_residuals"][best], dtype=float)
    sd = e.std(ddof=1)
    c = e - e.mean()
    lb_fail = sum(v["ljung_box_lag30_p"] < 0.05 for v in r["residual_diagnostics"].values())
    res_key[city] = {"best": best, "t": float(d["mean"] / (sd / np.sqrt(len(e)))),
                     "acf1": float(sm_acf(e, nlags=1, fft=True)[1]),
                     "skew": float(stats.skew(e)), "exkurt": float(stats.kurtosis(e)),
                     "lo": float(100 * np.mean(c < -1.96 * sd)), "hi": float(100 * np.mean(c > 1.96 * sd)),
                     "lb_fail": int(lb_fail)}
    k = res_key[city]
    rows.append(f"{city} & {lab(best)} & {sgn(d['mean'])} & {sgn(k['t'], 1)} & {f(d['std'])} & "
                f"{f(d['durbin_watson'], 2)} & {p_fmt(d['ljung_box_lag10_p'])} & {p_fmt(d['ljung_box_lag30_p'])} & "
                f"{d['jarque_bera_stat']:.0f} & {sgn(k['skew'], 2)} & {f(k['exkurt'], 2)} & "
                f"{f(k['lo'], 1)} / {f(k['hi'], 1)} & {lb_fail} \\\\")
write("r2_residuals.tex", "\n".join(rows) + "\n")
KEY["residuals"] = res_key

# Difficulty correlates
rho_change = stats.spearmanr([min(v["rmse"] for v in B[c]["model_metrics"].values()) for c in CITY],
                             [desc[c]["abs_change"] for c in CITY])
rho_season = stats.spearmanr([min(v["rmse"] for v in B[c]["model_metrics"].values()) for c in CITY],
                             [A[c]["decomposition"]["seasonal_strength"] for c in CITY])
KEY["spearman_abs_change"] = [float(rho_change.statistic), float(rho_change.pvalue)]
KEY["spearman_seasonal_strength"] = [float(rho_season.statistic), float(rho_season.pvalue)]

# --------------------------------------------------------------------------- #
# 5. Derived figures (plots of saved results only)
# --------------------------------------------------------------------------- #
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "serif"})

# 5a. RMSE skill relative to persistence, city x model
show = [m for m in MODELS if m not in ("Persistence (t-1)", "Seasonal naive (t-7)",
                                         "Seasonal naive (t-365)", "Mean (climatology)")]
grid = np.array([[100 * (1 - B[c]["model_metrics"][m]["rmse"] / B[c]["model_metrics"]["Persistence (t-1)"]["rmse"])
                  for c in ranked] for m in show])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
lim = np.abs(grid).max()
im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(ranked)), ranked, rotation=30, ha="right")
ax.set_yticks(range(len(show)), [lab(m).replace("$", "").replace("--", "\u2013") for m in show])
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(grid[i, j]) > 0.6 * lim else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("RMSE reduction vs persistence (%)")
ax.set_title("One-step test RMSE relative to each city's persistence forecast")
fig.tight_layout()
fig.savefig(FIG / "rmse_skill_heatmap.png", dpi=220)
plt.close(fig)

# 5b. Forecast difficulty vs day-to-day variability
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
best_rmse = {c: min(v["rmse"] for v in B[c]["model_metrics"].values()) for c in CITY}
for ax, xkey, xlabel, rho in [
    (axes[0], lambda c: desc[c]["abs_change"], r"Mean $|\Delta y_t|$, 2010–2024 (°C)", rho_change),
    (axes[1], lambda c: A[c]["decomposition"]["seasonal_strength"], r"STL seasonal strength $F_S$", rho_season),
]:
    xs = [xkey(c) for c in CITY]
    ax.scatter(xs, [best_rmse[c] for c in CITY], color="#2b6c9e", s=22, zorder=3)
    left_panel = ax is axes[0]
    nudge = ({"Chennai": (-4, -11), "Mumbai": (4, 5), "Bengaluru": (5, -4)} if left_panel
             else {"Chennai": (4, 4), "Bengaluru": (-5, -9)})
    right_aligned = {"Delhi", "Chennai"} if left_panel else {"Delhi", "Bengaluru"}
    for c, x in zip(CITY, xs):
        ax.annotate(c, (x, best_rmse[c]), textcoords="offset points", xytext=nudge.get(c, (4, 3)),
                    fontsize=7, ha="right" if c in right_aligned else "left")
    ax.set_ylim(0.78, 1.38)
    if left_panel:
        ax.set_xlim(0.54, 1.17)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Spearman $\\rho$ = {rho.statistic:.2f} (p = {rho.pvalue:.3f})", fontsize=9)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Best one-step test RMSE (°C)")
fig.tight_layout()
fig.savefig(FIG / "difficulty_vs_variability.png", dpi=220)
plt.close(fig)

(SUB / "scripts" / "key_numbers.json").write_text(json.dumps(KEY, indent=2, default=float), encoding="utf-8")
print("figures:", len(list(FIG.glob("*.png"))), "tables:", len(list(TAB.glob("*.tex"))))
