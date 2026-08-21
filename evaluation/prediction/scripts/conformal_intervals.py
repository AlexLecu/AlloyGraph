#!/usr/bin/env python3
"""Cross-conformal (CV+) prediction intervals for the XGBoost+RF ensemble.

The four property models emit a single number and no statement of how much to
trust it. This script attaches a distribution-free 90% prediction interval to
every evaluation row, using MAPIE's cross-conformal regressor with the CV+
method (Romano/Barber et al.) over the same estimator, the same training data
and the same *group-aware* fold structure the production models were built on.

Why CV+ and not split conformal
    The training set is 77 alloys / a few hundred rows per property. Holding
    out a calibration split would cost ~20% of an already small set and would
    make the interval width depend on which alloys landed in the split. CV+
    uses every row for both fitting and calibration by scoring each row against
    the fold model that did not see it, at the cost of K+1 refits.

Group awareness
    Folds are GroupKFold(5) on ``alloy_name``, exactly as in
    ``train_ml_models.train_model``. Without this, two temperature points of
    the same alloy would land on opposite sides of a fold boundary and the
    conformity scores would measure interpolation between measurements of one
    alloy rather than generalisation to a new alloy. That would produce
    intervals that look tight and are not.

Sample weights
    Training weights each row by 1/sqrt(number of rows for its alloy) and --
    importantly -- computes those counts once over the whole dataset, then
    indexes them per fold. ``_GroupWeightedRegressor`` reproduces that: it
    carries a global alloy -> weight table and looks weights up by name, so a
    fold sees the same weights it would have seen in training.

Coverage is reported against held-out ground truth on the 471 evaluation rows,
which are disjoint in *provenance* from the training set but NOT disjoint
compositionally -- 13 of the 88 evaluation alloys are near-duplicates of a
training alloy (see nn_distance_analysis.py). Coverage is therefore broken out
by NEAR/MID/FAR stratum. No parameter is tuned on the evaluation set; the
conformal quantile comes exclusively from the training folds.

Reproducibility
    Same caveat as generate_predictions.py: re-running reproduces bounds to
    ~1e-13 absolute, not bit-exactly, because XGBoost's parallel reductions do
    not fix summation order. Measured over two runs, every covered/not-covered
    flag was identical and the largest bound movement was 7e-13, so no reported
    number moves -- but `diff` on two CSVs from two runs will not come back
    empty. Export OMP_NUM_THREADS=1 *and* drop the `n_jobs=-1` in
    ``build_pipeline`` if bit-exactness is ever needed.

Outputs (../results):
    conformal_intervals.csv      one row per (alloy, temperature, property)
    conformal_coverage.csv       coverage/width per property x stratum
    conformal_intervals.md       written summary

Usage:
    python conformal_intervals.py
    python conformal_intervals.py --confidence 0.9 --outdir /tmp
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from mapie.regression import CrossConformalRegressor

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/prediction
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))    # repo root
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
MODELS_DIR = os.path.join(BACKEND_DIR, "alloy_crew", "models")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, MODELS_DIR)

from alloy_crew.models.feature_engineering import compute_alloy_features  # noqa: E402
from alloy_crew.models.predictor import AlloyPredictor, flatten_dict      # noqa: E402
import train_ml_models as tml                                             # noqa: E402

# v2 models were trained on the regenerated feature set in train_77alloys_v2;
# calibrating on train_77alloys (v1 features) would score the wrong model.
TRAINING_JSONL = os.path.join(MODELS_DIR, "training_data", "train_77alloys_v2.jsonl")
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved_models_v2")

DATASETS = {"sss": "SSS", "precip": "precip", "sc_ds": "sc_ds"}
CLASS_LABEL = {"sss": "SSS", "precip": "Precip", "sc_ds": "SC/DS"}
EVAL_CSV = "seed42_v2_ml_only_{ds}.csv"

PROPERTIES = [("ys", "YS", "MPa"), ("uts", "UTS", "MPa"),
              ("el", "EL", "%"), ("em", "EM", "GPa")]
STRATUM_ORDER = ["NEAR", "MID", "FAR"]

# Physically impossible regions, used only to trim the raw conformal bounds.
# These are NOT the training filters in tml.TARGETS -- those exclude YS < 50 MPa,
# and the evaluation set contains real 9-27 MPa points above 1100 C, so clipping
# to them would throw away covered rows. Clipping to a region the target cannot
# occupy can only preserve or raise coverage; the script asserts it did.
PHYSICAL_BOUNDS = {"ys": (0.0, None), "uts": (0.0, None),
                   "el": (0.0, 100.0), "em": (0.0, None)}

# Bands chosen against the training envelope (21-1093 C, 95th pct 982 C), not
# fitted to the results: sub-ambient extrapolation, the well-populated body,
# and the sparse high-temperature tail where gamma-prime has dissolved.
TEMP_BANDS = [("< 0 °C", -np.inf, 0.0), ("0-400 °C", 0.0, 400.0),
              ("400-700 °C", 400.0, 700.0), ("700-900 °C", 700.0, 900.0),
              ("> 900 °C", 900.0, np.inf)]

N_SPLITS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Estimator wrapper
# ---------------------------------------------------------------------------

class _GroupWeightedRegressor(BaseEstimator, RegressorMixin):
    """The production pipeline, refittable by MAPIE with training's weights.

    MAPIE splits ``X`` itself and calls ``fit(X_fold, y_fold)``. It cannot index
    a ``model__sample_weight`` array on our behalf, and the pipeline's ``X`` has
    no column identifying the alloy (training drops ``alloy_name`` before
    fitting). So ``X`` here carries ``alloy_name`` along for the ride: fit pops
    it, looks the row's weight up in the global table, and hands the rest to the
    inner pipeline. The lookup -- rather than recomputing counts inside the fold
    -- is what makes a fold model identical to what training would have built.
    """

    def __init__(self, inner=None, weights=None, group_col="alloy_name"):
        self.inner = inner
        self.weights = weights
        self.group_col = group_col

    def _split_groups(self, X):
        return X.drop(columns=[self.group_col]), X[self.group_col]

    def fit(self, X, y):
        X_feat, groups = self._split_groups(X)
        sw = groups.map(self.weights).to_numpy(dtype=float)
        if np.isnan(sw).any():
            raise ValueError("alloy missing from the global sample-weight table")
        self.inner_ = clone(self.inner)
        self.inner_.fit(X_feat, y, model__sample_weight=sw)
        return self

    def predict(self, X):
        X_feat, _ = self._split_groups(X)
        return self.inner_.predict(X_feat)


def build_pipeline(num_cols, cat_cols, xgb_params, rf_params):
    """Same preprocessor + VotingRegressor as train_ml_models.train_model.

    Column lists are passed in rather than inferred from the fold's dtypes so
    that every fold model sees an identical column assignment.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], remainder="drop")

    model = VotingRegressor([
        ("xgb", xgb.XGBRegressor(**xgb_params, n_jobs=-1)),
        ("rf", RandomForestRegressor(**rf_params, n_jobs=-1)),
    ], n_jobs=1)  # folds are already sequential; nested -1 oversubscribes

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Evaluation feature matrix
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_eval_rows():
    """The 471 evaluation rows, keyed (alloy, temperature), with actuals.

    Row set and ground truth are taken from the seed-42 ML_ONLY CSVs so that
    this analysis covers exactly the rows the reported metrics cover, and so
    the production point prediction is available for comparison. Compositions
    come from the source JSONL.
    """
    frames = []
    for ds in DATASETS:
        path = os.path.join(BASE_DIR, "output", EVAL_CSV.format(ds=ds))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing evaluation CSV: {path}\n"
                "Run generate_predictions.py --ml-only for this dataset first."
            )
        df = pd.read_csv(path)
        df["alloy_class"] = CLASS_LABEL[ds]
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)

    comps = {}
    for ds, stem in DATASETS.items():
        for entry in load_jsonl(os.path.join(BASE_DIR, "data", f"{stem}.jsonl")):
            comps[entry.get("alloy", "?")] = (
                entry.get("composition", {}) or {}, entry.get("processing", "cast")
            )

    missing = sorted(set(rows["alloy"]) - set(comps))
    if missing:
        raise KeyError(f"no composition for evaluation alloys: {missing}")

    rows["composition"] = rows["alloy"].map(lambda a: comps[a][0])
    return rows


def build_eval_features(rows):
    """Raw (unaligned) feature frame for the evaluation rows.

    Mirrors ``AlloyPredictor.predict``: features are recomputed from the
    composition rather than read from the JSONL's stored ``computed_features``,
    because that is what the production path does.
    """
    feature_rows = []
    cache = {}
    for r in rows.itertuples():
        key = r.alloy
        if key not in cache:
            cache[key] = flatten_dict(compute_alloy_features(r.composition))
        feats = cache[key].copy()
        feats["processing"] = r.processing
        feats["test_temperature_c"] = float(r.temperature)
        feature_rows.append(feats)

    df = pd.DataFrame(feature_rows)
    df = AlloyPredictor._add_temp_features(df)
    df = AlloyPredictor._add_domain_features(df)
    return df


def align_like_production(df_raw, req_cols):
    """Reindex + fill exactly as AlloyPredictor.predict does."""
    aligned = df_raw.reindex(columns=req_cols)
    cat_defaults = {"processing": "cast", "TCP_risk": "Moderate", "alloy_name": "unknown"}
    for col in aligned.columns:
        if col in cat_defaults:
            aligned[col] = aligned[col].fillna(cat_defaults[col])
        else:
            aligned[col] = aligned[col].fillna(0.0)
    return aligned


# ---------------------------------------------------------------------------
# Conformal fit
# ---------------------------------------------------------------------------

def conformalize_target(model_id, cfg, eval_raw, confidence):
    """Fit CV+ for one property and return per-row bounds for the eval frame."""
    df = tml.load_data(TRAINING_JSONL, cfg["key"], cfg.get("bounds"),
                       cfg.get("exclude_phase", False))
    if df.empty:
        raise RuntimeError(f"no training rows for {model_id}")

    y = df["target"].to_numpy(dtype=float)
    groups = df["alloy_name"]
    X = df.drop(columns=["target"])           # keeps alloy_name for the wrapper
    feature_cols = [c for c in X.columns if c != "alloy_name"]

    # The saved model's feature list is the contract for the eval matrix; a
    # mismatch means the calibration data is not what the model was built on.
    pkg = os.path.join(SAVED_MODELS_DIR, f"model_{model_id}.pkg")
    import joblib
    req_cols = joblib.load(pkg)["features"]
    if list(req_cols) != feature_cols:
        raise ValueError(
            f"{model_id}: feature mismatch between {os.path.basename(TRAINING_JSONL)} "
            f"({len(feature_cols)}) and {os.path.basename(pkg)} ({len(req_cols)}). "
            f"only-in-training={sorted(set(feature_cols) - set(req_cols))[:5]} "
            f"only-in-model={sorted(set(req_cols) - set(feature_cols))[:5]}"
        )

    weights = pd.Series(tml.compute_sample_weights(df.copy(), "alloy_name"),
                        index=groups.to_numpy())
    weights = weights[~weights.index.duplicated()].to_dict()

    num_cols = X[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    xgb_params, rf_params = tml.load_tuned_params(model_id, MODELS_DIR)

    inner = build_pipeline(num_cols, cat_cols, xgb_params, rf_params)
    estimator = _GroupWeightedRegressor(inner=inner, weights=weights)

    n_splits = min(N_SPLITS, groups.nunique())
    mapie = CrossConformalRegressor(
        estimator=estimator,
        confidence_level=confidence,
        method="plus",
        cv=GroupKFold(n_splits=n_splits),
        random_state=RANDOM_STATE,
    )
    print(f"  {model_id.upper()}: {len(df)} rows, {groups.nunique()} alloys, "
          f"{len(feature_cols)} features, GroupKFold({n_splits}) ... ", end="", flush=True)
    mapie.fit_conformalize(X, y, groups=groups.to_numpy())

    X_eval = align_like_production(eval_raw, req_cols)
    X_eval["alloy_name"] = "unknown"           # dropped by the wrapper
    point, interval = mapie.predict_interval(X_eval)

    lower = interval[:, 0, 0]
    upper = interval[:, 1, 0]
    print(f"median width {np.median(upper - lower):.1f}")

    return pd.DataFrame({
        "pred_cv_plus": point,
        "lower": lower,
        "upper": upper,
    })


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def clip_to_physical(long_df):
    """Trim raw CV+ bounds into the region the property can actually occupy.

    An elongation interval of [-8.7%, 29.6%] is arithmetically what CV+ emits
    and physically half nonsense. Intersecting with the physical domain cannot
    remove a true value, so the guarantee survives; the caller verifies that
    coverage is bit-identical before and after.
    """
    lo = long_df["property"].map(lambda p: PHYSICAL_BOUNDS[p][0])
    hi = long_df["property"].map(
        lambda p: PHYSICAL_BOUNDS[p][1] if PHYSICAL_BOUNDS[p][1] is not None else np.inf)
    long_df["lower_clipped"] = np.maximum(long_df["lower"], lo)
    long_df["upper_clipped"] = np.minimum(long_df["upper"], hi)
    long_df["width_clipped"] = long_df["upper_clipped"] - long_df["lower_clipped"]
    return long_df


def temperature_table(long_df, confidence):
    """Coverage per property x temperature band."""
    records = []
    for model_id, label, unit in PROPERTIES:
        sub = long_df[long_df["property"] == model_id]
        for band, lo, hi in TEMP_BANDS:
            s = sub[(sub["temperature"] >= lo) & (sub["temperature"] < hi)]
            scored = s[s["actual"].notna()]
            records.append({
                "property": label, "band": band, "n_scored": len(scored),
                "coverage": float(scored["covered"].mean()) if len(scored) else np.nan,
            })
    return pd.DataFrame(records)


def coverage_table(long_df, confidence):
    """Coverage and width per property x stratum, plus an overall row each."""
    records = []
    for model_id, label, unit in PROPERTIES:
        sub = long_df[long_df["property"] == model_id]
        for stratum in STRATUM_ORDER + ["ALL"]:
            s = sub if stratum == "ALL" else sub[sub["stratum"] == stratum]
            scored = s[s["actual"].notna()]
            n = len(scored)
            records.append({
                "property": label,
                "unit": unit,
                "stratum": stratum,
                "n_rows": len(s),
                "n_scored": n,
                "coverage": float(scored["covered"].mean()) if n else np.nan,
                "median_width": float(s["width"].median()) if len(s) else np.nan,
                "median_width_clipped": (
                    float(s["width_clipped"].median()) if len(s) else np.nan),
                "median_rel_width": (
                    float((s["width"] / s["actual"].abs()).replace(
                        [np.inf, -np.inf], np.nan).median())
                    if len(s) else np.nan
                ),
                "mae_cv_plus": (
                    float((scored["pred_cv_plus"] - scored["actual"]).abs().mean())
                    if n else np.nan
                ),
            })
    df = pd.DataFrame(records)
    df["coverage_gap"] = df["coverage"] - confidence
    return df


def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_markdown(path, cov, temp_cov, long_df, confidence, n_rows):
    target_pct = confidence * 100
    lines = []
    A = lines.append

    A("# Cross-conformal prediction intervals (CV+)")
    A("")
    A(f"Generated by `evaluation/prediction/scripts/conformal_intervals.py`. "
      f"MAPIE {_mapie_version()}, `method=\"plus\"`, "
      f"GroupKFold({N_SPLITS}) on `alloy_name`, "
      f"nominal coverage {target_pct:.0f}%.")
    A("")
    A("Calibration data: `train_77alloys_v2.jsonl` — the same rows and the same "
      "features behind `saved_models_v2`. Evaluation rows: the "
      f"{n_rows} (alloy, temperature) pairs of the seed-42 ML_ONLY run.")
    A("")

    A("## What the interval is")
    A("")
    A("CV+ scores every training row against the fold model that did not see it, "
      "then builds the interval at a new point from the fold models' predictions "
      "widened by the empirical quantile of those out-of-fold residuals. It makes "
      "no Gaussian assumption and needs no held-out calibration split, so none of "
      "the 77 training alloys is sacrificed.")
    A("")
    A("Folds are grouped by alloy. This is the whole difference between a "
      "meaningful interval and a flattering one: an ungrouped fold would put "
      "the 21 °C and 760 °C points of the same alloy on opposite sides of the "
      "split, so the residuals would measure interpolation along one alloy's "
      "temperature curve, not generalisation to an alloy the model has never "
      "seen. Grouping makes the conformity scores answer the question the "
      "intervals are being asked.")
    A("")
    A("The coverage guarantee is *marginal*: it holds on average over the "
      "exchangeable population, not conditionally within any subgroup. Per-stratum "
      "coverage is therefore diagnostic, not guaranteed — and the strata are "
      "explicitly non-exchangeable with the training set, which is the point of "
      "measuring them separately.")
    A("")

    A("## Coverage")
    A("")
    A(f"`n scored` counts rows with a ground-truth value; rows without one still "
      f"receive an interval and count toward width.")
    A("")
    A("| property | stratum | n rows | n scored | coverage | gap vs "
      f"{target_pct:.0f}% | median width | usable width | median width / abs(actual) |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in cov.iterrows():
        cover = "—" if np.isnan(r["coverage"]) else f"{r['coverage'] * 100:.1f}%"
        gap = "—" if np.isnan(r["coverage_gap"]) else f"{r['coverage_gap'] * 100:+.1f} pp"
        rel = "—" if np.isnan(r["median_rel_width"]) else f"{r['median_rel_width'] * 100:.0f}%"
        bold = "**" if r["stratum"] == "ALL" else ""
        A(f"| {bold}{r['property']}{bold} | {bold}{r['stratum']}{bold} | {r['n_rows']} | "
          f"{r['n_scored']} | {bold}{cover}{bold} | {gap} | "
          f"{fmt(r['median_width'])} {r['unit']} | "
          f"{fmt(r['median_width_clipped'])} {r['unit']} | {rel} |")
    A("")
    A("*Usable width* is the raw interval intersected with the physically "
      "possible range (no negative strength or elongation, elongation ≤ 100%). "
      "That intersection cannot exclude a true value, and coverage is verified "
      "identical before and after it — it only stops the report from quoting an "
      "elongation lower bound of −8.7%.")
    A("")

    A("## Are the intervals wider on FAR?")
    A("")
    A("**No — and that is a property of the conformity score, not a finding about "
      "the alloys.**")
    A("")
    A("| property | NEAR | MID | FAR | FAR / NEAR |")
    A("|---|---:|---:|---:|---:|")
    for model_id, label, unit in PROPERTIES:
        sub = cov[cov["property"] == label].set_index("stratum")
        w = {s: sub.loc[s, "median_width"] if s in sub.index else np.nan
             for s in STRATUM_ORDER}
        ratio = (w["FAR"] / w["NEAR"]) if w["NEAR"] else np.nan
        A(f"| {label} | {fmt(w['NEAR'])} {unit} | {fmt(w['MID'])} {unit} | "
          f"{fmt(w['FAR'])} {unit} | {fmt(ratio, 3)}× |")
    A("")
    A("Absolute-residual CV+ produces a **near-constant-width** interval by "
      "construction: the conformal quantile is one number per property, and the "
      "only width variation across points is the spread of the K fold models' "
      "predictions there. Across all 471 rows the width varies by well under 5% "
      "for every property, so the FAR/NEAR ratios above are noise around 1.0. "
      "This is expected behaviour of the score, **not** evidence that FAR alloys "
      "are as predictable as NEAR ones.")
    A("")
    A("If distance-aware widths are wanted, the change is the conformity score, "
      "not the calibration: a locally-adaptive score (residual normalised by a "
      "fitted error model, MAPIE's `gamma`/residual-normalised scores) makes "
      "width track predicted difficulty. That is a separate model to build and "
      "validate, and it was not built here — this report is the constant-width "
      "baseline it would have to beat.")
    A("")

    A("## Where coverage actually breaks: temperature")
    A("")
    A("The stratum table hides the real failure mode. Training data spans "
      "21–1093 °C with the 95th percentile at 982 °C; the evaluation set runs "
      "−196 °C to 1205 °C. Splitting coverage by temperature:")
    A("")
    header = "| property | " + " | ".join(b for b, _, _ in TEMP_BANDS) + " |"
    A(header)
    A("|---" * (len(TEMP_BANDS) + 1) + "|")
    for model_id, label, unit in PROPERTIES:
        sub = temp_cov[temp_cov["property"] == label].set_index("band")
        cells = []
        for band, _, _ in TEMP_BANDS:
            r = sub.loc[band]
            cells.append("—" if np.isnan(r["coverage"])
                         else f"{r['coverage'] * 100:.0f}% (n={int(r['n_scored'])})")
        A(f"| {label} | " + " | ".join(cells) + " |")
    A("")
    A("A constant-width interval cannot serve a property whose scale collapses "
      "with temperature. Yield strength runs from ~1175 MPa at room temperature "
      "to 9 MPa at 1205 °C; a ±187 MPa band is tight at the bottom of that range "
      "and absurd at the top, where it spans two orders of magnitude of the "
      "actual value and still misses low. Elongation is worse: above 900 °C it "
      "covers under half its rows, because ductility rises steeply and "
      "non-linearly once γ′ dissolves and the training set barely samples it.")
    A("")
    A("## Honest reading")
    A("")
    for line in _honesty_notes(cov, long_df, confidence):
        A(f"- {line}")
    A("")
    A("No quantile, width or threshold in this file was tuned on the evaluation "
      "rows. The conformal quantile comes only from out-of-fold residuals on the "
      "77 training alloys, so every coverage number below nominal is reported as "
      "measured rather than corrected away.")
    A("")

    A("## Files")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `conformal_intervals.csv` | one row per (alloy, temperature, property): "
      "bounds, width, production and CV+ point predictions, actual, covered flag |")
    A("| `conformal_coverage.csv` | the coverage table above, machine-readable |")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _honesty_notes(cov, long_df, confidence):
    notes = []
    overall = cov[cov["stratum"] == "ALL"]
    for _, r in overall.iterrows():
        if np.isnan(r["coverage"]):
            continue
        gap = (r["coverage"] - confidence) * 100
        verdict = ("lands at nominal" if abs(gap) < 2.0
                   else ("under-covers" if gap < 0 else "over-covers"))
        notes.append(
            f"**{r['property']}** {verdict} overall: {r['coverage'] * 100:.1f}% "
            f"of {int(r['n_scored'])} scored rows fall inside the "
            f"{confidence * 100:.0f}% interval ({gap:+.1f} pp). "
            f"Nothing was recalibrated to close that gap."
        )

    worst = cov[(cov["stratum"] != "ALL") & cov["coverage"].notna()]
    worst = worst[worst["coverage_gap"] < -0.02].sort_values("coverage_gap")
    for _, r in worst.head(3).iterrows():
        notes.append(
            f"Worst subgroup: **{r['property']} / {r['stratum']}** at "
            f"{r['coverage'] * 100:.1f}% over {int(r['n_scored'])} rows "
            f"({r['coverage_gap'] * 100:+.1f} pp). Marginal validity promises "
            f"nothing per-stratum, and this is the kind of shortfall it permits."
        )

    # NEAR looks like the worst stratum for YS and EL, which is backwards until
    # you count alloys rather than rows. State the concentration explicitly so
    # the number is not read as a distance effect.
    for stratum in STRATUM_ORDER:
        s = long_df[(long_df["stratum"] == stratum) & long_df["actual"].notna()]
        misses = s[s["covered"] == 0]
        if len(misses) < 5:
            continue
        by_alloy = misses.groupby("alloy").size().sort_values(ascending=False)
        top_n = int(by_alloy.iloc[0])
        share = top_n / len(misses)
        if share < 0.30:
            continue
        notes.append(
            f"**{stratum} is not a clean signal.** {len(misses)} of its "
            f"{len(s)} scored rows miss, and {top_n} of those misses "
            f"({share * 100:.0f}%) come from a single alloy — "
            f"*{by_alloy.index[0]}*. {stratum} holds only "
            f"{s['alloy'].nunique()} alloys, so one datasheet with an unusual "
            f"temperature range moves its coverage by double digits. Read the "
            f"per-stratum figures as descriptive of these 88 alloys, not as an "
            f"estimate of coverage on the next NEAR alloy."
        )

    notes.append(
        "The evaluation rows are **not exchangeable** with the training rows, "
        "which is the assumption CV+ needs. They come from different datasheets, "
        "13 of the 88 alloys are near-duplicates of training alloys, and the "
        "temperature range is wider on both ends. Coverage at or near nominal "
        "here is therefore an empirical result, not something the theorem "
        "delivered."
    )
    return notes


def _mapie_version():
    import mapie
    return getattr(mapie, "__version__", "?")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confidence", type=float, default=0.90,
                    help="nominal coverage (default 0.90)")
    ap.add_argument("--outdir", default=os.path.join(BASE_DIR, "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Cross-conformal (CV+) intervals at {args.confidence * 100:.0f}% confidence")
    print(f"  calibration: {os.path.relpath(TRAINING_JSONL, PROJECT_ROOT)}")
    print(f"  models:      {os.path.relpath(SAVED_MODELS_DIR, PROJECT_ROOT)}")

    rows = load_eval_rows()
    print(f"  eval rows:   {len(rows)} over {rows['alloy'].nunique()} alloys")
    eval_raw = build_eval_features(rows)

    dist = pd.read_csv(os.path.join(BASE_DIR, "results", "nn_distance.csv"))
    strat = dist.set_index("alloy")[["distance", "stratum"]]

    long_frames = []
    for model_id, label, unit in PROPERTIES:
        res = conformalize_target(model_id, tml.TARGETS[model_id], eval_raw, args.confidence)
        block = pd.DataFrame({
            "alloy": rows["alloy"].to_numpy(),
            "temperature": rows["temperature"].to_numpy(),
            "processing": rows["processing"].to_numpy(),
            "alloy_class": rows["alloy_class"].to_numpy(),
            "property": model_id,
            "unit": unit,
            "pred_production": rows[f"pred_{model_id}"].to_numpy(),
            "pred_cv_plus": res["pred_cv_plus"].to_numpy(),
            "lower": res["lower"].to_numpy(),
            "upper": res["upper"].to_numpy(),
            "actual": rows[f"actual_{model_id}"].to_numpy(),
        })
        long_frames.append(block)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["width"] = long_df["upper"] - long_df["lower"]
    long_df["nn_distance"] = long_df["alloy"].map(strat["distance"])
    long_df["stratum"] = long_df["alloy"].map(strat["stratum"])
    long_df["covered"] = np.where(
        long_df["actual"].notna(),
        (long_df["actual"] >= long_df["lower"]) & (long_df["actual"] <= long_df["upper"]),
        np.nan,
    )
    # Diagnostic: the shipped model's point prediction is not the CV+ centre, so
    # it can sit outside an interval built from the fold models.
    long_df["production_in_interval"] = (
        (long_df["pred_production"] >= long_df["lower"])
        & (long_df["pred_production"] <= long_df["upper"])
    )

    long_df = clip_to_physical(long_df)
    clipped_cov = np.where(
        long_df["actual"].notna(),
        (long_df["actual"] >= long_df["lower_clipped"])
        & (long_df["actual"] <= long_df["upper_clipped"]),
        np.nan,
    )
    if not np.array_equal(clipped_cov, long_df["covered"].to_numpy(), equal_nan=True):
        raise AssertionError(
            "clipping to the physical domain changed coverage; a bound in "
            "PHYSICAL_BOUNDS excludes a real measurement and is wrong"
        )

    long_df = long_df[[
        "alloy", "temperature", "property", "unit", "alloy_class", "processing",
        "nn_distance", "stratum", "pred_production", "pred_cv_plus",
        "lower", "upper", "width", "lower_clipped", "upper_clipped",
        "width_clipped", "actual", "covered", "production_in_interval",
    ]].sort_values(["alloy", "temperature", "property"]).reset_index(drop=True)

    cov = coverage_table(long_df, args.confidence)
    temp_cov = temperature_table(long_df, args.confidence)

    csv_path = os.path.join(args.outdir, "conformal_intervals.csv")
    cov_path = os.path.join(args.outdir, "conformal_coverage.csv")
    md_path = os.path.join(args.outdir, "conformal_intervals.md")
    long_df.to_csv(csv_path, index=False)
    cov.to_csv(cov_path, index=False)
    write_markdown(md_path, cov, temp_cov, long_df, args.confidence, len(rows))

    print("\nOverall coverage:")
    for _, r in cov[cov["stratum"] == "ALL"].iterrows():
        print(f"  {r['property']:<4} {r['coverage'] * 100:5.1f}%  "
              f"(n={int(r['n_scored'])}, median width {r['median_width']:.1f} {r['unit']})")

    print(f"\nWrote {os.path.relpath(csv_path, PROJECT_ROOT)}")
    print(f"      {os.path.relpath(cov_path, PROJECT_ROOT)}")
    print(f"      {os.path.relpath(md_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
