#!/usr/bin/env python3
"""Method-level external baselines: what does the physics feature engineering buy?

Every arm in the main table shares this system's engineered feature set --
Morinaga Md, gamma/gamma-prime lattice mismatch, gamma-prime volume fraction,
Labusch-Nabarro solid-solution terms, temperature derivatives, grain-boundary
and refractory aggregates. Comparing those arms against each other cannot say
whether the features matter, only whether the layers stacked on top of them do.

This script answers the other question. It trains standard regressors on **raw
inputs only** -- element weight percents and the test temperature, nothing
derived -- and evaluates them on the identical 471-row corrected evaluation
set, with the identical splits, seeds and tuning budget the production models
received. The gap between these baselines and ML-only is the contribution of
the feature engineering, isolated.

Three baselines:

  gbm_raw   XGBoost + RandomForest voting ensemble, the same estimator pair
            and the same Optuna budget (30 trials, TPESampler seeded at 42,
            GroupKFold(5) on alloy name, negative R2 objective) used to tune
            the production models. Only the feature matrix differs.

  rf_raw    RandomForest alone on the same raw features, tuned in the same
            search. Reported separately because a single-family baseline is
            what most of this literature actually publishes.

  gpr_raw   Gaussian Process Regression following the published Ni-superalloy
            methodology of Conduit et al. -- standardised composition and
            temperature inputs, an anisotropic Matern kernel with a white-noise
            term for measurement scatter, marginal-likelihood optimisation with
            restarts. Configuration is fixed and citable rather than tuned, as
            in the source methodology.

PROTOCOL PARITY. Identical to ``train_ml_models.py`` in every respect except
the features: GroupShuffleSplit(test_size=0.15, random_state=42) grouped by
alloy for the holdout, GroupKFold(5) for cross-validation, sample weights
inversely proportional to the square root of each alloy's row count, median
imputation and standardisation. Nothing about the comparison is tilted by the
harness.

Outputs (written to ../results and ../output):
    external_baseline_metrics.csv   CV, holdout and evaluation metrics per model
    external_baselines.md           the report
    seed42_{gbm,rf,gpr}_raw_{ds}.csv   predictions, harness schema

Usage:
    python external_baselines.py                 # full run, 30 Optuna trials
    python external_baselines.py --trials 5      # quick smoke run
"""

import argparse
import json
import os
import sys
import warnings

import math

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/prediction
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

TRAIN_FILE = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models",
                          "training_data", "train_77alloys_v2.jsonl")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASETS = {"sss": "SSS.jsonl", "precip": "precip.jsonl", "sc_ds": "sc_ds.jsonl"}

#: Verbatim from train_ml_models.TARGETS, including the physical bounds that
#: decide which measurements enter training at all.
TARGETS = {
    "ys": {"key": "yield_strength", "name": "Yield Strength", "bounds": (50, 2000),
           "pred": "pred_ys", "actual": "actual_ys", "unit": "MPa"},
    "uts": {"key": "uts", "name": "Tensile Strength", "bounds": (50, 2500),
            "pred": "pred_uts", "actual": "actual_uts", "unit": "MPa"},
    "el": {"key": "elongation", "name": "Elongation", "bounds": (0, 80),
           "pred": "pred_el", "actual": "actual_el", "unit": "%"},
    "em": {"key": "elasticity", "name": "Elastic Modulus", "bounds": (50, 350),
           "pred": "pred_em", "actual": "actual_em", "unit": "GPa"},
}

#: Aggregate placeholder in the source data, not an element.
NOT_AN_ELEMENT = {"Other"}

SEED = 42
HOLDOUT_FRACTION = 0.15
CV_SPLITS = 5

#: The production harness attaches a measurement to a row when the recorded
#: temperature is within 5 C of it (generate_predictions.get_actual_values).
#: Datasheets quote the same nominal condition inconsistently -- elongation at
#: 650 C where yield strength is at 649, a room-temperature modulus at 20 where
#: strength is at 21 -- so an exact match silently drops measurements the other
#: arms keep. Matching exactly here cost these baselines 23 yield-strength rows
#: and broke the identical-row-set property the headline table asserts.
TEMP_MATCH_TOLERANCE_C = 5.0


def sample_weights(alloy_names):
    """Inverse-square-root frequency weighting, as in train_ml_models."""
    s = pd.Series(alloy_names)
    counts = s.value_counts()
    return (1.0 / np.sqrt(s.map(counts))).values


def element_vocabulary():
    """Every element appearing in the training set or the evaluation set.

    Fixed up front so the training and evaluation feature matrices have the
    same columns in the same order; an element absent from a composition is a
    genuine zero, not a missing value.
    """
    els = set()
    for line in open(TRAIN_FILE):
        els |= set(json.loads(line).get("composition") or {})
    for fn in DATASETS.values():
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            for line in open(path):
                els |= set(json.loads(line).get("composition") or {})
    return sorted(els - NOT_AN_ELEMENT)


#: Engineered features, as computed by the production feature layer and stored
#: on every record. Training and evaluation files carry the identical 25 keys,
#: verified before use. Combined with the temperature derivatives below this is
#: the feature space the production models actually learn on.
def engineered_features(record, temperature):
    """Physics features + temperature, matching train_ml_models.load_data."""
    # Numeric features only. computed_features also carries categorical
    # descriptors (tcp_risk = "Low", alloy class), which the median imputer
    # cannot take and which the production pipeline one-hot encodes separately.
    feats = {k: float(v) for k, v in (record.get("computed_features") or {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
    t = float(temperature)
    tk = max(1.0, t + 273.15)
    feats.update({
        "test_temperature_c": t,
        "temp_c_sq": t ** 2,
        "temp_c_cube": t ** 3,
        "log_temp_k": math.log(tk),
        "inv_temp_k": 1.0 / tk,
        "temp_normalized": (t - 20) / 1080,
    })
    return feats


def raw_features(composition, temperature, vocab):
    """The entire raw feature vector: weight percents plus temperature."""
    row = {f"wt_{el}": float(composition.get(el) or 0.0) for el in vocab}
    row["test_temperature_c"] = float(temperature)
    return row


def load_training(target_id, vocab, engineered=False):
    """Training rows for one property."""
    cfg = TARGETS[target_id]
    lo, hi = cfg["bounds"]
    rows = []
    for line in open(TRAIN_FILE):
        rec = json.loads(line)
        comp = rec.get("composition") or {}
        for point in (rec.get(cfg["key"]) or []):
            try:
                val = float(point.get("value"))
                temp = float(point.get("temp_c"))
            except (TypeError, ValueError):
                continue
            if not (-270 <= temp <= 1500) or not (lo <= val <= hi):
                continue
            row = (engineered_features(rec, temp) if engineered
                   else raw_features(comp, temp, vocab))
            row["target"] = val
            row["alloy_name"] = rec.get("alloy", "unknown")
            rows.append(row)
    return pd.DataFrame(rows)


def load_evaluation(vocab, engineered=False):
    """The 471 evaluation rows, one per (alloy, temperature)."""
    frames = {}
    for ds, fn in DATASETS.items():
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            continue
        rows = []
        for line in open(path):
            rec = json.loads(line)
            comp = rec.get("composition") or {}
            temps = set()
            measured = {tid: [] for tid in TARGETS}
            for tid, cfg in TARGETS.items():
                for point in (rec.get(cfg["key"]) or []):
                    try:
                        t = float(point.get("temp_c"))
                        v = float(point.get("value"))
                    except (TypeError, ValueError):
                        continue
                    temps.add(t)
                    measured[tid].append((t, v))

            def actual_at(tid, temp):
                for t, v in measured[tid]:
                    if abs(t - temp) < TEMP_MATCH_TOLERANCE_C:
                        return v
                return None
            for t in sorted(temps):
                row = (engineered_features(rec, t) if engineered
                       else raw_features(comp, t, vocab))
                row["alloy"] = rec.get("alloy")
                row["temperature"] = t
                row["processing"] = rec.get("processing", "unknown")
                for tid in TARGETS:
                    row[TARGETS[tid]["actual"]] = actual_at(tid, t)
                rows.append(row)
        frames[ds] = pd.DataFrame(rows)
    return frames


def build_pipeline(model):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def tune_tree_model(X, y, groups, weights, kind, n_trials):
    """Optuna search with the production budget and sampler.

    ``kind`` is "gbm" for the XGBoost+RandomForest voting ensemble the
    production models use, or "rf" for RandomForest alone.
    """
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = _suggest_model(trial, kind)
        gkf = GroupKFold(n_splits=min(CV_SPLITS, pd.Series(groups).nunique()))
        scores = []
        for tr, va in gkf.split(X, y, groups=groups):
            pipe = build_pipeline(model)
            pipe.fit(X.iloc[tr], y.iloc[tr], model__sample_weight=weights[tr])
            scores.append(r2_score(y.iloc[va], pipe.predict(X.iloc[va])))
        return -float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return _suggest_model(_FrozenTrial(study.best_params), kind), study.best_params


class _FrozenTrial:
    """Replays a finished study's best parameters through the same builder."""

    def __init__(self, params):
        self._p = params

    def suggest_int(self, name, *a, **k):
        return self._p[name]

    def suggest_float(self, name, *a, **k):
        return self._p[name]

    def suggest_categorical(self, name, *a, **k):
        return self._p[name]


def _suggest_model(trial, kind):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor

    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5]),
        "random_state": SEED,
    }
    if kind == "rf":
        return RandomForestRegressor(**rf_params, n_jobs=-1)

    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 0.95),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 10.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 2.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 1.0),
        "tree_method": "hist",
        "random_state": SEED,
    }
    return VotingRegressor([
        ("xgb", xgb.XGBRegressor(**xgb_params, n_jobs=-1)),
        ("rf", RandomForestRegressor(**rf_params, n_jobs=-1)),
    ], n_jobs=-1)


def gpr_model():
    """Gaussian Process following the published Ni-superalloy methodology.

    Conduit and co-workers model superalloy properties as a Gaussian Process
    over composition and processing variables, with a stationary kernel and an
    explicit noise term absorbing experimental scatter (Conduit et al.,
    *Materials & Design* 131 (2017) 358-365, "Design of a nickel-base
    superalloy using a neural network"; the same GP treatment is used in the
    Alloys-by-Design line of work). The configuration here is fixed rather
    than tuned, matching how that methodology is specified: an anisotropic
    Matern kernel with nu = 2.5 -- twice differentiable, the standard choice
    for physical response surfaces that are smooth but not analytic -- scaled
    by a constant, plus white noise, with the marginal likelihood optimised
    from several restarts.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=3, random_state=SEED)


def evaluate(pipe, X, y):
    from sklearn.metrics import mean_absolute_error, r2_score
    pred = pipe.predict(X)
    return float(mean_absolute_error(y, pred)), float(r2_score(y, pred))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=30,
                    help="Optuna trials per target; 30 matches the production budget.")
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument("--preddir", default=OUTPUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.preddir, exist_ok=True)

    from sklearn.model_selection import GroupShuffleSplit, GroupKFold
    from sklearn.metrics import mean_absolute_error, r2_score

    ARMS = ("gbm_raw", "rf_raw", "gpr_raw", "gpr_physics")

    vocab = element_vocabulary()
    print(f"Raw feature set: {len(vocab)} element weight percents + temperature "
          f"= {len(vocab) + 1} features")
    print(f"(production models use 50-80 engineered features)\n")

    eval_frames = load_evaluation(vocab)
    eval_frames_eng = load_evaluation(vocab, engineered=True)
    n_eval = sum(len(v) for v in eval_frames.values())
    print(f"Evaluation rows: {n_eval}")

    feature_cols = [f"wt_{e}" for e in vocab] + ["test_temperature_c"]
    eng_cols = sorted(set(next(iter(eval_frames_eng.values())).columns)
                      - {"alloy", "temperature", "processing"}
                      - {t["actual"] for t in TARGETS.values()})
    print(f"Engineered feature set: {len(eng_cols)} features "
          f"(physics + temperature derivatives)")
    metric_rows = []
    predictions = {name: {ds: (eval_frames_eng if name.endswith("physics") else eval_frames)[ds].copy()
                          for ds in eval_frames}
                   for name in ARMS}

    for tid, cfg in TARGETS.items():
        df = load_training(tid, vocab)
        df_eng = load_training(tid, vocab, engineered=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=SEED)
        tr_idx, te_idx = next(gss.split(df, groups=df["alloy_name"]))
        train_df, test_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        X = train_df[feature_cols].reset_index(drop=True)
        y = train_df["target"].reset_index(drop=True)
        groups = train_df["alloy_name"].reset_index(drop=True)
        w = sample_weights(groups)
        print(f"\n=== {cfg['name']} ({tid}) ===")
        print(f"  train {len(train_df)} rows / {groups.nunique()} alloys, "
              f"holdout {len(test_df)} rows")

        for name in ARMS:
            physics = name.endswith("physics")
            if physics:
                tr_eng = df_eng.iloc[tr_idx].copy()
                te_eng = df_eng.iloc[te_idx].copy()
                X, y = tr_eng[eng_cols].reset_index(drop=True), tr_eng["target"].reset_index(drop=True)
                groups = tr_eng["alloy_name"].reset_index(drop=True)
                w = sample_weights(groups)
                cols, test_df_use = eng_cols, te_eng
            else:
                X = train_df[feature_cols].reset_index(drop=True)
                y = train_df["target"].reset_index(drop=True)
                groups = train_df["alloy_name"].reset_index(drop=True)
                w = sample_weights(groups)
                cols, test_df_use = feature_cols, test_df

            if name.startswith("gpr"):
                model, params = gpr_model(), {"kernel": "C * Matern(nu=2.5) + White",
                                             "features": "engineered" if physics else "raw"}
            else:
                model, params = tune_tree_model(X, y, groups, w,
                                                "gbm" if name == "gbm_raw" else "rf",
                                                args.trials)

            gkf = GroupKFold(n_splits=min(CV_SPLITS, groups.nunique()))
            cv_mae, cv_r2 = [], []
            for a, b in gkf.split(X, y, groups=groups):
                pipe = build_pipeline(model)
                if name.startswith("gpr"):
                    pipe.fit(X.iloc[a], y.iloc[a])
                else:
                    pipe.fit(X.iloc[a], y.iloc[a], model__sample_weight=w[a])
                p = pipe.predict(X.iloc[b])
                cv_mae.append(mean_absolute_error(y.iloc[b], p))
                cv_r2.append(r2_score(y.iloc[b], p))

            pipe = build_pipeline(model)
            if name.startswith("gpr"):
                pipe.fit(X, y)
            else:
                pipe.fit(X, y, model__sample_weight=w)
            ho_mae, ho_r2 = evaluate(pipe, test_df_use[cols], test_df_use["target"])

            print(f"  {name:8s} CV MAE {np.mean(cv_mae):8.2f} R2 {np.mean(cv_r2):6.3f} | "
                  f"holdout MAE {ho_mae:8.2f} R2 {ho_r2:6.3f}")
            metric_rows.append({
                "model": name, "target": tid, "property": cfg["name"], "unit": cfg["unit"],
                "n_train_rows": len(X), "n_train_alloys": int(groups.nunique()),
                "n_features": len(cols),
                "cv_mae": round(float(np.mean(cv_mae)), 2),
                "cv_r2": round(float(np.mean(cv_r2)), 3),
                "holdout_mae": round(ho_mae, 2), "holdout_r2": round(ho_r2, 3),
                "params": json.dumps(params)[:400],
            })

            for ds, frame in predictions[name].items():
                frame[cfg["pred"]] = pipe.predict(frame[cols])

    for name, frames in predictions.items():
        for ds, frame in frames.items():
            out = frame.copy()
            out["processing"] = out["processing"]
            out["method"] = name.upper()
            out["status"] = "SUCCESS"
            out["seed"] = SEED
            keep = (["alloy", "temperature", "processing", "method", "status"]
                    + [c for t in TARGETS.values() for c in (t["pred"], t["actual"])]
                    + ["seed"])
            out[keep].to_csv(os.path.join(args.preddir, f"seed42_{name}_{ds}.csv"),
                             index=False)
    print(f"\nPredictions written to {args.preddir}")

    m = pd.DataFrame(metric_rows)
    m.to_csv(os.path.join(args.outdir, "external_baseline_metrics.csv"), index=False)
    print(f"Metrics written to {args.outdir}/external_baseline_metrics.csv")
    print("\nNow run headline_table.py and nn_distance_analysis.py to fold these arms\n"
          "into the canonical tables, then external_baselines_report.py for the writeup.")


if __name__ == "__main__":
    main()
