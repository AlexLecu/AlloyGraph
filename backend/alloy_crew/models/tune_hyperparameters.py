import json
import os
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')

from train_ml_models import load_data, compute_sample_weights


def objective(trial, df, use_ensemble=True):
    """Optuna objective: returns negative R² (minimized)."""
    X = df.drop(columns=['target', 'alloy_name'])
    y = df['target']
    groups = df['alloy_name']
    sample_weights = compute_sample_weights(df.copy(), 'alloy_name')

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ], remainder='drop')

    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 0.95),
        'colsample_bylevel': trial.suggest_float('xgb_colsample_bylevel', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 2.0),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 1.0),
        'tree_method': 'hist',
        'random_state': 42
    }

    if use_ensemble:
        # RandomForest hyperparameters
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 600, step=100),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 8),
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        model = VotingRegressor([
            ('xgb', xgb.XGBRegressor(**xgb_params, n_jobs=-1)),
            ('rf', RandomForestRegressor(**rf_params))
        ], n_jobs=-1)
    else:
        model = xgb.XGBRegressor(**xgb_params, n_jobs=-1)

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    # Cross-validation
    n_splits = min(5, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores, mae_scores = [], []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr, model__sample_weight=sample_weights[train_idx])
        preds = pipeline.predict(X_val)

        r2_scores.append(r2_score(y_val, preds))
        mae_scores.append(mean_absolute_error(y_val, preds))

    trial.set_user_attr('mae', float(np.mean(mae_scores)))
    return -np.mean(r2_scores)  # Optuna minimizes


def tune_target(target_key: str, target_name: str, data_file: str,
                bounds: tuple = None, exclude_phase: bool = False,
                n_trials: int = 50, use_ensemble: bool = True) -> dict:
    """Tune hyperparameters for a target property."""
    print(f"\n{'='*60}\nTUNING: {target_name}\n{'='*60}")

    df = load_data(data_file, target_key, bounds, exclude_phase)
    if df.empty or len(df) < 50:
        print(f"  Insufficient data: {len(df)} samples")
        return None

    print(f"  Data: {len(df)} samples, {df['alloy_name'].nunique()} alloys")

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, df, use_ensemble),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )

    best = study.best_trial
    print(f"\n  Best R²: {-best.value:.4f}, MAE: {best.user_attrs.get('mae', 'N/A'):.2f}")

    # Extract XGBoost params
    xgb_params = {k.replace('xgb_', ''): v for k, v in best.params.items() if k.startswith('xgb_')}
    xgb_params['tree_method'] = 'hist'
    xgb_params['random_state'] = 42

    # Extract RF params if ensemble
    rf_params = None
    if use_ensemble:
        rf_params = {k.replace('rf_', ''): v for k, v in best.params.items() if k.startswith('rf_')}
        rf_params['max_features'] = 'sqrt'
        rf_params['bootstrap'] = True
        rf_params['random_state'] = 42

    return {
        'target_key': target_key,
        'target_name': target_name,
        'best_r2': float(-best.value),
        'best_mae': float(best.user_attrs.get('mae', 0)),
        'xgb_params': xgb_params,
        'rf_params': rf_params,
        'exclude_phase_compositions': exclude_phase,
        'bounds': list(bounds) if bounds else None
    }


TARGETS = {
    "ys": {"key": "yield_strength", "name": "Yield Strength", "bounds": (50, 2000), "exclude_phase": False},
    "uts": {"key": "uts", "name": "Ultimate Tensile Strength", "bounds": (50, 2500), "exclude_phase": False},
    "el": {"key": "elongation", "name": "Elongation", "bounds": (0, 80), "exclude_phase": True},
    "em": {"key": "elasticity", "name": "Elastic Modulus", "bounds": (50, 350), "exclude_phase": True},
}


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(current_dir, "training_data", "final_alloy_data_enriched.jsonl")

    # Check for v2 data file
    v2_file = os.path.join(current_dir, "training_data", "final_alloy_data_enriched_v2.jsonl")
    if os.path.exists(v2_file):
        DATA_FILE = v2_file

    N_TRIALS = 30  # Increase for better results (50-100 recommended)

    print(f"\nHYPERPARAMETER TUNING\nData: {DATA_FILE}\nTrials per target: {N_TRIALS}\n")

    for model_id, cfg in TARGETS.items():
        result = tune_target(
            cfg["key"], cfg["name"], DATA_FILE,
            bounds=cfg.get("bounds"),
            exclude_phase=cfg.get("exclude_phase", False),
            n_trials=N_TRIALS,
            use_ensemble=True
        )

        if result:
            params_dir = os.path.join(current_dir, "tuned_params")
            os.makedirs(params_dir, exist_ok=True)
            output_file = os.path.join(params_dir, f"{model_id}.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {output_file}")

    print("\nTuning complete. Run train_ml_models.py to train with tuned parameters.")
