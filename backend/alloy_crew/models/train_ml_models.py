import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)


def safe_float(value, default=None):
    """Safely convert to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def flatten_dict(d, parent_key='', sep='_', exclude_keys=None):
    """Flatten nested dict, optionally excluding keys."""
    exclude_keys = exclude_keys or set()
    items = []
    for k, v in d.items():
        if k in exclude_keys:
            continue
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep, exclude_keys=exclude_keys).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_temp_features(df, temp_col='test_temperature_c'):
    """Add temperature-based features for Arrhenius-type behavior."""
    if temp_col not in df.columns:
        return df

    df[temp_col] = df[temp_col].clip(lower=-270, upper=1500)
    t_kelvin = (df[temp_col] + 273.15).clip(lower=1.0)

    df['temp_c_sq'] = df[temp_col] ** 2
    df['temp_c_cube'] = df[temp_col] ** 3
    df['log_temp_k'] = np.log(t_kelvin)
    df['inv_temp_k'] = 1.0 / t_kelvin
    df['temp_normalized'] = (df[temp_col] - 20) / 1080  # Range: 20-1100C

    return df


def add_domain_features(df):
    """Add physics-based features for Ni superalloys."""
    # Grain boundary elements (C, B, Hf, Zr)
    df['grain_boundary_total_at'] = sum(
        df.get(f'atomic_percent_{el}', pd.Series(0)).fillna(0)
        for el in ['C', 'B', 'Hf', 'Zr']
    )

    # Ductility-reducing refractories
    df['ductility_reducer_at'] = sum(
        df.get(f'atomic_percent_{el}', pd.Series(0)).fillna(0)
        for el in ['Re', 'W', 'Mo']
    )

    # Heavy elements (affect modulus)
    df['heavy_element_at'] = sum(
        df.get(f'atomic_percent_{el}', pd.Series(0)).fillna(0)
        for el in ['W', 'Re', 'Ta', 'Hf']
    )

    # Light elements
    df['light_element_at'] = sum(
        df.get(f'atomic_percent_{el}', pd.Series(0)).fillna(0)
        for el in ['Al', 'Ti']
    )

    if 'lattice_mismatch_pct' in df.columns:
        df['lattice_mismatch_sq'] = df['lattice_mismatch_pct'] ** 2

    if 'gamma_prime_estimated_vol_pct' in df.columns:
        df['gp_modulus_contrib'] = df['gamma_prime_estimated_vol_pct'] * df.get('density_calculated_gcm3', 8.0)

    return df


def compute_sample_weights(df, groupby_col='alloy_name'):
    """Compute sample weights inversely proportional to alloy frequency."""
    counts = df[groupby_col].value_counts()
    df['_cnt'] = df[groupby_col].map(counts)
    weights = 1.0 / np.sqrt(df['_cnt'])
    df.drop(columns=['_cnt'], inplace=True)
    return weights.values


def load_data(filepath: str, target_key: str, bounds: tuple = None,
              exclude_phase_compositions: bool = False) -> pd.DataFrame:
    """Load and preprocess data for a target property."""
    rows = []
    exclude_keys = {'gamma_composition_at', 'gamma_prime_composition_at'} if exclude_phase_compositions else set()

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                computed_feats = entry.get('computed_features', {})
                if not computed_feats:
                    continue

                base_feats = flatten_dict(computed_feats, exclude_keys=exclude_keys)
                base_feats['processing'] = entry.get('processing', 'unknown')
                alloy_name = entry.get('alloy', f'Unknown_{line_num}')

                measurements = entry.get(target_key)
                if not measurements or not isinstance(measurements, list):
                    continue

                for point in measurements:
                    val = safe_float(point.get('value'))
                    temp = safe_float(point.get('temp_c'))

                    if val is None or temp is None:
                        continue
                    if not (-270 <= temp <= 1500):
                        continue
                    if bounds and not (bounds[0] <= val <= bounds[1]):
                        continue

                    row = base_feats.copy()
                    row['test_temperature_c'] = temp
                    row['target'] = val
                    row['alloy_name'] = alloy_name
                    rows.append(row)
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_temp_features(df)
        df = add_domain_features(df)

    return df


def get_feature_importance(pipeline, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance from trained model."""
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        num_features = list(preprocessor.transformers_[0][2])
        cat_features = list(preprocessor.transformers_[1][2])
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(cat_features).tolist()

        num_feature_names = []
        for feat in num_features:
            num_feature_names.extend([feat, f'{feat}_missing_indicator'])

        all_names = num_feature_names + cat_feature_names

        if isinstance(model, VotingRegressor):
            importances = [e.feature_importances_ for e in model.estimators_ if hasattr(e, 'feature_importances_')]
            avg_importance = np.mean(importances, axis=0) if importances else None
        elif hasattr(model, 'feature_importances_'):
            avg_importance = model.feature_importances_
        else:
            return {}

        if avg_importance is None:
            return {}

        # Aggregate to original feature names
        aggregated = {}
        for name, imp in zip(all_names, avg_importance):
            base = name.split('_missing_indicator')[0].split('_x0_')[0]
            aggregated[base] = aggregated.get(base, 0) + imp

        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}


def train_model(df: pd.DataFrame, target_name: str, xgb_params: Dict,
                rf_params: Dict = None, use_ensemble: bool = True) -> Tuple:
    """Train regression model with cross-validation."""
    print(f"\n{'='*60}\nTRAINING: {target_name.upper()}\n{'='*60}")

    # Holdout split (15% test)
    if len(df) > 50 and df['alloy_name'].nunique() >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['alloy_name']))
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    else:
        train_df, test_df = df.copy(), pd.DataFrame()

    X_train = train_df.drop(columns=['target', 'alloy_name'])
    y_train = train_df['target']
    groups_train = train_df['alloy_name']
    sample_weights = compute_sample_weights(train_df, 'alloy_name')

    print(f"  Train: {len(train_df)} samples, {groups_train.nunique()} alloys, {X_train.shape[1]} features")
    print(f"  Target: [{y_train.min():.1f}, {y_train.max():.1f}], mean={y_train.mean():.1f}")

    # Preprocessing pipeline
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

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

    # Model
    if use_ensemble and rf_params:
        model = VotingRegressor([
            ('xgb', xgb.XGBRegressor(**xgb_params, n_jobs=-1)),
            ('rf', RandomForestRegressor(**rf_params, n_jobs=-1))
        ], n_jobs=-1)
    else:
        model = xgb.XGBRegressor(**xgb_params, n_jobs=-1)

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    # Cross-validation
    n_splits = min(5, groups_train.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    cv_metrics = {'mae': [], 'rmse': [], 'r2': []}

    print(f"\n  CV ({n_splits}-fold): ", end="")
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr, model__sample_weight=sample_weights[train_idx])
        preds = pipeline.predict(X_val)

        cv_metrics['mae'].append(mean_absolute_error(y_val, preds))
        cv_metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_metrics['r2'].append(r2_score(y_val, preds))

    avg_mae, avg_r2 = np.mean(cv_metrics['mae']), np.mean(cv_metrics['r2'])
    print(f"MAE={avg_mae:.2f}, R2={avg_r2:.3f}")

    # Holdout evaluation
    test_metrics = {}
    if not test_df.empty:
        X_test = test_df.drop(columns=['target', 'alloy_name'])
        y_test = test_df['target']
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
        test_preds = pipeline.predict(X_test)
        test_metrics = {
            'mae': mean_absolute_error(y_test, test_preds),
            'r2': r2_score(y_test, test_preds)
        }
        print(f"  Holdout: MAE={test_metrics['mae']:.2f}, R2={test_metrics['r2']:.3f}")

    # Final training on all data
    X_full = df.drop(columns=['target', 'alloy_name'])
    y_full = df['target']
    pipeline.fit(X_full, y_full, model__sample_weight=compute_sample_weights(df, 'alloy_name'))

    feat_importance = get_feature_importance(pipeline, X_train.columns.tolist())

    metrics = {
        "cv_mae": float(avg_mae),
        "cv_r2": float(avg_r2),
        "test_mae": test_metrics.get('mae'),
        "test_r2": test_metrics.get('r2'),
        "n_samples": len(df),
        "n_alloys": int(df['alloy_name'].nunique()),
        "n_features": X_train.shape[1],
    }

    return pipeline, X_train.columns.tolist(), metrics, feat_importance


# =============================================================================
# CONFIGURATION
# =============================================================================

XGB_DEFAULT = {
    'n_estimators': 800, 'learning_rate': 0.02, 'max_depth': 5,
    'min_child_weight': 3, 'subsample': 0.75, 'colsample_bytree': 0.75,
    'colsample_bylevel': 0.75, 'reg_lambda': 2.0, 'reg_alpha': 0.5,
    'gamma': 0.1, 'tree_method': 'hist', 'random_state': 42
}

RF_DEFAULT = {
    'n_estimators': 400, 'max_depth': 12, 'min_samples_split': 5,
    'min_samples_leaf': 2, 'max_features': 'sqrt', 'bootstrap': True, 'random_state': 42
}

TARGETS = {
    "ys": {"key": "yield_strength", "name": "Yield Strength", "bounds": (50, 2000), "exclude_phase": False},
    "uts": {"key": "uts", "name": "Ultimate Tensile Strength", "bounds": (50, 2500), "exclude_phase": False},
    "el": {"key": "elongation", "name": "Elongation", "bounds": (0, 80), "exclude_phase": True},
    "em": {"key": "elasticity", "name": "Elastic Modulus", "bounds": (50, 350), "exclude_phase": True},
}


def load_tuned_params(model_id: str, models_dir: str) -> Tuple[Dict, Dict]:
    """Load tuned parameters from tuned_params/ folder, or return defaults."""
    params_file = os.path.join(models_dir, "tuned_params", f"{model_id}.json")
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            tuned = json.load(f)
        print(f"  Using tuned params (R²={tuned.get('best_r2', 0):.3f})")
        return tuned.get('xgb_params', XGB_DEFAULT), tuned.get('rf_params', RF_DEFAULT)
    print("  Using default params")
    return XGB_DEFAULT.copy(), RF_DEFAULT.copy()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(current_dir, "training_data", "train_77alloys.jsonl")

    OUTPUT_DIR = os.path.join(current_dir, "saved_models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nSUPERALLOY ML TRAINING\nData: {DATA_FILE}\n")

    for model_id, cfg in TARGETS.items():
        df = load_data(DATA_FILE, cfg["key"], cfg.get("bounds"), cfg.get("exclude_phase", False))

        if df.empty or len(df) < 20:
            print(f"Skipping {cfg['name']}: insufficient data ({len(df)} samples)")
            continue

        # Load tuned params if available, otherwise use defaults
        xgb_params, rf_params = load_tuned_params(model_id, current_dir)

        pipeline, features, metrics, feat_imp = train_model(
            df, cfg["name"], xgb_params, rf_params, use_ensemble=True
        )

        # Save
        filename = os.path.join(OUTPUT_DIR, f"model_{model_id}.pkg")
        joblib.dump({
            'model': pipeline, 'features': features, 'feature_importance': feat_imp,
            'metrics': metrics, 'target_name': cfg["name"], 'target_key': cfg["key"],
            'bounds': cfg.get("bounds"), 'exclude_phase_compositions': cfg.get("exclude_phase", False)
        }, filename)
        print(f"  Saved: {filename}")

    print("\nTraining complete.")
