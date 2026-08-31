import pandas as pd
import numpy as np
import joblib
import os
import threading
import logging
from .feature_engineering import compute_alloy_features

logger = logging.getLogger(__name__)



def flatten_dict(d, parent_key='', sep='_'):
    """Flattens the computed features (same logic as training)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Singleton Trace
_SHARED_PREDICTOR = None
# Guards construction of the singleton above; see get_shared_predictor.
_PREDICTOR_LOCK = threading.Lock()

# Active model generation. v2 was trained after the Vegard/lattice-mismatch fix
# (Ni given its correct 0.0 coefficient, unlisted elements no longer picking up
# a spurious 0.1). v1 lives on in saved_models/ for comparison and is selectable
# via ALLOYGRAPH_MODEL_DIR.
DEFAULT_MODEL_SUBDIR = "saved_models_v2"
LEGACY_MODEL_SUBDIR = "saved_models"
_MODEL_IDS = ('ys', 'uts', 'el', 'em')


def resolve_model_dir(model_dir=None):
    """Pick the model directory: explicit arg > $ALLOYGRAPH_MODEL_DIR > v2 > v1.

    Falls back to the legacy directory only if v2 is absent or incomplete, and
    says so loudly -- silently serving a different model generation than the
    caller expects is worse than a noisy start-up.
    """
    if model_dir:
        return model_dir

    env_dir = os.environ.get("ALLOYGRAPH_MODEL_DIR")
    if env_dir:
        logger.info("Using ALLOYGRAPH_MODEL_DIR=%s", env_dir)
        return env_dir

    current_dir = os.path.dirname(os.path.abspath(__file__))

    def complete(d):
        return all(os.path.exists(os.path.join(d, f"model_{m}.pkg")) for m in _MODEL_IDS)

    preferred = os.path.join(current_dir, DEFAULT_MODEL_SUBDIR)
    if complete(preferred):
        return preferred

    legacy = os.path.join(current_dir, LEGACY_MODEL_SUBDIR)
    if complete(legacy):
        logger.warning(
            "%s missing or incomplete; falling back to legacy models in %s "
            "(these predate the lattice-mismatch fix). Retrain with "
            "train_ml_models.py --out %s to restore v2.",
            DEFAULT_MODEL_SUBDIR, LEGACY_MODEL_SUBDIR, preferred,
        )
        return legacy

    logger.error("No complete model set found in %s or %s", preferred, legacy)
    return preferred


class AlloyPredictor:
    @staticmethod
    def get_shared_predictor(model_dir=None):
        """Returns a singleton instance of AlloyPredictor to avoid reloading models.

        Locked because this is a check-then-act on shared state and the server
        is multi-threaded: background job threads already call it concurrently,
        and under gthread request threads do too. Unsynchronised, two threads
        arriving together both saw None and both constructed -- each loading
        four model pickles, ~275 MB and a couple of seconds, for a result one of
        them then discarded. Worse, `_SHARED_PREDICTOR` was published before
        `_model_dir` was attached, so a third thread could observe the instance
        mid-initialisation and take the mismatch-warning branch against a
        missing attribute.

        Double-checked inside the lock so the steady-state path stays a plain
        attribute read for all but the first caller.
        """
        model_dir = resolve_model_dir(model_dir)

        global _SHARED_PREDICTOR
        if _SHARED_PREDICTOR is None:
            with _PREDICTOR_LOCK:
                if _SHARED_PREDICTOR is None:
                    predictor = AlloyPredictor(model_dir)
                    # Attach before publishing, so no thread can see a
                    # half-initialised singleton.
                    predictor._model_dir = model_dir
                    _SHARED_PREDICTOR = predictor
                    return _SHARED_PREDICTOR

        if getattr(_SHARED_PREDICTOR, '_model_dir', None) != model_dir:
            logger.warning(
                "AlloyPredictor singleton already initialized with model_dir=%s; "
                "ignoring request for %s",
                getattr(_SHARED_PREDICTOR, '_model_dir', None), model_dir,
            )
        return _SHARED_PREDICTOR

    def __init__(self, model_dir="."):
        self.models = {}
        self.required_features = {}
        
        # Load the 4 models
        for name in ['ys', 'uts', 'el', 'em']:
            filename = os.path.join(model_dir, f"model_{name}.pkg")
            if os.path.exists(filename):
                logger.info(f"Loading {name.upper()} model")
                pkg = joblib.load(filename)
                self.models[name] = pkg['model']
                self.required_features[name] = pkg['features']
            else:
                logger.warning(f"Model file not found: {filename}")

    @staticmethod
    def _add_temp_features(df, temp_col='test_temperature_c'):
        """Add temperature-derived features matching training pipeline."""
        if temp_col not in df.columns:
            return df
        df[temp_col] = df[temp_col].clip(lower=-270, upper=1500)
        t_kelvin = (df[temp_col] + 273.15).clip(lower=1.0)
        df['temp_c_sq'] = df[temp_col] ** 2
        df['temp_c_cube'] = df[temp_col] ** 3
        df['log_temp_k'] = np.log(t_kelvin)
        df['inv_temp_k'] = 1.0 / t_kelvin
        df['temp_normalized'] = (df[temp_col] - 20) / 1080
        return df

    @staticmethod
    def _add_domain_features(df):
        """Add physics-based domain features matching training pipeline."""
        for col_name, elements in [
            ('grain_boundary_total_at', ['C', 'B', 'Hf', 'Zr']),
            ('ductility_reducer_at', ['Re', 'W', 'Mo']),
            ('heavy_element_at', ['W', 'Re', 'Ta', 'Hf']),
            ('light_element_at', ['Al', 'Ti']),
        ]:
            df[col_name] = sum(
                df.get(f'atomic_percent_{el}', pd.Series(0, index=df.index)).fillna(0)
                for el in elements
            )
        if 'lattice_mismatch_pct' in df.columns:
            df['lattice_mismatch_sq'] = df['lattice_mismatch_pct'] ** 2
        if 'gamma_prime_estimated_vol_pct' in df.columns:
            df['gp_modulus_contrib'] = (
                df['gamma_prime_estimated_vol_pct']
                * df.get('density_calculated_gcm3', pd.Series(8.0, index=df.index)).fillna(8.0)
            )
        return df

    def predict(self, composition_wt, extra_params=None, temperatures=None):
        """
        Main prediction logic.
        
        Args:
            composition_wt (dict): {'Ni': 60, 'Al': 5...}
            extra_params (dict): Optional. {'family': 'sx_3rd_gen', 'TCP_risk': 'low'}
            temperatures (list): List of temperatures to test. Defaults to [20, 800, 1000].
        """
        if extra_params is None: extra_params = {}
        if temperatures is None: temperatures = [20, 600, 800, 900, 1000, 1100]

        # 1. COMPUTE PHYSICAL FEATURES (The "Bridge")
        computed_feats = compute_alloy_features(composition_wt)
        
        # 2. FLATTEN
        flat_feats = flatten_dict(computed_feats)
        
        # 3. MERGE WITH OVERRIDES (e.g., family, TCP_risk)
        flat_feats.update(extra_params)
        
        # 4. PREPARE INPUT DATAFRAME (Expand for temperatures)
        rows = []
        for t in temperatures:
            row = flat_feats.copy()
            row['test_temperature_c'] = t
            rows.append(row)
            
        df_raw = pd.DataFrame(rows)

        # 4b. ADD DERIVED FEATURES
        df_raw = self._add_temp_features(df_raw)
        df_raw = self._add_domain_features(df_raw)

        # 5. RUN PREDICTIONS FOR EACH MODEL
        results = {'Temp': temperatures}
        
        for name, model in self.models.items():
            # A. Align Columns (The Fix)
            req_cols = self.required_features[name]
            df_aligned = df_raw.reindex(columns=req_cols)
            
            # B. Smart Fill
            cat_defaults = {
                'processing': 'cast',
                'TCP_risk': 'Moderate',
                'alloy_name': 'unknown'
            }

            for col in df_aligned.columns:
                if col in cat_defaults:
                    df_aligned[col] = df_aligned[col].fillna(cat_defaults[col])
                else:
                    df_aligned[col] = df_aligned[col].fillna(0.0)


            
            # C. Predict
            results[name] = model.predict(df_aligned)
            
        return pd.DataFrame(results)
