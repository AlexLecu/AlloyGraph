import pandas as pd
import joblib
import os
from .feature_engineering import compute_alloy_features



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

class AlloyPredictor:
    @staticmethod
    def get_shared_predictor(model_dir=None):
        """Returns a singleton instance of AlloyPredictor to avoid reloading models."""
        if model_dir is None:
            # Default to the 'saved_models' directory relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "saved_models")
            
        global _SHARED_PREDICTOR
        if _SHARED_PREDICTOR is None:
            _SHARED_PREDICTOR = AlloyPredictor(model_dir)
        return _SHARED_PREDICTOR

    def __init__(self, model_dir="."):
        self.models = {}
        self.required_features = {}
        
        # Load the 4 models
        for name in ['ys', 'uts', 'el', 'em']:
            filename = os.path.join(model_dir, f"model_{name}.pkg")
            if os.path.exists(filename):
                print(f"Loading {name.upper()} model...")
                pkg = joblib.load(filename)
                self.models[name] = pkg['model']
                self.required_features[name] = pkg['features']
            else:
                print(f"Warning: {filename} not found.")

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
        # print("-> Computing metallurgical features...")
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
        
        # 5. RUN PREDICTIONS FOR EACH MODEL
        results = {'Temp': temperatures}
        
        for name, model in self.models.items():
            # A. Align Columns (The Fix)
            req_cols = self.required_features[name]
            df_aligned = df_raw.reindex(columns=req_cols)
            
            # B. Smart Fill - use 'cast' for processing (in-distribution default)
            for col in df_aligned.columns:
                if col == 'processing':
                    df_aligned[col] = df_aligned[col].fillna('cast')
                else:
                    df_aligned[col] = df_aligned[col].fillna(0.0)


            
            # C. Predict
            results[name] = model.predict(df_aligned)
            
        return pd.DataFrame(results)
