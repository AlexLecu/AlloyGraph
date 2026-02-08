from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Type, Any
import json

from ..models.predictor import AlloyPredictor
from ..models.feature_engineering import calculate_density, estimate_gamma_prime_vol_pct, wt_to_at_percent

class AlloyPredictionInput(BaseModel):
    """Input for the Alloy Prediction Tool."""
    composition: Dict[str, float] = Field(
        ..., 
        description="Dictionary of element symbols and their weight percentages (e.g., {'Ni': 60, 'Al': 5})."
    )
    temperature_c: int = Field(
        20,
        description="Target temperature in Celsius for property prediction."
    )
    processing: str = Field(
        "cast",
        description="Processing type: 'cast' or 'wrought'. Defaults to 'cast' (more conservative predictions)."
    )

class AlloyPredictorTool(BaseTool):
    name: str = "AlloyPredictorTool"
    description: str = (
        "Predicts mechanical properties (Yield Strength, UTS, Elongation, Elastic Modulus) of a superalloy "
        "given its composition. Use this to VALIDATE if a design meets specifications."
    )
    args_schema: Type[BaseModel] = AlloyPredictionInput

    def _run(self, composition: Dict[str, float], temperature_c: int = 20, processing: str = "cast", **kwargs: Any) -> str:
        try:
            # 1. Initialize Predictor (Cached via Factory)
            predictor = AlloyPredictor.get_shared_predictor(model_dir=None)

            # 2. Prepare Options
            options = {
                "processing": processing
            }

            # 3. Predict - get individual model predictions if available
            report_df = predictor.predict(composition, options, temperatures=[temperature_c])

            if report_df.empty:
                return '{"error": "No data returned"}'

            row = report_df.iloc[0]
            
            
            dens = calculate_density(composition)
            at_pct = wt_to_at_percent(composition)
            gp = estimate_gamma_prime_vol_pct(at_pct)

            # Extract predictions
            ys_pred = float(row['ys'])
            uts_pred = float(row['uts'])
            el_pred = float(row['el'])
            em_pred = float(row['em'])

            # Estimate variance based on prediction magnitude and property type
            # YS typically has higher variance than elongation
            ys_variance = max(15.0, ys_pred * 0.05)  # ~5% of prediction
            uts_variance = max(20.0, uts_pred * 0.05)
            el_variance = max(1.0, el_pred * 0.08)  # Elongation more variable
            em_variance = max(5.0, em_pred * 0.04)  # Elastic modulus ~4% uncertainty
            
            # Calculate uncertainty intervals (±2σ for 95% confidence)
            ys_uncertainty = ys_variance * 2.0
            uts_uncertainty = uts_variance * 2.0
            el_uncertainty = el_variance * 2.0
            em_uncertainty = em_variance * 2.0
            
            # Calculate ML confidence from relative variance (variance/prediction)
            # Consistent scaling: all properties use the same formula
            # Higher relative variance → lower confidence
            def _relative_confidence(variance, prediction):
                if prediction <= 0:
                    return 0.4
                relative_var = variance / prediction
                return max(0.4, min(1.0, 1.0 - relative_var * 5.0))

            ys_confidence = _relative_confidence(ys_variance, ys_pred)
            uts_confidence = _relative_confidence(uts_variance, uts_pred)
            el_confidence = _relative_confidence(el_variance, el_pred)
            em_confidence = _relative_confidence(em_variance, em_pred)

            result_dict = {
                "Yield Strength": round(ys_pred, 1),
                "Tensile Strength": round(uts_pred, 1),
                "Elongation": round(el_pred, 1),
                "Elastic Modulus": round(em_pred, 1),
                "Density": float(dens),
                "Gamma Prime": float(gp),
                "source": "ML Prediction + Physics",
                # Add intervals as separate field
                "property_intervals": {
                    "Yield Strength": {
                        "lower": round(ys_pred - ys_uncertainty, 1),
                        "upper": round(ys_pred + ys_uncertainty, 1),
                        "uncertainty": round(ys_uncertainty, 1)
                    },
                    "Tensile Strength": {
                        "lower": round(uts_pred - uts_uncertainty, 1),
                        "upper": round(uts_pred + uts_uncertainty, 1),
                        "uncertainty": round(uts_uncertainty, 1)
                    },
                    "Elongation": {
                        "lower": round(max(0.0, el_pred - el_uncertainty), 1),
                        "upper": round(el_pred + el_uncertainty, 1),
                        "uncertainty": round(el_uncertainty, 1)
                    },
                    "Elastic Modulus": {
                        "lower": round(em_pred - em_uncertainty, 1),
                        "upper": round(em_pred + em_uncertainty, 1),
                        "uncertainty": round(em_uncertainty, 1)
                    }
                },
                "model_confidence": {
                    "Yield Strength": round(ys_confidence, 3),
                    "Tensile Strength": round(uts_confidence, 3),
                    "Elongation": round(el_confidence, 3),
                    "Elastic Modulus": round(em_confidence, 3),
                    "note": "Estimated from prediction variance (5% heuristic)"
                }
            }
            
            return json.dumps(result_dict)

        except Exception as e:
            return json.dumps({"error": f"Prediction Failed: {str(e)}"})
