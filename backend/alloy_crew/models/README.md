# AlloyGraph Property Prediction Models

This directory contains the machine learning core of AlloyGraph, responsible for predicting the mechanical properties of Ni-based superalloys across a wide temperature range.

## Overview
The system employs a **Hybrid Physics-ML Engine**. Instead of training on raw composition alone, we first derive meaningful metallurgical parameters to guide the models.

We train **4 ensemble models** (Random Forest + XGBoost) to predict:
1. **Yield Strength (YS)** - MPa
2. **Ultimate Tensile Strength (UTS)** - MPa
3. **Elongation (EL)** - %
4. **Elastic Modulus (EM)** - GPa

Each model is trained using **GroupKFold cross-validation** (grouped by alloy name) to prevent data leakage.

## Core Components

### 1. Feature Engineering (`feature_engineering.py`)
This is the "Metallurgical Bridge" that transforms raw input into physical descriptors:
- **Phase Stability**: Calculates average Md (d-electron parameter) to estimate TCP phase risk.
- **Microstructure**: Estimates Gamma Prime ($\gamma'$) volume fraction based on Al, Ti, Nb, and Ta content.
- **Physical Properties**: Calculates theoretical density using the rule of mixtures.
- **Compositional Ratios**: Derives critical ratios (e.g., Al/Ti, Cr/Co) that influence oxidation resistance and strengthening.

### 2. Training Pipeline (`train_ml_models.py`)
Responsible for building and updating the models:
- **Data Source**: Consumes `train_77alloys.jsonl` (77 alloys for training).
- **Architecture**: A **Voting Ensemble** combining **XGBoost** and **Random Forest Regressors**.
- **Cross-Validation**: Uses `GroupKFold` (grouped by Alloy name) to ensure the models generalize to unseen alloys, not just unseen temperature points for known alloys.

### 3. Inference Engine (`predictor.py`)
Provides a clean API for the rest of the application:
- **Singleton Pattern**: Ensures models are loaded into memory once.
- **Input**: Accepts a dictionary of weight percentages (wt%).
- **Execution**:
    1. Derives physical features.
    2. Expands the input across target temperatures (default 20°C to 1100°C).
    3. Returns a structured DataFrame of predictions (YS, UTS, EL, EM).

## Model Storage
- **saved_models/**: Contains the serialized `.pkg` files. These files bundle the trained ensemble along with the feature names required for alignment.

### Current Benchmarks (trained on 77 alloys)
| Property | Samples | Alloys | CV R² | CV MAE | Holdout R² | Holdout MAE |
|----------|---------|--------|-------|--------|------------|-------------|
| Yield Strength | 264 | 51 | **0.841** | 92.3 MPa | **0.904** | 75.2 MPa |
| Tensile Strength | 296 | 54 | **0.878** | 97.4 MPa | **0.906** | 81.9 MPa |
| Elongation | 285 | 53 | 0.422 | 8.4% | 0.029 | 6.7% |
| Elastic Modulus | 355 | 47 | 0.376 | 15.4 GPa | **0.737** | 14.5 GPa |

### Tuned Hyperparameters

Tuned parameters are stored in `tuned_params/` folder:
- `ys.json` - Yield Strength (tuned R²: 0.864)
- `uts.json` - Ultimate Tensile Strength (tuned R²: 0.860)
- `el.json` - Elongation (tuned R²: 0.541)
- `em.json` - Elastic Modulus (tuned R²: 0.635)

## Usage

#### To re-tune hyperparameters:
```bash
python tune_hyperparameters.py
```

#### To train models with tuned parameters:
```bash
python train_ml_models.py
```
