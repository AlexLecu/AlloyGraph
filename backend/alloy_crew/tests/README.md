# AlloyMind Test Suite

This directory contains a suite of incremental tests designed to verify the core components of the AlloyMind system, from low-level ML models to complex agentic loops.

## How to Run

### Run All Tests with pytest
```bash
cd backend
source .venv/bin/activate
pytest alloy_crew/tests/
```

### Run Individual Tests
```bash
python -m alloy_crew.tests.test_step_1_ml_tool
python -m alloy_crew.tests.test_step_2_rag_tool
python -m alloy_crew.tests.test_step_3_1_evaluator
python -m alloy_crew.tests.test_step_3_2_evaluator
python -m alloy_crew.tests.test_feature_engineering
python -m alloy_crew.tests.test_metallurgy_validation
```

## Test Suite Overview

#### 🧪 Step 1: ML Tool (`test_step_1_ml_tool.py`)
Tests the `AlloyPredictorTool` wrapper for CrewAI. Validates tool signature, physics engine integration (Density/Gamma Prime), and JSON output format.

#### 🧪 Step 2: RAG Tool (`test_step_2_rag_tool.py`)
Verifies the `AlloySearchTool` integration with Weaviate. Tests connectivity, property retrieval, and multi-step GraphQL fetch logic.

#### 🧪 Step 3.1: Evaluator - Good Alloy (`test_step_3_1_evaluator.py`)
Tests the `AlloyEvaluationCrew` with a valid Ni-based superalloy composition. Validates ML predictions, physics verification (TCP risk, Md calculation), and confidence scoring.

#### 🧪 Step 3.2: Evaluator - Bad Alloy (`test_step_3_2_evaluator.py`)
Tests the `AlloyEvaluationCrew` with a problematic composition (excessive Re → high TCP risk). Verifies correct identification of phase instability issues.

#### 🧪 Feature Engineering (`test_feature_engineering.py`)

Tests physics-based feature calculations for alloy compositions:
- **Test 1:** Waspaloy validation (Md_gamma, lattice mismatch, γ' fraction, density)
- **Test 2:** Edge cases (pure Ni, high refractory content)

**Key validations:**
- Md_gamma calculation (TCP phase stability)
- Lattice mismatch between γ and γ' phases
- Gamma prime volume fraction estimation
- Density calculations

#### 🧪 Metallurgy Validation (`test_metallurgy_validation.py`)

Tests property coherency validation rules:
- **Test 1:** High strength requires adequate γ' fraction (>40%)
- **Test 2:** Density should correlate with refractory content
- **Test 3:** High ductility + heavy refractories is unusual
- **Test 4:** UTS/YS ratio validation (1.1-1.6)
- **Test 5:** γ' fraction should align with γ' formers (Al+Ti+Ta)
