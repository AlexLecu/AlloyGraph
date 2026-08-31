# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlloyGraph is a multi-agent AI platform for nickel-based superalloy property prediction, inverse design, and knowledge-driven research. It combines a Knowledge Graph (77 alloys in Weaviate + GraphDB), physics-informed ML (XGBoost/RandomForest), and LLM agents (CrewAI with Llama 3.3-70B via any OpenAI-compatible provider, currently DeepInfra) in three modes: Research Chat, Property Evaluator, and Alloy Designer.

## Commands

### Docker (full platform — 6 services)
```bash
docker compose up -d                    # Start all services
docker compose build backend frontend   # Rebuild after code changes
docker compose down -v                  # Stop and remove volumes
docker compose logs -f backend          # Follow backend logs
```

### Backend (local development)
```bash
cd backend
source .venv/bin/activate
pip install -r requirements.txt
python app.py                           # Flask on port 5001
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # Vite dev server on :5173
npm run build      # Production build to dist/
```

### Tests
```bash
# Unit tests (require Weaviate/GraphDB running)
pytest backend/alloy_crew/tests/ -v

# Single test module
pytest backend/alloy_crew/tests/test_feature_engineering.py -v

# Root-level integration tests
python test_agent_corrections.py
```

### ML model training
```bash
python backend/alloy_crew/models/train_ml_models.py
```

## Architecture

### Three-Mode System
- **Research Chat** (`services/chat_service.py`): Streaming LLM responses with KG context retrieval. Uses an OpenAI-compatible client via `alloy_crew.agents.resolve_chat_endpoint()` — the same provider order as the agents.
- **Property Evaluator** (`alloy_crew/alloy_evaluator.py`): Sequential Analyst → Reviewer agent pipeline
- **Alloy Designer** (`alloy_crew/alloy_designer.py`): 3-phase loop — LLM synthesis → deterministic optimizer → LLM evaluation

### Agent Pipeline (Evaluator & Designer Phase 3)
```
AlloyAnalysisTool (ML + Physics + KG triangulation)
    → Analyst agent investigates discrepancies via AlloyKGSearchTool
    → MetallurgyVerifierTool (validation-only: bounds, coherency, TCP risk)
    → Reviewer agent validates Analyst's predictions
    → Deterministic validation (compute_metallurgy_validation)
    → Trust system (_evaluate_agent_trust) applies ignored HIGH-confidence proposals
```

### Designer 3-Phase Pipeline
- **Phase 1**: Designer agent invents composition using `QuickCheckTool` for fast physics feedback
- **Phase 2**: `DeterministicOptimizer` (Guard + Tuner) — no LLM, ~1 sec, constrained gradient steps
- **Phase 3**: Full Analyst → Reviewer evaluation (authoritative property assessment)

### Alloy Classification (drives all physics logic)
- **SSS** (Al+Ti+Ta+0.35*Nb < 2%): Solid solution strengthened
- **γ'** (Al+Ti+Ta >= 2%): Gamma prime precipitation hardened — further split wrought/cast
- **SC/DS**: Single crystal/directionally solidified (Re >= 2%, high Ta+W)

### Key Data Flow
Flask endpoints (`app.py`) → `AlloyEvaluationCrew` or `IterativeDesignCrew` → agent tools → ML models (`models/predictor.py`) + physics (`config/alloy_parameters.py`) + KG search (`tools/rag_tools.py` via Weaviate)

## Key Files

| File | Purpose |
|------|---------|
| `backend/app.py` | Flask API — `/api/validate`, `/api/design`, `/api/chat` |
| `backend/alloy_crew/agents.py` | Agent definitions + LLM resolution (DeepInfra → Together → Groq → OpenAI → Ollama) |
| `backend/alloy_crew/alloy_evaluator.py` | Evaluation crew + trust system |
| `backend/alloy_crew/alloy_designer.py` | 3-phase design pipeline + CrewAI event bus reset |
| `backend/alloy_crew/deterministic_optimizer.py` | Guard (surgical fixes) + Tuner (±2% gradient) |
| `backend/alloy_crew/config/alloy_parameters.py` | All physics constants, thresholds, bounds, decay curves |
| `backend/alloy_crew/tools/analysis_tool.py` | ML/Physics/KG triangulation + temperature calibration |
| `backend/alloy_crew/tools/metallurgy_tools.py` | Validation: bounds, coherency, penalties, TCP risk |
| `backend/alloy_crew/tools/rag_tools.py` | Weaviate vector search (KG) |
| `backend/alloy_crew/models/feature_engineering.py` | Morinaga Md, lattice mismatch, γ' fraction, SSS |
| `backend/alloy_crew/models/predictor.py` | XGBoost + RandomForest ensemble wrapper |
| `backend/alloy_crew/schemas.py` | Pydantic models (JSON string workaround for Groq schema limits) |
| `backend/services/chat_service.py` | Streaming chat with intent routing |
| `backend/pipeline/run_pipeline_docker.py` | One-time data initialization into Weaviate + GraphDB |

## Domain-Specific Conventions

### TCP Risk Classification
Uses **Md_avg** (bulk) for primary thresholds — NOT Md_gamma (matrix). Md_gamma is only a secondary boost. Morinaga's 1984 thresholds were calibrated against bulk Md. Using matrix Md causes false positives.

### Temperature Modeling
Physics predictions must apply temperature degradation factors. Key calibration points are Waspaloy wrought bar (538–982°C). GP decay uses two tau constants (TAU1=450, TAU2=66) and takes the alloy's own γ' fraction
via `get_temperature_factor(..., gp_fraction=gp)`, which sets the solvus
(`SOLVUS_BASE + SOLVUS_GP_COEFF * gp`) and hence where stage-3 dissolution begins. EM uses `1.0 - 0.00032 * ΔT`.

### Elastic Modulus
`calculate_em_rule_of_mixtures` returns the **Voigt–Reuss–Hill average** — the midpoint
`(voigt + reuss) / 2` of the Reuss harmonic bound (`1/E = Σ(w_i/E_i)`) and the Voigt
arithmetic bound. It is not the Reuss bound alone; pure Voigt overshoots by 5–22% for
refractory-rich alloys, which is why the Hill midpoint is used rather than Voigt.

### Groq Schema Compatibility
All agent tool Pydantic schemas use `str` instead of `Dict[str, float]` for composition fields, with `field_validator` to accept both dict and str inputs. Groq rejects `additionalProperties: {"type": "number"}`.

### CrewAI Event Bus
The global event bus accumulates events across multiple crew kickoffs. `_reset_crewai_event_bus()` must be called between iterations — sets `max_stack_depth=0` to disable the 100-event depth limit.

### Optimizer Philosophy
The optimizer is intentionally light-touch. LLM chooses alloy architecture; optimizer only adjusts performance knobs within tight guardrails (±2% per element). No overshoot factors, no fixed trace additions, no composition rewrites.

## Environment

Requires `.env` in the repo root with at least one provider key — `DEEPINFRA_API_KEY` preferred (Groq withdrew llama-3.3-70b-versatile on 2026-08-16). See `.env.example`. Docker services: frontend (:3000), Weaviate (:8081), GraphDB (:7200). Backend is internal-only (proxied via Nginx).
