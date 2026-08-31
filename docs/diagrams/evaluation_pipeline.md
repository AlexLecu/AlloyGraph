# Evaluation Pipeline - Agent Workflow

## Overview

The Evaluation Pipeline takes an alloy composition and predicts its mechanical properties through a **5-agent sequential workflow**. Each agent has a specific role, tool, and responsibility in the validation chain.

## Mermaid Diagram - Full Detail

```mermaid
flowchart TB
    classDef inputOutput fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef agent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef tool fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef computation fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef decision fill:#ffebee,stroke:#c62828,stroke-width:2px

    %% Input
    Input["📥 <b>INPUT</b><br/>─────────────<br/>Alloy Composition (wt%)<br/>Temperature (°C)<br/>Processing (cast/wrought)"]:::inputOutput

    %% Agent 1: Validator
    subgraph Agent1["🔬 AGENT 1: Validator"]
        direction TB
        A1Role["<b>Role:</b> Virtual Lab Technician"]
        A1Tool["<b>Tool:</b> AlloyPredictorTool"]:::tool
        A1Compute["<b>Computes:</b><br/>• ML predictions (XGBoost + RF ensemble)<br/>• Yield Strength (MPa)<br/>• Tensile Strength (MPa)<br/>• Elongation (%)<br/>• Elastic Modulus (GPa)<br/>• γ' fraction, Density, Md_gamma"]:::computation
    end

    %% Agent 2: Arbitrator
    subgraph Agent2["⚖️ AGENT 2: Arbitrator"]
        direction TB
        A2Role["<b>Role:</b> Data Fusion Specialist"]
        A2Tool["<b>Tool:</b> DataFusionTool"]:::tool
        A2Compute["<b>Computes:</b><br/>• KG similarity search (cosine distance)<br/>• ML + KG weighted fusion<br/>• Confidence scores per property<br/>• Property intervals (±bounds)<br/>• Source attribution"]:::computation
    end

    %% Agent 3: Physicist
    subgraph Agent3["🛡️ AGENT 3: Physicist"]
        direction TB
        A3Role["<b>Role:</b> Thermodynamic Auditor"]
        A3Tool["<b>Tool:</b> MetallurgyVerifierTool"]:::tool
        A3Compute["<b>Computes:</b><br/>• TCP risk via Md_gamma (threshold: 0.96)<br/>• Lattice mismatch δ (target: |δ| < 0.5%)<br/>• γ/γ' coherency validation<br/>• Refractory element balance<br/>• Penalty score (0-100)"]:::computation
    end

    %% Decision Gate
    Decision{"Penalty<br/>> 50?"}:::decision

    %% Agent 4: Corrector
    subgraph Agent4["🔧 AGENT 4: Corrector"]
        direction TB
        A4Role["<b>Role:</b> Physics Corrections Specialist"]
        A4Tool["<b>Tool:</b> PhysicsCorrectionsProposalTool"]:::tool
        A4Compute["<b>Computes:</b><br/>• Physics constraint violations<br/>• YS/γ' consistency check<br/>• UTS/YS ratio validation (1.0-1.5)<br/>• Elongation bounds (0-30%)<br/>• Corrected property values"]:::computation
    end

    %% Agent 5: Summarizer
    subgraph Agent5["📝 AGENT 5: Summarizer"]
        direction TB
        A5Role["<b>Role:</b> Materials Science Communicator"]
        A5Tool["<b>Tool:</b> None (LLM reasoning only)"]:::tool
        A5Compute["<b>Generates:</b><br/>• Human-readable explanation<br/>• Strengthening mechanism analysis<br/>• Application recommendations<br/>• Trade-off assessment<br/>• Confidence interpretation"]:::computation
    end

    %% Output
    Output["📤 <b>OUTPUT</b><br/>─────────────<br/>Predicted Properties (MPa, %, GPa)<br/>Confidence Intervals<br/>TCP Risk Assessment<br/>Metallurgy Metrics (γ', Md, δ)<br/>Expert Explanation"]:::inputOutput

    Reject["❌ <b>HIGH PENALTY</b><br/>─────────────<br/>Penalty details<br/>Physics violations<br/>Recommendations"]:::decision

    %% Flow
    Input --> Agent1
    Agent1 --> Agent2
    Agent2 --> Agent3
    Agent3 --> Decision
    Decision -->|"No (proceed)"| Agent4
    Decision -->|"Yes (flag)"| Reject
    Agent4 --> Agent5
    Agent5 --> Output
    Reject -.->|"Still outputs<br/>with warnings"| Agent5

    %% Style subgraphs
    style Agent1 fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    style Agent2 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Agent3 fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    style Agent4 fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
    style Agent5 fill:#e0f2f1,stroke:#00695c,stroke-width:2px
```

---

## Paper Version (Simplified)

```mermaid
flowchart TB
    Input["<b>Alloy Composition</b><br/>(wt%, T°C, processing)"]

    subgraph Pipeline["<b>Multi-Agent Evaluation Pipeline</b>"]
        direction TB

        Agent1["<b>① Validator</b><br/><i>AlloyPredictorTool</i><br/>ML property prediction"]
        Agent2["<b>② Arbitrator</b><br/><i>DataFusionTool</i><br/>ML + KG confidence fusion"]
        Agent3["<b>③ Physicist</b><br/><i>MetallurgyVerifierTool</i><br/>TCP risk & coherency audit"]
        Agent4["<b>④ Corrector</b><br/><i>PhysicsCorrectionsTool</i><br/>Physics-based adjustments"]
        Agent5["<b>⑤ Summarizer</b><br/><i>LLM Reasoning</i><br/>Expert explanation"]

        Agent1 --> Agent2 --> Agent3 --> Agent4 --> Agent5
    end

    Output["<b>Evaluation Report</b><br/>Properties ± confidence<br/>TCP risk assessment<br/>Metallurgy metrics"]

    Input --> Pipeline --> Output

    style Agent1 fill:#fff3e0,stroke:#e65100
    style Agent2 fill:#e3f2fd,stroke:#1565c0
    style Agent3 fill:#fce4ec,stroke:#c2185b
    style Agent4 fill:#e8eaf6,stroke:#5e35b1
    style Agent5 fill:#e0f2f1,stroke:#00897b
```

---

## Agent Interaction Matrix

| Agent | Receives From | Sends To | Key Decision |
|-------|---------------|----------|--------------|
| **① Validator** | User composition | Arbitrator | Run ML ensemble |
| **② Arbitrator** | ML predictions | Physicist | Blend with KG data |
| **③ Physicist** | Fused predictions | Corrector OR Penalty | Physics validation |
| **④ Corrector** | Predictions + violations | Summarizer | Apply corrections |
| **⑤ Summarizer** | All pipeline data | User | Generate explanation |

---

## Tool Summary Table

| Agent | Tool | Input | Output |
|-------|------|-------|--------|
| Validator | AlloyPredictorTool | Composition (wt%) | YS, UTS, EL, EM, γ', ρ, Md |
| Arbitrator | DataFusionTool | ML predictions | Fused values + confidence |
| Physicist | MetallurgyVerifierTool | Fused predictions | Penalty score, TCP risk |
| Corrector | PhysicsCorrectionsTool | Predictions + violations | Corrected values |
| Summarizer | (LLM only) | All data | Human explanation |

---

## Evaluation Example

```
Input: CMSX-4 composition
  Ni: 61.7%, Cr: 6.5%, Co: 9.0%, Mo: 0.6%, W: 6.0%
  Al: 5.6%, Ti: 1.0%, Ta: 6.5%, Re: 3.0%, Hf: 0.1%
  Temperature: 850°C, Processing: cast

① Validator (AlloyPredictorTool):
   ML Predictions:
   • YS = 1180 MPa (XGBoost: 1195, RF: 1165)
   • UTS = 1420 MPa
   • EL = 8.2%
   • γ' = 68.5 vol%
   • Md_gamma = 0.92

② Arbitrator (DataFusionTool):
   KG Match: "CMSX-4" (distance: 0.002 → 98% KG weight)
   Fused Values:
   • YS = 1190 MPa ± 45 (confidence: HIGH)
   • UTS = 1410 MPa ± 60 (confidence: HIGH)
   • Source: 98% KG, 2% ML

③ Physicist (MetallurgyVerifierTool):
   TCP Risk: Md_gamma = 0.92 → LOW RISK ✓
   Lattice Mismatch: δ = 0.35% → OPTIMAL ✓
   Coherency: γ' = 68.5% consistent with YS ✓
   Penalty Score: 12/100 (PASS)

④ Corrector (PhysicsCorrectionsTool):
   No corrections needed (penalty < 50)
   UTS/YS ratio = 1.19 → VALID ✓
   Elongation = 8.2% → VALID ✓

⑤ Summarizer (LLM):
   "CMSX-4 exhibits excellent high-temperature strength due to
    its high γ' volume fraction (~68%) and optimal lattice mismatch.
    The low Md_gamma (0.92) indicates good TCP phase stability,
    making it suitable for single-crystal turbine blade applications."

Output:
  ┌─────────────────────────────────────┐
  │ YS:  1190 ± 45 MPa   (HIGH conf.)   │
  │ UTS: 1410 ± 60 MPa   (HIGH conf.)   │
  │ EL:  8.2 ± 1.5%      (MED conf.)    │
  │ γ':  68.5 vol%                      │
  │ TCP: LOW RISK (Md=0.92)             │
  └─────────────────────────────────────┘
```

---

## Agent Details

### Agent 1: Validator (Virtual Lab Technician)
- **Purpose**: Run ML models on the input composition
- **Tool**: `AlloyPredictorTool`
- **Input**: Composition, temperature, processing method
- **Output**: Raw ML predictions for all properties
- **Models**: XGBoost + Random Forest ensemble

### Agent 2: Arbitrator (Data Fusion Specialist)
- **Purpose**: Blend ML predictions with Knowledge Graph data
- **Tool**: `DataFusionTool`
- **Logic**:
  - Searches KG for similar alloys (cosine distance)
  - Weights ML vs KG based on compositional similarity:
    - Close match (distance < 0.01) → Trust KG 99%
    - Moderate match (0.01-0.1) → Weighted blend
    - No match (distance > 0.1) → Trust ML 100%
- **Output**: Fused predictions with confidence scores

### Agent 3: Physicist (Thermodynamic Auditor)
- **Purpose**: Validate physics constraints
- **Tool**: `MetallurgyVerifierTool`
- **Checks**:
  - TCP phase stability: Md_gamma < 0.96 (preferred)
  - Lattice mismatch: |δ| < 0.5% (optimal coherency)
  - Property coherency: YS correlates with γ'
  - Density vs refractory content
- **Output**: Penalty score (0-100), warnings, or REJECTION

### Agent 4: Corrector (Physics Corrections Specialist)
- **Purpose**: Apply physics-based corrections when needed
- **Tool**: `PhysicsCorrectionsProposalTool`
- **Logic**:
  - Low confidence + physics violation → Apply correction
  - High confidence + KG match → Trust data
  - Validates: UTS/YS ratio (1.0-1.5), Elongation bounds
- **Output**: Corrected properties with explanations

### Agent 5: Summarizer (Materials Science Communicator)
- **Purpose**: Generate human-readable explanation
- **Tool**: None (LLM reasoning only)
- **Covers**:
  - Strengthening mechanisms
  - Application suitability
  - Trade-off analysis
  - Confidence interpretation
- **Output**: 3-paragraph expert summary
