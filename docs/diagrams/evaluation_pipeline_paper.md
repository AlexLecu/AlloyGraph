# Evaluation Pipeline - Paper Version

A cleaner diagram suitable for publication.

```mermaid
flowchart LR
    subgraph Input[" "]
        I["<b>Input</b><br/>Composition<br/>Temperature<br/>Processing"]
    end

    subgraph Pipeline["Multi-Agent Evaluation Pipeline"]
        direction TB

        subgraph V["① Validator"]
            V1["AlloyPredictorTool"]
            V2["XGBoost + RF Ensemble<br/>→ YS, UTS, EL, EM"]
        end

        subgraph A["② Arbitrator"]
            A1["DataFusionTool"]
            A2["ML + KG Fusion<br/>→ Confidence Scores"]
        end

        subgraph P["③ Physicist"]
            P1["MetallurgyVerifierTool"]
            P2["TCP Risk, Coherency<br/>→ Penalty Score"]
        end

        subgraph C["④ Corrector"]
            C1["PhysicsCorrectionsTool"]
            C2["Constraint Validation<br/>→ Adjusted Values"]
        end

        subgraph S["⑤ Summarizer"]
            S1["LLM Reasoning"]
            S2["Expert Explanation<br/>→ Recommendations"]
        end

        V --> A --> P --> C --> S
    end

    subgraph Output[" "]
        O["<b>Output</b><br/>Properties ± CI<br/>TCP Risk<br/>Explanation"]
    end

    Input --> Pipeline --> Output

    style V fill:#fff3e0
    style A fill:#e3f2fd
    style P fill:#fce4ec
    style C fill:#e8eaf6
    style S fill:#e0f2f1
```

---

## Alternative: Vertical Flow (Better for Papers)

```mermaid
flowchart TB
    Input["<b>Alloy Composition</b><br/>(wt%, T°C, processing)"]

    subgraph Agents["<b>Multi-Agent Pipeline</b>"]
        direction TB

        Agent1["<b>① Validator</b><br/><i>AlloyPredictorTool</i><br/>ML property prediction"]
        Agent2["<b>② Arbitrator</b><br/><i>DataFusionTool</i><br/>ML + KG confidence fusion"]
        Agent3["<b>③ Physicist</b><br/><i>MetallurgyVerifierTool</i><br/>TCP risk & coherency audit"]
        Agent4["<b>④ Corrector</b><br/><i>PhysicsCorrectionsTool</i><br/>Physics-based adjustments"]
        Agent5["<b>⑤ Summarizer</b><br/><i>LLM Reasoning</i><br/>Expert explanation"]

        Agent1 --> Agent2 --> Agent3 --> Agent4 --> Agent5
    end

    Output["<b>Evaluation Report</b><br/>Properties ± confidence<br/>TCP risk assessment<br/>Metallurgy metrics"]

    Input --> Agents --> Output

    style Agent1 fill:#fff3e0,stroke:#e65100
    style Agent2 fill:#e3f2fd,stroke:#1565c0
    style Agent3 fill:#fce4ec,stroke:#c2185b
    style Agent4 fill:#e8eaf6,stroke:#5e35b1
    style Agent5 fill:#e0f2f1,stroke:#00897b
```

---

## Tool Summary Table

| Agent | Tool | Input | Output |
|-------|------|-------|--------|
| Validator | AlloyPredictorTool | Composition | YS, UTS, EL, EM, γ', ρ, Md |
| Arbitrator | DataFusionTool | ML predictions + KG | Fused values + confidence |
| Physicist | MetallurgyVerifierTool | Fused predictions | Penalty score, TCP risk |
| Corrector | PhysicsCorrectionsTool | Predictions + violations | Corrected values |
| Summarizer | (LLM only) | All data | Human explanation |
