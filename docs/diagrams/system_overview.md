# AlloyGraph System Overview

## Overview

AlloyGraph is a multi-agent AI platform for Ni-based superalloy research. It combines a Knowledge Graph (106 alloys), Vector Database (146K+ variants), ML models, and LLM agents to provide three main applications.

## Mermaid Diagram - Full System Architecture

```mermaid
flowchart TB
    classDef user fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef app fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef db fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef ml fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef agent fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    %% User Interface
    User["👤 <b>USER</b><br/>Materials Scientist"]:::user

    %% Frontend
    subgraph Frontend["🖥️ FRONTEND (Streamlit)"]
        direction LR
        ChatUI["💬 Chat Interface<br/>Natural language queries"]
        EvalUI["📊 Evaluation Tool<br/>Composition → Properties"]
        DesignUI["🎨 Design Tool<br/>Properties → Composition"]
    end

    %% Backend API
    subgraph Backend["⚙️ BACKEND (FastAPI)"]
        direction TB
        API["REST API<br/>/chat, /evaluate, /design"]

        subgraph Agents["Multi-Agent System (CrewAI)"]
            direction LR
            RAGAgent["RAG Pipeline<br/>Intent + Retrieval"]:::agent
            EvalAgents["Evaluation Crew<br/>5 Agents"]:::agent
            DesignAgents["Design Crew<br/>7 Agents"]:::agent
        end
    end

    %% Data Layer
    subgraph Data["📚 DATA LAYER"]
        direction LR

        subgraph KG["Knowledge Graph"]
            GraphDB["<b>GraphDB</b><br/>RDF Triplestore<br/>106 Ni-superalloys<br/>Ontology"]:::db
        end

        subgraph Vector["Vector Store"]
            Weaviate["<b>Weaviate</b><br/>Vector DB<br/>146K+ variants<br/>Hybrid search"]:::db
        end
    end

    %% ML Models
    subgraph ML["🤖 ML MODELS"]
        direction LR
        XGB["<b>XGBoost</b><br/>YS, UTS, EL"]:::ml
        RF["<b>Random Forest</b><br/>Ensemble"]:::ml
        Physics["<b>Physics Models</b><br/>γ', Md, mismatch"]:::ml
    end

    %% LLM
    LLM["☁️ <b>Groq LLM</b><br/>Llama 3.3-70B<br/>Intent, Generation, Reasoning"]

    %% Connections
    User --> Frontend
    Frontend --> Backend

    ChatUI --> API
    EvalUI --> API
    DesignUI --> API

    API --> RAGAgent
    API --> EvalAgents
    API --> DesignAgents

    RAGAgent --> Data
    EvalAgents --> ML
    EvalAgents --> Data
    DesignAgents --> ML
    DesignAgents --> Data

    RAGAgent --> LLM
    EvalAgents --> LLM
    DesignAgents --> LLM

    style Frontend fill:#e3f2fd,stroke:#1565c0
    style Backend fill:#fff8e1,stroke:#ff8f00
    style Data fill:#ede7f6,stroke:#5e35b1
    style ML fill:#e8f5e9,stroke:#2e7d32
```

---

## Paper Version (Simplified)

```mermaid
flowchart TB
    User["User Query"]

    subgraph System["<b>AlloyGraph Platform</b>"]
        direction TB

        subgraph Apps["Applications"]
            direction LR
            Chat["💬 Research<br/>Assistant"]
            Eval["📊 Property<br/>Evaluator"]
            Design["🎨 Alloy<br/>Designer"]
        end

        subgraph Core["Multi-Agent Core"]
            direction LR
            RAG["RAG<br/>Pipeline"]
            EvalPipe["Evaluation<br/>Pipeline<br/>(5 agents)"]
            DesignPipe["Design<br/>Pipeline<br/>(7 agents)"]
        end

        subgraph Data["Knowledge Base"]
            direction LR
            KG["Knowledge Graph<br/>(106 alloys)"]
            VDB["Vector DB<br/>(146K variants)"]
            ML["ML Ensemble<br/>(XGB + RF)"]
        end

        Apps --> Core
        Core --> Data
    end

    Output["Alloy Properties /<br/>Design / Explanation"]

    User --> System --> Output

    style Apps fill:#e3f2fd
    style Core fill:#fff3e0
    style Data fill:#ede7f6
```

---

## Component Summary

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| **Frontend** | Chat UI | Streamlit | Natural language queries |
| | Evaluation UI | Streamlit | Composition input & results |
| | Design UI | Streamlit | Target properties input |
| **Backend** | API | FastAPI | REST endpoints |
| | Agents | CrewAI | Multi-agent orchestration |
| **Data** | Knowledge Graph | GraphDB | RDF ontology, 106 alloys |
| | Vector Store | Weaviate | 146K variants, hybrid search |
| **ML** | Property Models | XGBoost + RF | YS, UTS, EL, EM prediction |
| | Physics Models | Python | γ', Md, lattice mismatch |
| **LLM** | Generation | Groq (Llama 3.3) | Intent, reasoning, summaries |

---

## Three Applications

### 1. Research Assistant (RAG Chat)
```
User Query → Intent Classification → Knowledge Retrieval → Response Generation
```
- Answer questions about alloys
- Compare compositions
- Explain metallurgical concepts

### 2. Property Evaluator
```
Composition → 5-Agent Pipeline → Predicted Properties + Confidence
```
- Predict mechanical properties (YS, UTS, EL)
- Calculate metallurgical metrics (γ', Md, TCP risk)
- Physics-based validation

### 3. Alloy Designer
```
Target Properties → 7-Agent Iterative Loop → Optimized Composition
```
- Inverse design problem
- Iterative refinement with feedback
- Physics-constrained optimization

---

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Input["Input"]
        Q["Query/Composition/Targets"]
    end

    subgraph Processing["Processing"]
        direction TB
        Intent["Intent<br/>Classification"]
        Retrieval["Knowledge<br/>Retrieval"]
        ML["ML<br/>Prediction"]
        Physics["Physics<br/>Validation"]
        LLM["LLM<br/>Generation"]

        Intent --> Retrieval
        Retrieval --> ML
        ML --> Physics
        Physics --> LLM
    end

    subgraph Output["Output"]
        R["Response/Properties/Design"]
    end

    Input --> Processing --> Output

    style Processing fill:#fff8e1
```

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Python |
| **Backend** | FastAPI, CrewAI, LangChain |
| **Databases** | GraphDB (RDF), Weaviate (Vector) |
| **ML** | XGBoost, Random Forest, scikit-learn |
| **LLM** | Groq API (Llama 3.3-70B) |
| **Physics** | Custom metallurgy models |
| **Deployment** | Docker, Docker Compose |
