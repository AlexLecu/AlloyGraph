# AlloyGraph Architecture Documentation

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph User["User Interface"]
        Browser["Web Browser"]
    end

    subgraph Frontend["Frontend Container (Port 3000)"]
        Nginx["Nginx Reverse Proxy"]
        Vue["Vue.js SPA"]

        subgraph Tabs["Application Tabs"]
            Chat["Research Chat"]
            Evaluate["Evaluate Tool"]
            Design["Design Tool"]
        end
    end

    subgraph Backend["Backend Container (Port 5001)"]
        Flask["Flask API"]

        subgraph Endpoints["API Endpoints"]
            ChatAPI["/api/chat"]
            ValidateAPI["/api/validate"]
            DesignAPI["/api/design"]
        end

        subgraph Services["Services Layer"]
            ChatService["Chat Service"]
            EvalCrew["AlloyEvaluationCrew"]
            DesignCrew["IterativeDesignCrew"]
        end

        subgraph Core["Core Components"]
            MLModels["ML Models<br/>(XGBoost + RF Ensemble)"]
            FeatureEng["Feature Engineering"]
            PhysicsTools["Physics Tools"]
            RAGTools["RAG Tools"]
        end
    end

    subgraph DataStores["Data Stores"]
        Weaviate["Weaviate<br/>Vector DB<br/>(146K+ variants)"]
        GraphDB["GraphDB<br/>RDF Triplestore<br/>(Ontology)"]
        T2V["T2V Transformers<br/>(Embeddings)"]
    end

    subgraph External["External Services"]
        Groq["LLM API — DeepInfra<br/>(Llama 3.3-70B, OpenAI-compatible)"]
    end

    Browser --> Nginx
    Nginx --> Vue
    Vue --> Chat & Evaluate & Design

    Nginx -->|"/api/*"| Flask

    ChatAPI --> ChatService
    ValidateAPI --> EvalCrew
    DesignAPI --> DesignCrew

    ChatService --> RAGTools
    ChatService --> Groq

    EvalCrew --> MLModels
    EvalCrew --> RAGTools
    EvalCrew --> PhysicsTools
    EvalCrew --> Groq

    DesignCrew --> MLModels
    DesignCrew --> RAGTools
    DesignCrew --> PhysicsTools
    DesignCrew --> Groq

    MLModels --> FeatureEng
    RAGTools --> Weaviate
    Weaviate --> T2V
    RAGTools -.-> GraphDB
```

## 2. Chat RAG Application Flow

```mermaid
flowchart TD
    subgraph Input["User Input"]
        Query["User Query"]
        History["Conversation History"]
    end

    subgraph IntentClassification["Intent Classification (LLM)"]
        Classify{"Classify Intent"}
        SEARCH["SEARCH<br/>Find specific alloys"]
        ANALYTICS["ANALYTICS<br/>Rankings/extrema"]
        TARGET["TARGET<br/>Find by value"]
        DESIGN["DESIGN<br/>Route to designer"]
        CONVERSATION["CONVERSATION<br/>Casual chat"]
    end

    subgraph Retrieval["Data Retrieval"]
        ExtractAlloy["Extract Alloy Names<br/>(LLM)"]
        WeaviateSearch["Weaviate Hybrid Search"]
        BestMatch["Find Best Match<br/>(Similarity Score)"]
        AnalyticsProcess["Process Analytics Query"]
        TargetProcess["Process Target Query"]
    end

    subgraph Response["Response Generation"]
        FormatContext["Format Alloy Data<br/>for LLM Context"]
        StreamAlloys["Stream Alloy Data<br/>(JSON)"]
        StreamLLM["Stream LLM Response<br/>(Chunks)"]
    end

    subgraph Output["Output to User"]
        AlloyCards["Alloy Cards<br/>(Composition, Properties)"]
        TextResponse["Natural Language<br/>Response"]
    end

    Query --> Classify
    History --> Classify

    Classify --> SEARCH
    Classify --> ANALYTICS
    Classify --> TARGET
    Classify --> DESIGN
    Classify --> CONVERSATION

    SEARCH --> ExtractAlloy
    ExtractAlloy --> WeaviateSearch
    WeaviateSearch --> BestMatch

    ANALYTICS --> AnalyticsProcess
    AnalyticsProcess --> WeaviateSearch

    TARGET --> TargetProcess
    TargetProcess --> WeaviateSearch

    BestMatch --> FormatContext
    AnalyticsProcess --> FormatContext
    TargetProcess --> FormatContext

    FormatContext --> StreamAlloys
    FormatContext --> StreamLLM

    StreamAlloys --> AlloyCards
    StreamLLM --> TextResponse

    CONVERSATION --> StreamLLM
    DESIGN -->|"Suggest Designer Tool"| TextResponse
```

## 3. Evaluation Tool Flow (Composition → Properties)

```mermaid
flowchart TD
    subgraph Input["User Input"]
        Composition["Alloy Composition<br/>(wt%)"]
        Temp["Temperature (°C)"]
        Processing["Processing Method<br/>(cast/wrought/forged)"]
    end

    subgraph Validation["Input Validation"]
        ValidateComp["Validate Composition<br/>(sum ≈ 100%)"]
        KGLookup["Knowledge Graph Lookup<br/>(Similar Alloys)"]
    end

    subgraph AgentPipeline["CrewAI Agent Pipeline (5 Agents)"]
        subgraph Task1["Task 1: Validation"]
            Validator["Validator Agent"]
            MLPredict["AlloyPredictorTool<br/>(XGBoost + RF)"]
        end

        subgraph Task2["Task 2: Arbitration"]
            Arbitrator["Arbitrator Agent"]
            DataFusion["DataFusionTool<br/>(ML + KG Fusion)"]
        end

        subgraph Task3["Task 3: Physics Audit"]
            Physicist["Physicist Agent"]
            MetallurgyVerify["MetallurgyVerifierTool"]
        end

        subgraph Task4["Task 4: Corrections"]
            Corrector["Corrector Agent"]
            PhysicsCorrections["PhysicsCorrectionsProposalTool"]
        end

        subgraph Task5["Task 5: Summary"]
            Summarizer["Summarizer Agent"]
        end
    end

    subgraph PostProcess["Post-Processing"]
        Calibration["Apply Calibration<br/>(Database-driven)"]
        PhysicsEnforce["Enforce Physics<br/>Constraints"]
        CoherencyCheck["Property Coherency<br/>Check"]
        Cleanup["Cleanup & Normalize<br/>Output"]
    end

    subgraph Output["Output"]
        Properties["Predicted Properties<br/>(YS, UTS, EL, EM)"]
        Intervals["Confidence Intervals"]
        TCPRisk["TCP Risk Assessment"]
        Metrics["Metallurgy Metrics<br/>(γ', Md, density)"]
        Explanation["Expert Explanation"]
    end

    Composition --> ValidateComp
    Temp --> ValidateComp
    Processing --> ValidateComp

    ValidateComp --> KGLookup
    KGLookup --> Validator

    Validator --> MLPredict
    MLPredict --> Arbitrator

    Arbitrator --> DataFusion
    DataFusion --> Physicist

    Physicist --> MetallurgyVerify
    MetallurgyVerify --> Corrector

    Corrector --> PhysicsCorrections
    PhysicsCorrections --> Summarizer

    Summarizer --> Calibration
    Calibration --> PhysicsEnforce
    PhysicsEnforce --> CoherencyCheck
    CoherencyCheck --> Cleanup

    Cleanup --> Properties
    Cleanup --> Intervals
    Cleanup --> TCPRisk
    Cleanup --> Metrics
    Cleanup --> Explanation
```

## 4. Design Tool Flow (Properties → Composition)

```mermaid
flowchart TD
    subgraph Input["User Input"]
        TargetProps["Target Properties<br/>(YS, UTS, EL, EM, Density, γ')"]
        DesignTemp["Temperature (°C)"]
        DesignProc["Processing Method"]
        MaxIter["Max Iterations"]
    end

    subgraph IterativeLoop["Iterative Design Loop"]
        subgraph Phase1["Phase 1: Synthesis"]
            Designer["Designer Agent"]
            NoveltyCheck["Novelty Check<br/>(KG Lookup)"]
            QuickPhysics["Quick Physics<br/>Pre-check"]
        end

        subgraph Phase2["Phase 2: Analysis"]
            DesignValidator["Validator Agent"]
            DesignArbitrator["Arbitrator Agent"]
            DesignPhysicist["Physicist Agent"]
            DesignCorrector["Corrector Agent"]
        end

        subgraph Phase3["Phase 3: Evaluation"]
            SuccessCheck{"Design<br/>Successful?"}
            FailureClassify["Classify Failures<br/>(TCP/Property/Physics)"]
            OptimizationAdvisor["Optimization Advisor<br/>Agent"]
        end

        subgraph Feedback["Feedback Generation"]
            BuildFeedback["Build Structured<br/>Feedback"]
            PriorityFocus["Determine Priority<br/>Focus"]
        end
    end

    subgraph PostDesign["Post-Design Processing"]
        DesignCalibration["Apply Calibration"]
        DesignPhysicsEnforce["Physics Enforcement"]
        DesignSummarizer["Summarizer Agent"]
    end

    subgraph Output["Output"]
        FinalComp["Designed Composition"]
        AchievedProps["Achieved Properties"]
        DesignStatus["Design Status<br/>(success/incomplete)"]
        Issues["Issues & Recommendations"]
        DesignExplanation["Technical Summary"]
    end

    TargetProps --> Designer
    DesignTemp --> Designer
    DesignProc --> Designer

    Designer --> NoveltyCheck
    NoveltyCheck --> QuickPhysics

    QuickPhysics -->|"Critical Issues"| OptimizationAdvisor
    QuickPhysics -->|"OK"| DesignValidator

    DesignValidator --> DesignArbitrator
    DesignArbitrator --> DesignPhysicist
    DesignPhysicist --> DesignCorrector

    DesignCorrector --> SuccessCheck

    SuccessCheck -->|"Yes"| DesignCalibration
    SuccessCheck -->|"No"| FailureClassify

    FailureClassify --> OptimizationAdvisor
    OptimizationAdvisor --> BuildFeedback
    BuildFeedback --> PriorityFocus
    PriorityFocus -->|"Next Iteration"| Designer

    DesignCalibration --> DesignPhysicsEnforce
    DesignPhysicsEnforce --> DesignSummarizer

    DesignSummarizer --> FinalComp
    DesignSummarizer --> AchievedProps
    DesignSummarizer --> DesignStatus
    DesignSummarizer --> Issues
    DesignSummarizer --> DesignExplanation
```

## 5. ML Model Architecture

```mermaid
flowchart LR
    subgraph Input["Input"]
        RawComp["Raw Composition<br/>(wt%)"]
    end

    subgraph FeatureEngineering["Feature Engineering"]
        BasicFeatures["Basic Features<br/>(element wt%)"]

        subgraph Derived["Derived Features"]
            MdGamma["Md_gamma<br/>(TCP stability)"]
            LatticeMismatch["Lattice Mismatch<br/>(coherency)"]
            GammaPrime["γ' Fraction<br/>(strength)"]
            Density["Density<br/>(rule of mixtures)"]
            Ratios["Element Ratios<br/>(Al/Ti, Cr/Co, etc.)"]
            SSS["Solid Solution<br/>Strengthening"]
        end
    end

    subgraph Ensemble["Voting Ensemble"]
        XGBoost["XGBoost<br/>Regressor"]
        RandomForest["Random Forest<br/>Regressor"]
        Voting["Soft Voting<br/>(Average)"]
    end

    subgraph Output["Predicted Properties"]
        YS["Yield Strength<br/>(R²=0.81)"]
        UTS["Tensile Strength<br/>(R²=0.76)"]
        EL["Elongation<br/>(R²=0.45)"]
        EM["Elastic Modulus<br/>(R²=0.50)"]
    end

    RawComp --> BasicFeatures
    BasicFeatures --> Derived

    MdGamma --> XGBoost & RandomForest
    LatticeMismatch --> XGBoost & RandomForest
    GammaPrime --> XGBoost & RandomForest
    Density --> XGBoost & RandomForest
    Ratios --> XGBoost & RandomForest
    SSS --> XGBoost & RandomForest

    XGBoost --> Voting
    RandomForest --> Voting

    Voting --> YS & UTS & EL & EM
```

## 6. Data Flow Overview

```mermaid
flowchart TB
    subgraph DataSources["Data Sources"]
        JSONL["Alloy Data<br/>(JSONL)"]
        OWL["Metallurgy Ontology<br/>(OWL)"]
    end

    subgraph Pipeline["Data Pipeline (One-time Init)"]
        BuildOntology["Build Ontology"]
        EnrichData["Enrich with<br/>Computed Features"]
        CreateSchema["Create Weaviate<br/>Schema"]
        Ingest["Ingest Data"]
    end

    subgraph Storage["Persistent Storage"]
        WeaviateStore["Weaviate<br/>146K+ Variants<br/>Vector Embeddings"]
        GraphDBStore["GraphDB<br/>RDF Triples<br/>Ontology Classes"]
    end

    subgraph Runtime["Runtime Queries"]
        HybridSearch["Hybrid Search<br/>(Vector + Keyword)"]
        SPARQL["SPARQL Queries<br/>(Structured)"]
        SimilaritySearch["Composition<br/>Similarity Search"]
    end

    JSONL --> EnrichData
    OWL --> BuildOntology

    BuildOntology --> GraphDBStore
    EnrichData --> CreateSchema
    CreateSchema --> Ingest
    Ingest --> WeaviateStore

    WeaviateStore --> HybridSearch
    WeaviateStore --> SimilaritySearch
    GraphDBStore --> SPARQL
```

## 7. Docker Services Architecture

```mermaid
flowchart TB
    subgraph Network["alloygraph-network"]
        subgraph FE["Frontend Service"]
            FEContainer["alloygraph-frontend<br/>:3000 → :80"]
            FENginx["Nginx"]
            FEVue["Vue.js App"]
        end

        subgraph BE["Backend Service"]
            BEContainer["alloygraph-backend<br/>:5001 (internal)"]
            BEFlask["Flask + Gunicorn"]
            BECrewAI["CrewAI Agents"]
        end

        subgraph PL["Pipeline Service"]
            PLContainer["alloygraph-pipeline<br/>(one-shot)"]
            PLScript["run_pipeline_docker.py"]
        end

        subgraph WV["Weaviate Service"]
            WVContainer["alloygraph-weaviate<br/>:8081 → :8080"]
            WVData[("weaviate_data<br/>volume")]
        end

        subgraph T2V["Transformers Service"]
            T2VContainer["alloygraph-transformers<br/>:8080 (internal)"]
            T2VModel["MiniLM-L6<br/>Embeddings"]
        end

        subgraph GDB["GraphDB Service"]
            GDBContainer["alloygraph-graphdb<br/>:7200"]
            GDBData[("graphdb_data<br/>volume")]
        end
    end

    FEContainer -->|"/api/*"| BEContainer
    BEContainer --> WVContainer
    BEContainer --> GDBContainer
    WVContainer --> T2VContainer
    PLContainer --> WVContainer
    PLContainer --> GDBContainer

    WVContainer --> WVData
    GDBContainer --> GDBData
```

---

## Summary Table

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Vue.js 3 + Nginx | User interface with 3 tabs |
| Backend API | Flask + Gunicorn | REST API endpoints |
| Agent Framework | CrewAI | Multi-agent orchestration |
| ML Models | XGBoost + Random Forest | Property prediction |
| Vector DB | Weaviate | Semantic search (146K+ variants) |
| Graph DB | GraphDB | RDF triplestore (ontology) |
| Embeddings | MiniLM-L6 | Text-to-vector conversion |
| LLM | Llama 3.3-70B via any OpenAI-compatible provider (currently DeepInfra) | Intent classification, response generation |

## Key Data Flows

1. **Chat RAG**: Query → Intent Classification → Weaviate Search → LLM Response
2. **Evaluation**: Composition → ML Prediction → KG Fusion → Physics Audit → Calibration
3. **Design**: Targets → Designer Agent → Validation Loop → Optimization → Final Composition
