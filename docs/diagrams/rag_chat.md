# RAG Chat System - Research Assistant

## Overview

The RAG (Retrieval-Augmented Generation) chat system allows users to query the alloy knowledge base using natural language. It uses intent classification to route queries appropriately.

## Mermaid Diagram - Full Detail

```mermaid
flowchart TB
    classDef user fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef llm fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef db fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef process fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px

    %% User Input
    User["👤 <b>USER QUERY</b><br/>─────────────<br/>'What is the composition<br/>of CMSX-4?'"]:::user

    %% Intent Classification
    subgraph Intent["🧠 INTENT CLASSIFICATION (LLM)"]
        direction TB
        IC["Groq LLM (Llama 3.3-70B)<br/>Temperature: 0.0"]:::llm

        subgraph Types["Query Types"]
            T1["<b>SEARCH</b><br/>Find specific alloys"]
            T2["<b>ANALYTICS</b><br/>Rankings, extrema"]
            T3["<b>TARGET</b><br/>Find by property value"]
            T4["<b>DESIGN</b><br/>Route to Designer"]
            T5["<b>CONVERSATION</b><br/>General chat"]
        end
    end

    %% Knowledge Graph
    subgraph KG["📚 KNOWLEDGE GRAPH"]
        direction LR
        Weaviate["<b>Weaviate</b><br/>Vector DB<br/>146K+ variants"]:::db
        GraphDB["<b>GraphDB</b><br/>RDF Triplestore<br/>Ontology"]:::db
    end

    %% Retrieval Process
    subgraph Retrieval["🔍 RETRIEVAL"]
        direction TB
        Extract["Extract alloy names<br/>(LLM)"]:::process
        Hybrid["Hybrid Search<br/>(vector + keyword)"]:::process
        Match["Best Match Selection<br/>(similarity score)"]:::process
        Format["Format for Context<br/>(composition, properties)"]:::process

        Extract --> Hybrid --> Match --> Format
    end

    %% Response Generation
    subgraph Response["💬 RESPONSE GENERATION"]
        direction TB
        Context["Context Assembly<br/>─────────────<br/>Retrieved alloy data<br/>Conversation history"]:::process
        LLM2["Groq LLM<br/>Temperature: 0.2"]:::llm
        Stream["Streaming Response"]:::process

        Context --> LLM2 --> Stream
    end

    %% Output
    subgraph Output["📤 OUTPUT TO USER"]
        direction LR
        Cards["<b>Alloy Cards</b><br/>Composition<br/>Properties<br/>Metrics"]:::output
        Text["<b>Natural Language</b><br/>Explanation<br/>Comparisons"]:::output
    end

    %% Flow
    User --> Intent
    Intent -->|SEARCH| Retrieval
    Intent -->|ANALYTICS| Retrieval
    Intent -->|TARGET| Retrieval
    Intent -->|DESIGN| DesignRoute["Route to<br/>Design Tool"]
    Intent -->|CONVERSATION| Response

    Retrieval --> KG
    KG --> Retrieval
    Retrieval --> Response
    Response --> Output

    style Intent fill:#fff8e1,stroke:#ff8f00
    style KG fill:#ede7f6,stroke:#5e35b1
    style Retrieval fill:#e3f2fd,stroke:#1565c0
    style Response fill:#e8f5e9,stroke:#2e7d32
```

---

## Paper Version (Simplified)

```mermaid
flowchart LR
    Query["User Query"]

    subgraph RAG["<b>RAG Pipeline</b>"]
        direction TB
        Intent["Intent<br/>Classification"]

        subgraph Retrieve["Retrieval"]
            Search["Hybrid Search"]
            KG["Knowledge Graph<br/>(146K alloys)"]
            Search <--> KG
        end

        Generate["Response<br/>Generation"]

        Intent --> Retrieve --> Generate
    end

    Output["Alloy Cards +<br/>Explanation"]

    Query --> RAG --> Output

    style Intent fill:#fff3e0
    style Retrieve fill:#e3f2fd
    style Generate fill:#e8f5e9
```

---

## Intent Types & Examples

| Intent | Example Query | Action |
|--------|---------------|--------|
| **SEARCH** | "What is CMSX-4?" | Find specific alloy in KG |
| **ANALYTICS** | "Which alloy has highest YS?" | Rank alloys by property |
| **TARGET** | "Find alloys with YS > 1000 MPa" | Filter by property value |
| **DESIGN** | "Design an alloy for turbine blades" | Route to Design Pipeline |
| **CONVERSATION** | "What is gamma prime?" | General LLM response |

## Data Flow Example

```
User: "What is the composition of CMSX-4?"
    ↓
Intent Classification → SEARCH
    ↓
Extract alloy name → "CMSX-4"
    ↓
Weaviate hybrid search → Find best match
    ↓
Retrieve: composition, properties, metrics
    ↓
Format context for LLM
    ↓
Generate response with alloy card
    ↓
User sees: Composition table + explanation
```
