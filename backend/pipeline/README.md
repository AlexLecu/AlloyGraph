# AlloyGraph Data Pipeline

This pipeline automates the ingestion of superalloy data into the Knowledge Graph (GraphDB) and Vector Database (Weaviate).

## How It Works

The pipeline performs the following steps sequentially:
1.  **Environment Setup**: Creates a virtual environment and installs dependencies.
2.  **Reset Weaviate**: Restarts the Weaviate Docker container to ensure a clean state.
3.  **Enrich Data**: Runs `enrich_jsonl_with_features.py` to compute metallurgical features (e.g., density, Phase stability) and saves to `final_enriched.jsonl`.
4.  **Build Ontology**: Generates the OWL ontology schema (`AlloyGraph_Ont_GEN.rdf`).
5.  **Enrich GraphDB**: Creates the repository (if missing) and uploads both the ontology and the JSONL data to GraphDB.
6.  **Ingest to Weaviate**: Creates the Weaviate schema and imports the enriched data from GraphDB.

## Configuration
The pipeline uses environment variables which can be overridden:
- `GRAPHDB_URL`: Default `http://localhost:7200`
- `GRAPHDB_REPO`: Default `AlloyGraph`
- `WEAVIATE_HOST`: Default `localhost`
- `WEAVIATE_PORT`: Default `8081`

## Prerequisites

Before running the pipeline, ensure you have:
-   **Docker** running (for Weaviate).
-   **GraphDB** running locally at `http://localhost:7200`.
-   **Python 3.10+** installed.

## How to Run

Simply execute the automation script from this directory:

```bash
./run_pipeline.sh
```

The script will handle all setup and execution automatically.

## After Running

Once the pipeline completes successfully:
1.  **Verify GraphDB**: Check `http://localhost:7200` to see the `AlloyGraph` repository populated with data.
2.  **Run the Chatbot**: You can now start the RAG application to query the data.
