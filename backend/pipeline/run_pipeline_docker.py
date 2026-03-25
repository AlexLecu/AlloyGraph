import os
import sys
import time
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Configuration from environment
GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://graphdb:7200")
GRAPHDB_REPO = os.getenv("GRAPHDB_REPO", "AlloyGraph")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")

# Data paths
DATA_DIR = "/app/alloy_crew/models/training_data"
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "train_77alloys.jsonl")
ONTOLOGY_OUTPUT = "/app/ontology/alloygraph.owl"


def wait_for_service(url: str, name: str, max_attempts: int = 60) -> bool:
    """Wait for a service to become available."""
    log.info(f"Waiting for {name}...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                log.info(f"  ✓ {name} is ready")
                return True
        except requests.exceptions.RequestException:
            pass

        if attempt % 10 == 0 and attempt > 0:
            log.info(f"  Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)

    log.error(f"  ✗ {name} failed to start")
    return False


def check_data_exists() -> bool:
    """Check if data is already loaded in both databases."""
    log.info("Checking if data already exists...")

    # Check GraphDB
    try:
        response = requests.get(
            f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPO}/size",
            headers={"Accept": "text/plain"},
            timeout=10
        )
        if response.status_code == 200:
            count = int(response.text.strip())
            if count > 100:  # More than just the ontology
                log.info(f"  GraphDB has {count} triples")
            else:
                log.info(f"  GraphDB is empty or minimal ({count} triples)")
                return False
        else:
            log.info("  GraphDB repository not found")
            return False
    except Exception as e:
        log.info(f"  GraphDB check failed: {e}")
        return False

    # Check Weaviate
    try:
        import weaviate
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=int(WEAVIATE_PORT),
            grpc_port=int(WEAVIATE_GRPC_PORT)
        )
        try:
            collection = client.collections.get("Variant")
            result = collection.aggregate.over_all(total_count=True)
            count = result.total_count
            if count > 0:
                log.info(f"  Weaviate has {count} variants")
                log.info("  ✓ Data already loaded - skipping pipeline")
                return True
            else:
                log.info("  Weaviate Variant collection is empty")
                return False
        except Exception:
            log.info("  Weaviate Variant collection not found")
            return False
        finally:
            client.close()
    except Exception as e:
        log.info(f"  Weaviate check failed: {e}")
        return False


def create_graphdb_repository() -> bool:
    """Create GraphDB repository if it doesn't exist."""
    log.info("Setting up GraphDB repository...")

    repos_url = f"{GRAPHDB_URL}/rest/repositories"
    try:
        response = requests.get(repos_url)
        repos = response.json()

        if any(r.get("id") == GRAPHDB_REPO for r in repos):
            log.info(f"  Repository '{GRAPHDB_REPO}' exists")
            return True

        log.info(f"  Creating repository '{GRAPHDB_REPO}'...")

        config = f"""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep: <http://www.openrdf.org/config/repository#> .
@prefix sr: <http://www.openrdf.org/config/repository/sail#> .
@prefix sail: <http://www.openrdf.org/config/sail#> .
@prefix graphdb: <http://www.ontotext.com/config/graphdb#> .

[] a rep:Repository ;
    rep:repositoryID "{GRAPHDB_REPO}" ;
    rdfs:label "AlloyGraph Knowledge Graph" ;
    rep:repositoryImpl [
        rep:repositoryType "graphdb:SailRepository" ;
        sr:sailImpl [
            sail:sailType "graphdb:Sail" ;
            graphdb:ruleset "rdfs" ;
            graphdb:check-for-inconsistencies "true" ;
            graphdb:entity-index-size "10000000" ;
            graphdb:enableFtsIndex "true" ;
        ]
    ] .
"""

        response = requests.post(
            repos_url,
            files={"config": ("config.ttl", config, "text/turtle")},
        )

        if response.status_code in (200, 201, 204):
            log.info(f"  ✓ Repository created")
            return True
        else:
            log.error(f"  ✗ Failed: {response.text}")
            return False

    except Exception as e:
        log.error(f"  ✗ Error: {e}")
        return False


def find_data_file() -> str:
    """Find the best available data file."""
    candidates = [
        os.getenv("ALLOY_JSON"),
        DEFAULT_DATA_FILE,
        "/app/alloy_crew/models/training_data/final_alloy_data_enriched.jsonl",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    # List available files for debugging
    log.error("No data file found. Available files:")
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.jsonl'):
                log.error(f"  - {f}")
    return None


def run_pipeline(data_file: str):
    """Run the full data pipeline."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(ONTOLOGY_OUTPUT), exist_ok=True)

    # Step 1: Build ontology
    log.info("[1/4] Building ontology...")
    os.environ["ONTO_OUT"] = ONTOLOGY_OUTPUT
    os.environ["SAVE_ONTOLOGY"] = "1"
    from pipeline.build_ontology import main as build_main
    build_main()

    # Step 2: Load into GraphDB
    log.info("[2/4] Loading data into GraphDB...")
    os.environ["ALLOY_JSON"] = data_file
    os.environ["ONTOLOGY_FILE"] = ONTOLOGY_OUTPUT
    from pipeline.enrich_graphdb import main as enrich_main
    enrich_main()

    # Step 3: Create Weaviate schema
    log.info("[3/4] Creating Weaviate schema...")
    from pipeline.weaviate_schema import main as schema_main
    schema_main()

    # Step 4: Ingest into Weaviate
    log.info("[4/4] Ingesting data into Weaviate...")
    from pipeline.weaviate_ingest import main as ingest_main
    ingest_main()


def main():
    print()
    print("=" * 55)
    print("   AlloyGraph Data Pipeline")
    print("=" * 55)
    print()

    # Wait for services
    if not wait_for_service(f"{GRAPHDB_URL}/rest/repositories", "GraphDB"):
        sys.exit(1)

    weaviate_url = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}/v1/.well-known/ready"
    if not wait_for_service(weaviate_url, "Weaviate"):
        sys.exit(1)

    # Create repository (idempotent)
    if not create_graphdb_repository():
        sys.exit(1)

    # Check if data already exists
    if check_data_exists():
        print()
        print("=" * 55)
        print("   Data already loaded - Ready!")
        print("=" * 55)
        sys.exit(0)

    # Find data file
    data_file = find_data_file()
    if not data_file:
        log.error("Cannot proceed without data file")
        sys.exit(1)

    log.info(f"Using data file: {data_file}")
    print()

    # Run pipeline
    run_pipeline(data_file)

    print()
    print("=" * 55)
    print("   Pipeline Complete!")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
