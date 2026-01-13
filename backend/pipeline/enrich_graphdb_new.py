import ast
import json
import logging
import os
import uuid

from pathlib import Path
from typing import Optional, Dict, Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
import requests

from backend.alloy_crew.models.feature_engineering import compute_alloy_features

# =============================================================================
# Configuration
# =============================================================================

GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
REPO_ID = os.getenv("GRAPHDB_REPO", "NiSuperAlloy")
JSON_FILE = os.getenv("ALLOY_JSON", "../superalloy_preprocess/output_data/all_alloys.jsonl")
print(JSON_FILE)

# =============================================================================
# AlloyMind URI policy
# =============================================================================

BASE = "https://w3id.org/alloygraph/"

# Ontology (TBox) - Classes & Properties
ONTOLOGY_BASE = f"{BASE}ont#"
NS = Namespace(ONTOLOGY_BASE)

# Resources (ABox) - Data Instances
RESOURCE_BASE = f"{BASE}res/"
RES = Namespace(RESOURCE_BASE)

# Named graph - Dataset Container
NAMED_GRAPH = URIRef(f"{BASE}data/alloys")

# --- EMMO / EMBO / QUDT namespaces -----------------------------------------
EMMO = Namespace("http://emmo.info/emmo#")
EMBO = Namespace("http://emmo.info/emmo/domain/emo#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")

# ChEBI for chemical elements
CHEBI = Namespace("http://purl.obolibrary.org/obo/CHEBI_")

ELEMENT_MAP = {
    "Ni": CHEBI.CHEBI_28112,  # nickel atom
    "Cr": CHEBI.CHEBI_28073,
    "Mo": CHEBI.CHEBI_28685,
    "Nb": CHEBI.CHEBI_33345,
    "Al": CHEBI.CHEBI_28938,
    "Ti": CHEBI.CHEBI_28948,
    "C":  CHEBI.CHEBI_27594,
    "B":  CHEBI.CHEBI_27563,
    "Zr": CHEBI.CHEBI_33332,
    "Co": CHEBI.CHEBI_27638,
    "W":  CHEBI.CHEBI_27998,
    "Ta": CHEBI.CHEBI_33348,
    "Re": CHEBI.CHEBI_30189,
    "Hf": CHEBI.CHEBI_33343,
    "Fe": CHEBI.CHEBI_18248,
    "Mn": CHEBI.CHEBI_18291,
    "Si": CHEBI.CHEBI_27573,
    "Cu": CHEBI.CHEBI_28694,
    "V":  CHEBI.CHEBI_27698,
}

PROPERTY_MAP: Dict[str, str] = {
    "yield_strength": "YieldStrength",
    "uts": "UTS",
    "elongation": "Elongation",
    "elasticity": "Elasticity"
}

UNIT_MAP = {
    "MPa": UNIT.MegaPA,
    "GPa": UNIT.GigaPA,
    "%": UNIT.PERCENT,
    "°C": UNIT.DEG_C,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
log = logging.getLogger("json->graphdb")


def iri_local(s: str) -> str:
    """Normalize string for safe URI local part."""
    out = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in s.strip())
    if out and out[0].isdigit():
        out = "_" + out
    return out


# ---------------------------------------------------------------------------
# Ontology (TBox): classes & properties ONLY
# ---------------------------------------------------------------------------

def mint_class(local_name: str) -> URIRef:
    """
    Mint an ontology class URI.
    Example: NickelBasedSuperalloy
    """
    return NS[iri_local(local_name)]


def mint_property(local_name: str) -> URIRef:
    """
    Mint an ontology property URI.
    Example: hasGammaPrimeFraction
    """
    return NS[iri_local(local_name)]


# ---------------------------------------------------------------------------
# Resources (ABox): individuals ONLY
# ---------------------------------------------------------------------------

def mint_res(path: str) -> URIRef:
    """
    Mint a resource URI using a path-like structure.

    Examples:
      alloy/Alloy713C
      variant/Alloy713C_cast
      quantity/Alloy713C_cast_GP
    """
    return RES[path]


def mint_res_uuid(path: str) -> URIRef:
    """
    Mint a resource URI with a UUID suffix (use ONLY when needed).
    """
    return RES[f"{path}_{uuid.uuid4().hex[:8]}"]


def add_quantity(g: Graph, unit: Optional[str], data: Dict[str, Any]) -> URIRef:
    q = mint_res_uuid(f"quantity/{iri_local(unit or 'quantity')}")
    g.add((q, RDF.type, mint_class("Quantity")))
    if unit:
        g.add((q, mint_property("unitSymbol"), Literal(unit)))

    min_v = data.get("min")
    max_v = data.get("max")
    if (min_v is not None) and (max_v is not None) and (float(min_v) > float(max_v)):
        min_v, max_v = float(max_v), float(min_v)

    if data.get("value") is not None:
        g.add((q, mint_property("numericValue"), Literal(float(data["value"]), datatype=XSD.decimal)))
    if min_v is not None:
        g.add((q, mint_property("minInclusive"), Literal(float(min_v), datatype=XSD.decimal)))
    if max_v is not None:
        g.add((q, mint_property("maxInclusive"), Literal(float(max_v), datatype=XSD.decimal)))
    if data.get("approx"):
        g.add((q, mint_property("isApproximate"), Literal(True, datatype=XSD.boolean)))
    if data.get("qualifier"):
        g.add((q, mint_property("qualifier"), Literal(data["qualifier"])))
    if data.get("raw"):
        g.add((q, mint_property("rawString"), Literal(data["raw"])))

    return q


def add_comp_entry(g: Graph, comp_uri: URIRef, elem_uri: URIRef, data: Dict[str, Any], alloy_name: str):
    elem_name = Path(str(elem_uri)).name.replace("Element_", "")
    entry = mint_res(f"composition-entry/{iri_local(alloy_name)}_{iri_local(elem_name)}")
    g.add((entry, RDF.type, mint_class("CompositionEntry")))
    g.add((comp_uri, mint_property("hasComponent"), entry))
    g.add((entry, mint_property("element"), elem_uri))

    if isinstance(data, (int, float)):
        data = {"value": data}
        
    if data.get("is_balance_remainder"):
        g.add((entry, mint_property("isBalanceRemainder"), Literal(True, datatype=XSD.boolean)))
    else:
        q_data = {}
        if data.get("value") is not None:
            q_data["value"] = data["value"]
        if data.get("min") is not None:
            q_data["min"] = data["min"]
        if data.get("max") is not None:
            q_data["max"] = data["max"]
        if data.get("qualifier"):
            q_data["qualifier"] = data["qualifier"]
        if data.get("raw"):
            q_data["raw"] = data["raw"]
        if data.get("approx"):
            q_data["approx"] = data["approx"]

        if q_data:
            q_suffix = f"{alloy_name}_{elem_name}_Mass"

            q = mint_res(f"quantity/{iri_local(q_suffix)}")
            g.add((q, RDF.type, mint_class("Quantity")))
            g.add((entry, mint_property("hasMassFraction"), q))

            unit = data.get("unit", "%")
            if unit:
                g.add((q, mint_property("unitSymbol"), Literal(unit)))

            if q_data.get("value") is not None:
                g.add((q, mint_property("numericValue"), Literal(float(q_data["value"]), datatype=XSD.decimal)))
            if q_data.get("min") is not None:
                g.add((q, mint_property("minInclusive"), Literal(float(q_data["min"]), datatype=XSD.decimal)))
            if q_data.get("max") is not None:
                g.add((q, mint_property("maxInclusive"), Literal(float(q_data["max"]), datatype=XSD.decimal)))
            if q_data.get("approx"):
                g.add((q, mint_property("isApproximate"), Literal(True, datatype=XSD.boolean)))
            if q_data.get("qualifier"):
                g.add((q, mint_property("qualifier"), Literal(q_data["qualifier"])))
            if q_data.get("raw"):
                g.add((q, mint_property("rawString"), Literal(q_data["raw"])))


def build_graph(json_path: str) -> Graph:
    log.info("Reading JSON: %s", json_path)

    g = Graph()
    g.bind("ns", NS)
    g.bind("res", RES)
    g.bind("emmo", EMMO)
    g.bind("embo", EMBO)
    g.bind("qudt", QUDT)
    g.bind("unit", UNIT)
    g.bind("chebi", CHEBI)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    alloys = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith("==>"):
                try:
                    alloys.append(json.loads(line))
                except json.JSONDecodeError:
                    try:
                        line_fixed = line.replace("null", "None").replace("true", "True").replace("false", "False")
                        alloys.append(ast.literal_eval(line_fixed))
                    except (ValueError, SyntaxError):
                        log.warning(f"Skipping invalid JSON line: {line[:50]}...")

    primary_compositions = {}
    for alloy_data in alloys:
        name = alloy_data.get("alloy")
        comp = alloy_data.get("composition")
        if name and comp and name not in primary_compositions:
            primary_compositions[name] = comp

    elements_seen = set()

    for alloy_data in alloys:
        alloy_name = alloy_data.get("alloy", "")
        if not alloy_name: continue

        processing = alloy_data.get("processing")
        form_val = alloy_data.get("form")

        a_uri = mint_res(f"alloy/{iri_local(alloy_name)}")
        g.add((a_uri, RDF.type, mint_class("NickelBasedSuperalloy")))
        g.add((a_uri, RDFS.label, Literal(alloy_name)))
        g.add((a_uri, mint_property("tradeDesignation"), Literal(alloy_name)))

        uns = alloy_data.get("uns", "")
        if uns:
            g.add((a_uri, mint_property("unsNumber"), Literal(uns)))

        if alloy_data.get("family"):
            g.add((a_uri, mint_property("family"), Literal(alloy_data["family"])))

        parts = [alloy_name]
        if processing: parts.append(processing)
        if form_val: parts.append(form_val)
        unique_token = "_".join(parts)
        
        v_uri = mint_res(f"variant/{iri_local(unique_token)}")
        g.add((v_uri, RDF.type, mint_class("Variant")))
        g.add((v_uri, RDFS.label, Literal(unique_token.replace("_", " "))))

        g.add((a_uri, mint_property("hasVariant"), v_uri))

        if processing:
            g.add((v_uri, mint_property("processingMethod"), Literal(processing)))
            pm_uri = mint_res(f"method/{iri_local(processing)}")
            g.add((pm_uri, RDF.type, mint_class("ProcessingMethod")))
            g.add((pm_uri, RDFS.label, Literal(processing)))
            g.add((v_uri, mint_property("hasProcessingMethod"), pm_uri))

        if form_val:
            f_uri = mint_res(f"form/{iri_local(form_val)}")
            g.add((f_uri, RDF.type, mint_class("Form")))
            g.add((f_uri, RDFS.label, Literal(form_val)))
            g.add((f_uri, mint_property("form"), Literal(form_val)))
            g.add((v_uri, mint_property("hasForm"), f_uri))

        if alloy_data.get("density_gcm3"):
            g.add((v_uri, mint_property("density"), Literal(float(alloy_data["density_gcm3"]), datatype=XSD.decimal)))
        if alloy_data.get("gamma_prime_vol_pct") is not None:
            g.add((v_uri, mint_property("gammaPrimeVolPct"), Literal(float(alloy_data["gamma_prime_vol_pct"]), datatype=XSD.decimal)))
        if alloy_data.get("typical_heat_treatment"):
            g.add((v_uri, mint_property("typicalHeatTreatment"), Literal(alloy_data["typical_heat_treatment"])))

        try:
            computed = compute_alloy_features(alloy_data)

            log.info(f"[{alloy_name}] Computed Md_avg: {computed.get('Md_avg')}, TCP: {computed.get('TCP_risk')}")

            if "Md_avg" in computed:
                g.add((v_uri, mint_property("hasMdAverage"), Literal(float(computed["Md_avg"]), datatype=XSD.decimal)))

            if "gamma_prime_estimated_vol_pct" in computed:
                g.add((v_uri, mint_property("hasGammaPrimeEstimate"), Literal(float(computed["gamma_prime_estimated_vol_pct"]), datatype=XSD.decimal)))

            if "density_calculated_gcm3" in computed:
                g.add((v_uri, mint_property("hasDensityCalculated"), Literal(float(computed["density_calculated_gcm3"]), datatype=XSD.decimal)))

            if "TCP_risk" in computed:
                g.add((v_uri, mint_property("hasTcpRisk"), Literal(computed["TCP_risk"])))

            mapping = {
                "SSS_total_wt_pct": "hasSSSTotalWtPct",
                "refractory_total_wt_pct": "hasRefractoryTotalWtPct",
                "GP_formers_wt_pct": "hasGPFormersWtPct",
                "Al_Ti_ratio": "hasAlTiRatio",
                "Cr_Co_ratio": "hasCrCoRatio",
                "Cr_Ni_ratio": "hasCrNiRatio",
                "Mo_W_ratio": "hasMoWRatio",
                "Al_Ti_at_ratio": "hasAlTiAtRatio",
                "GP_formers_at_pct": "hasGPFormersAtPct",
            }
            
            for key, prop_name in mapping.items():
                if key in computed:
                    g.add((v_uri, mint_property(prop_name), Literal(float(computed[key]), datatype=XSD.decimal)))

            if "atomic_percent" in computed:
                ap_json = json.dumps(computed["atomic_percent"])
                g.add((v_uri, mint_property("hasAtomicCompositionJson"), Literal(ap_json)))

        except Exception as e:
            log.warning(f"Failed to compute features for {alloy_name}: {e}")

        composition = alloy_data.get("composition", {})
        has_own_comp = bool(composition)

        if not has_own_comp:
            composition = primary_compositions.get(alloy_name, {})

        if has_own_comp:
            c_uri = mint_res(f"composition/{iri_local(unique_token)}")
            comp_seed = unique_token
        else:
            c_uri = mint_res(f"composition/{iri_local(alloy_name)}")
            comp_seed = alloy_name

        g.add((c_uri, RDF.type, mint_class("Composition")))
        g.add((v_uri, mint_property("hasComposition"), c_uri))

        others = alloy_data.get("other_constituents")
        if others:
            g.add((c_uri, mint_property("otherConstituents"), Literal(others)))

        for elem_symbol, elem_data in composition.items():
            if elem_symbol == "other": continue
            
            if elem_symbol not in elements_seen:
                e_uri = mint_res(f"element/{iri_local(elem_symbol)}")
                g.add((e_uri, RDF.type, mint_class("Element")))
                g.add((e_uri, RDFS.label, Literal(elem_symbol)))
                elements_seen.add(elem_symbol)

            add_comp_entry(g, c_uri, mint_res(f"element/{iri_local(elem_symbol)}"), elem_data, comp_seed)

        for prop_key, prop_class in PROPERTY_MAP.items():
            measurements = alloy_data.get(prop_key, [])
            if isinstance(measurements, str):
                try:
                    measurements = ast.literal_eval(measurements)
                except (ValueError, SyntaxError):
                    log.warning(f"Failed to parse measurements for {prop_key}: {measurements[:50]}...")
                    continue

            if not measurements:
                continue

            propset_uri = mint_res(f"property-set/{iri_local(unique_token)}_{iri_local(prop_class)}")
            g.add((propset_uri, RDF.type, mint_class("PropertySet")))
            g.add((v_uri, mint_property("hasPropertySet"), propset_uri))
            g.add((propset_uri, mint_property("measuresProperty"), mint_class(prop_class)))

            for meas_data in measurements:
                meas = mint_res_uuid(f"measurement/{iri_local(unique_token)}_{iri_local(prop_class)}")
                g.add((meas, RDF.type, mint_class("Measurement")))
                g.add((propset_uri, mint_property("hasMeasurement"), meas))

                temp_c = meas_data.get("temp_c")
                temp_category = meas_data.get("temp_category")
                if temp_category:
                    g.add((meas, mint_property("temperatureCategory"), Literal(temp_category)))
                if temp_c is not None:
                    temp_qty = add_quantity(g, "°C", {"value": temp_c})
                    g.add((meas, mint_property("hasTestTemperature"), temp_qty))

                if "stress_mpa" in meas_data:
                    g.add((meas, mint_property("stress"), Literal(float(meas_data["stress_mpa"]), datatype=XSD.decimal)))
                if "life_hours" in meas_data:
                    g.add((meas, mint_property("lifeHours"), Literal(float(meas_data["life_hours"]), datatype=XSD.decimal)))

                q_data = {}
                if meas_data.get("value") is not None:
                    q_data["value"] = meas_data["value"]
                if meas_data.get("min") is not None:
                    q_data["min"] = meas_data["min"]
                if meas_data.get("max") is not None:
                     q_data["max"] = meas_data["max"]
                if meas_data.get("qualifier"):
                    q_data["qualifier"] = meas_data["qualifier"]
                if meas_data.get("raw"):
                    q_data["raw"] = meas_data["raw"]
                if meas_data.get("approx"):
                    q_data["approx"] = meas_data["approx"]

                unit = meas_data.get("unit", "")
                if not unit:
                    if "_mpa" in prop_key: unit = "MPa"
                    elif "_pct" in prop_key: unit = "%"
                    elif "_gpa" in prop_key: unit = "GPa"
                    elif prop_key == "yield_strength": unit = "MPa"
                    elif prop_key == "uts": unit = "MPa"
                    elif prop_key == "elongation": unit = "%"
                    elif prop_key == "elasticity": unit = "GPa"
                    elif prop_key == "hardness" and "scale" in meas_data:
                        unit = meas_data["scale"]

                q_uri = add_quantity(g, unit, q_data)
                g.add((meas, mint_property("hasQuantity"), q_uri))

    log.info("Graph built with %d triples", len(g))
    return g


def upload_graph(g: Graph):
    ttl_bytes = g.serialize(format="turtle").encode("utf-8")
    log.info("Uploading %d triples directly to GraphDB…", len(g))

    endpoint = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    params = {"context": f"<{NAMED_GRAPH}>"}
    headers = {"Content-Type": "text/turtle; charset=UTF-8"}

    # Clear graph first to prevent duplication
    log.info("Clearing graph %s...", NAMED_GRAPH)
    del_endpoint = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    del_params = {"context": f"<{NAMED_GRAPH}>"}
    requests.delete(del_endpoint, params=del_params)

    resp = requests.post(endpoint, params=params, data=ttl_bytes, headers=headers, timeout=120)
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")
    log.info("✅ Uploaded to %s (repo %s)", NAMED_GRAPH, REPO_ID)


ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE", "../Data/Ontology/NiSuperAlloy_Ont_GEN.rdf")

def create_repo_if_missing():
    """Checks if the repository exists, and if not, creates it."""
    # Check if repo exists
    resp = requests.get(f"{GRAPHDB_URL}/repositories/{REPO_ID}", headers={"Accept": "application/json"})
    if resp.status_code == 200:
        log.info(f"Repository '{REPO_ID}' already exists.")
        return
    elif resp.status_code != 404:
        log.warning(f"Unexpected status code checking for repo: {resp.status_code}. Assuming it doesn't exist.")
    
    log.info(f"Repository '{REPO_ID}' not found. Creating it...")
    
    # Repository Configuration
    repo_config_path = Path(__file__).parent / "repo-config.ttl"
    if not repo_config_path.exists():
         raise RuntimeError(f"Repository config file not found: {repo_config_path}")

    with open(repo_config_path, "r") as f:
        repo_config_template = f.read()
    
    # Replace placeholder if needed, or ensure ID matches. 
    # Since we hardcoded "NiSuperAlloy" in the ttl file, we can just use it.
    # If REPO_ID is dynamic, we should replace it.
    repo_config = repo_config_template.replace('repositoryID> "NiSuperAlloy"', f'repositoryID> "{REPO_ID}"')

    # Prepare multipart/form-data request
    files = {
        'config': ('config.ttl', repo_config, 'text/turtle')
    }
    
    create_resp = requests.post(f"{GRAPHDB_URL}/rest/repositories", files=files)
    
    if create_resp.status_code // 100 != 2:
        # If it failed because it already exists (race condition or check failure), just log it
        if create_resp.status_code == 400 and "already exists" in create_resp.text:
             log.info(f"Repository '{REPO_ID}' already exists (caught 400).")
             return
        raise RuntimeError(f"Failed to create repository: {create_resp.status_code} {create_resp.text}")
        
    log.info(f"✅ Repository '{REPO_ID}' created successfully.")


def upload_ontology():
    if not Path(ONTOLOGY_FILE).exists():
        log.warning(f"Ontology file not found: {ONTOLOGY_FILE}. Skipping ontology upload.")
        return

    log.info(f"Reading Ontology: {ONTOLOGY_FILE}")
    with open(ONTOLOGY_FILE, "rb") as f:
        rdf_data = f.read()

    log.info(f"Uploading Ontology ({len(rdf_data)} bytes) to GraphDB...")
    endpoint = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    params = {"context": f"<{NAMED_GRAPH}>"}
    headers = {"Content-Type": "application/rdf+xml; charset=UTF-8"}

    resp = requests.post(endpoint, params=params, data=rdf_data, headers=headers, timeout=120)
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Ontology upload failed: {resp.status_code} {resp.text}")
    log.info("✅ Ontology uploaded successfully")


def main():
    # Ensure Repo Exists
    create_repo_if_missing()

    # Upload Ontology first
    upload_ontology()

    if not Path(JSON_FILE).exists():
        raise SystemExit(f"Missing JSON file: {JSON_FILE}")
    g = build_graph(JSON_FILE)
    upload_graph(g)


if __name__ == "__main__":
    main()