import ast
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
try:
    from backend.alloy_crew.models.feature_engineering import compute_alloy_features
except ModuleNotFoundError:
    from alloy_crew.models.feature_engineering import compute_alloy_features

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD, OWL
import requests


GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
REPO_ID = os.getenv("GRAPHDB_REPO", "AlloyGraph")
JSON_FILE = os.getenv("ALLOY_JSON", "../alloy_crew/models/training_data/train_77alloys.jsonl")
ONTOLOGY_FILE = os.getenv("ONTOLOGY_FILE", "../Data/Ontology/alloygraph.owl")

BASE = "https://w3id.org/alloygraph/"
ONTOLOGY_BASE = f"{BASE}ont#"
NS = Namespace(ONTOLOGY_BASE)
RESOURCE_BASE = f"{BASE}res/"
RES = Namespace(RESOURCE_BASE)
ONTOLOGY_GRAPH = URIRef(f"{BASE}ont")
DATA_GRAPH = URIRef(f"{BASE}data/alloys")

QUDT_UNIT = Namespace("http://qudt.org/vocab/unit/")
CHEBI = Namespace("http://purl.obolibrary.org/obo/")

CHEBI_ELEMENT_MAP = {
    "Ni": "CHEBI_28112", "Cr": "CHEBI_28073", "Co": "CHEBI_27638",
    "Mo": "CHEBI_28685", "W": "CHEBI_27998", "Al": "CHEBI_28938",
    "Ti": "CHEBI_28948", "Ta": "CHEBI_33348", "Nb": "CHEBI_33345",
    "Re": "CHEBI_30189", "Hf": "CHEBI_33343", "Fe": "CHEBI_18248",
    "C": "CHEBI_27594", "B": "CHEBI_27563", "Zr": "CHEBI_33332",
    "Mn": "CHEBI_18291", "Si": "CHEBI_27573", "Cu": "CHEBI_28694",
    "V": "CHEBI_27698", "Ru": "CHEBI_30682",
}

PROPERTY_MAP: Dict[str, str] = {
    "yield_strength": "YieldStrength",
    "uts": "UTS",
    "elongation": "Elongation",
    "elasticity": "Elasticity"
}

UNIT_MAP = {
    "MPa": QUDT_UNIT.MegaPA,
    "GPa": QUDT_UNIT.GigaPA,
    "%": QUDT_UNIT.PERCENT,
    "°C": QUDT_UNIT.DEG_C,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("json-to-graphdb")


def safe_iri(s: str) -> str:
    """Normalize string for safe URI local part."""
    out = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in s.strip())
    if not out:
        return "_empty"
    return "_" + out if out[0].isdigit() else out


def mint_res(path: str) -> URIRef:
    """Mint a deterministic resource URI."""
    return RES[path]


def mint_res_uuid(prefix: str) -> URIRef:
    """Mint a resource URI with UUID for non-deterministic resources."""
    return RES[f"{prefix}_{uuid.uuid4().hex[:8]}"]


def add_quantity(g: Graph, unit: Optional[str], data: Dict[str, Any]) -> URIRef:
    """Create a Quantity resource with value, range, and unit."""
    q_uri = mint_res_uuid(f"quantity/{safe_iri(unit or 'q')}")
    g.add((q_uri, RDF.type, NS.Quantity))

    if unit:
        g.add((q_uri, NS.unitSymbol, Literal(unit)))
        if unit in UNIT_MAP:
            g.add((q_uri, RDFS.seeAlso, UNIT_MAP[unit]))

    min_v, max_v = data.get("min"), data.get("max")
    if min_v is not None and max_v is not None and float(min_v) > float(max_v):
        min_v, max_v = max_v, min_v

    if data.get("value") is not None:
        g.add((q_uri, NS.numericValue, Literal(float(data["value"]), datatype=XSD.decimal)))
    if min_v is not None:
        g.add((q_uri, NS.minInclusive, Literal(float(min_v), datatype=XSD.decimal)))
    if max_v is not None:
        g.add((q_uri, NS.maxInclusive, Literal(float(max_v), datatype=XSD.decimal)))
    if data.get("approx"):
        g.add((q_uri, NS.isApproximate, Literal(True, datatype=XSD.boolean)))
    if data.get("qualifier"):
        g.add((q_uri, NS.qualifier, Literal(data["qualifier"])))
    if data.get("raw"):
        g.add((q_uri, NS.rawString, Literal(data["raw"])))

    return q_uri


def add_composition_entry(g: Graph, comp_uri: URIRef, elem_symbol: str,
                          elem_data: Dict[str, Any], comp_id: str):
    """Add a composition entry (n-ary pattern for element + concentration)."""
    elem_uri = mint_res(f"element/{safe_iri(elem_symbol)}")
    entry_uri = mint_res(f"comp-entry/{safe_iri(comp_id)}_{safe_iri(elem_symbol)}")

    g.add((entry_uri, RDF.type, NS.CompositionEntry))
    g.add((comp_uri, NS.hasComponent, entry_uri))
    g.add((entry_uri, NS.element, elem_uri))

    if isinstance(elem_data, (int, float)):
        elem_data = {"value": elem_data}

    if elem_data.get("is_balance_remainder"):
        g.add((entry_uri, NS.isBalanceRemainder, Literal(True, datatype=XSD.boolean)))
        return

    q_data = {k: elem_data[k] for k in ["value", "min", "max", "qualifier", "raw", "approx"]
              if k in elem_data}

    if q_data:
        unit = elem_data.get("unit", "%")
        q_uri = mint_res(f"quantity/{safe_iri(comp_id)}_{safe_iri(elem_symbol)}_mass")
        g.add((q_uri, RDF.type, NS.Quantity))
        g.add((entry_uri, NS.hasMassFraction, q_uri))

        if unit:
            g.add((q_uri, NS.unitSymbol, Literal(unit)))

        if q_data.get("value") is not None:
            g.add((q_uri, NS.numericValue, Literal(float(q_data["value"]), datatype=XSD.decimal)))
        if q_data.get("min") is not None:
            g.add((q_uri, NS.minInclusive, Literal(float(q_data["min"]), datatype=XSD.decimal)))
        if q_data.get("max") is not None:
            g.add((q_uri, NS.maxInclusive, Literal(float(q_data["max"]), datatype=XSD.decimal)))
        if q_data.get("approx"):
            g.add((q_uri, NS.isApproximate, Literal(True, datatype=XSD.boolean)))
        if q_data.get("qualifier"):
            g.add((q_uri, NS.qualifier, Literal(q_data["qualifier"])))
        if q_data.get("raw"):
            g.add((q_uri, NS.rawString, Literal(q_data["raw"])))


def add_computed_features(g: Graph, v_uri: URIRef, computed: Dict[str, Any], alloy_name: str):
    """Add computed metallurgical features to a variant."""
    feature_map = {
        "Md_avg": ("hasMdAverage", XSD.decimal),
        "Md_gamma": ("hasMdGamma", XSD.decimal),
        "VEC_avg": ("hasVECAvg", XSD.decimal),
        "gamma_prime_estimated_vol_pct": ("hasGammaPrimeEstimate", XSD.decimal),
        "density_calculated_gcm3": ("hasDensityCalculated", XSD.decimal),
        "TCP_risk": ("hasTcpRisk", None),
        "lattice_mismatch_pct": ("hasLatticeMismatchPct", XSD.decimal),
        "SSS_total_wt_pct": ("hasSSSTotalWtPct", XSD.decimal),
        "SSS_coefficient": ("hasSSSCoefficient", XSD.decimal),
        "precipitation_hardening_coeff": ("hasPrecipitationHardeningCoeff", XSD.decimal),
        "creep_resistance_param": ("hasCreepResistanceParam", XSD.decimal),
        "refractory_total_wt_pct": ("hasRefractoryTotalWtPct", XSD.decimal),
        "GP_formers_wt_pct": ("hasGPFormersWtPct", XSD.decimal),
        "oxidation_resistance": ("hasOxidationResistance", XSD.decimal),
        "Al_Ti_ratio": ("hasAlTiRatio", XSD.decimal),
        "Al_Ti_at_ratio": ("hasAlTiAtRatio", XSD.decimal),
        "Cr_Co_ratio": ("hasCrCoRatio", XSD.decimal),
        "Cr_Ni_ratio": ("hasCrNiRatio", XSD.decimal),
        "Mo_W_ratio": ("hasMoWRatio", XSD.decimal),
        "GP_formers_at_pct": ("hasGPFormersAtPct", XSD.decimal),
        "Al_Ti_interaction": ("hasAlTiInteraction", XSD.decimal),
        "Cr_Al_interaction": ("hasCrAlInteraction", XSD.decimal),
    }

    for key, (prop_name, datatype) in feature_map.items():
        if key in computed:
            value = computed[key]
            if datatype:
                g.add((v_uri, NS[prop_name], Literal(float(value), datatype=datatype)))
            else:
                g.add((v_uri, NS[prop_name], Literal(value)))

    json_fields = [
        ("atomic_percent", "hasAtomicCompositionJson"),
        ("gamma_composition_at", "hasGammaCompositionJson"),
        ("gamma_prime_composition_at", "hasGammaPrimeCompositionJson"),
    ]

    for field, prop_name in json_fields:
        if field in computed:
            g.add((v_uri, NS[prop_name], Literal(json.dumps(computed[field]))))


def add_property_measurements(g: Graph, v_uri: URIRef, variant_id: str,
                              prop_key: str, measurements: list):
    """Add mechanical property measurements to a variant."""
    prop_class = PROPERTY_MAP.get(prop_key)
    if not prop_class or not measurements:
        return

    if isinstance(measurements, str):
        try:
            measurements = ast.literal_eval(measurements)
        except (ValueError, SyntaxError):
            log.warning(f"Failed to parse measurements for {prop_key}")
            return

    propset_uri = mint_res(f"propset/{safe_iri(variant_id)}_{safe_iri(prop_class)}")
    g.add((propset_uri, RDF.type, NS.PropertySet))
    g.add((v_uri, NS.hasPropertySet, propset_uri))
    g.add((propset_uri, NS.measuresProperty, NS[prop_class]))

    for meas_data in measurements:
        meas_uri = mint_res_uuid(f"meas/{safe_iri(variant_id)}_{safe_iri(prop_class)}")
        g.add((meas_uri, RDF.type, NS.Measurement))
        g.add((propset_uri, NS.hasMeasurement, meas_uri))

        temp_category = meas_data.get("temp_category")
        temp_c = meas_data.get("temp_c")
        if temp_category:
            g.add((meas_uri, NS.temperatureCategory, Literal(temp_category)))
        if temp_c is not None:
            temp_qty = add_quantity(g, "°C", {"value": temp_c})
            g.add((meas_uri, NS.hasTestTemperature, temp_qty))

        if "stress_mpa" in meas_data:
            g.add((meas_uri, NS.stress, Literal(float(meas_data["stress_mpa"]), datatype=XSD.decimal)))
        if "life_hours" in meas_data:
            g.add((meas_uri, NS.lifeHours, Literal(float(meas_data["life_hours"]), datatype=XSD.decimal)))

        q_data = {k: meas_data[k] for k in ["value", "min", "max", "qualifier", "raw", "approx"]
                  if k in meas_data}

        unit = meas_data.get("unit", "")
        if not unit:
            unit_map = {
                "yield_strength": "MPa", "uts": "MPa",
                "elongation": "%", "elasticity": "GPa"
            }
            unit = unit_map.get(prop_key, "")

        q_uri = add_quantity(g, unit, q_data)
        g.add((meas_uri, NS.hasQuantity, q_uri))


def build_data_graph(json_path: str) -> Graph:
    """Build RDF graph from JSON data."""
    log.info(f"Reading JSON: {json_path}")

    g = Graph()
    g.bind("ns", NS)
    g.bind("res", RES)
    g.bind("unit", QUDT_UNIT)
    g.bind("chebi", CHEBI)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    alloys = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("==>"):
                continue
            try:
                alloys.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    line_fixed = line.replace("null", "None").replace("true", "True").replace("false", "False")
                    alloys.append(ast.literal_eval(line_fixed))
                except (ValueError, SyntaxError):
                    log.warning(f"Skipping invalid line: {line[:50]}...")

    log.info(f"Loaded {len(alloys)} alloy records")

    primary_compositions = {}
    for alloy_data in alloys:
        name = alloy_data.get("alloy")
        comp = alloy_data.get("composition")
        if name and comp and name not in primary_compositions:
            primary_compositions[name] = comp

    elements_created = set()

    for alloy_data in alloys:
        alloy_name = alloy_data.get("alloy", "")
        if not alloy_name:
            continue

        alloy_uri = mint_res(f"alloy/{safe_iri(alloy_name)}")
        g.add((alloy_uri, RDF.type, NS.NickelBasedSuperalloy))
        g.add((alloy_uri, RDFS.label, Literal(alloy_name)))
        g.add((alloy_uri, NS.tradeDesignation, Literal(alloy_name)))

        if alloy_data.get("uns"):
            g.add((alloy_uri, NS.unsNumber, Literal(alloy_data["uns"])))
        if alloy_data.get("family"):
            g.add((alloy_uri, NS.family, Literal(alloy_data["family"])))

        processing = alloy_data.get("processing")
        form_val = alloy_data.get("form")
        variant_id = "_".join(filter(None, [alloy_name, processing, form_val]))

        variant_uri = mint_res(f"variant/{safe_iri(variant_id)}")
        g.add((variant_uri, RDF.type, NS.Variant))
        g.add((variant_uri, RDFS.label, Literal(variant_id.replace("_", " "))))
        g.add((alloy_uri, NS.hasVariant, variant_uri))

        if processing:
            g.add((variant_uri, NS.processingMethod, Literal(processing)))
            pm_uri = mint_res(f"method/{safe_iri(processing)}")
            g.add((pm_uri, RDF.type, NS.ProcessingMethod))
            g.add((pm_uri, RDFS.label, Literal(processing)))
            g.add((variant_uri, NS.hasProcessingMethod, pm_uri))

        if form_val:
            form_uri = mint_res(f"form/{safe_iri(form_val)}")
            g.add((form_uri, RDF.type, NS.Form))
            g.add((form_uri, RDFS.label, Literal(form_val)))
            g.add((form_uri, NS.form, Literal(form_val)))
            g.add((variant_uri, NS.hasForm, form_uri))

        if alloy_data.get("density_gcm3"):
            g.add((variant_uri, NS.density, Literal(float(alloy_data["density_gcm3"]), datatype=XSD.decimal)))
        if alloy_data.get("gamma_prime_vol_pct") is not None:
            g.add((variant_uri, NS.gammaPrimeVolPct, Literal(float(alloy_data["gamma_prime_vol_pct"]), datatype=XSD.decimal)))
        if alloy_data.get("typical_heat_treatment"):
            g.add((variant_uri, NS.typicalHeatTreatment, Literal(alloy_data["typical_heat_treatment"])))

        try:
            computed = compute_alloy_features(alloy_data)
            add_computed_features(g, variant_uri, computed, alloy_name)
            log.info(f"[{alloy_name}] Added features: Md_avg={computed.get('Md_avg')}, TCP={computed.get('TCP_risk')}")
        except Exception as e:
            log.warning(f"[{alloy_name}] Feature computation failed: {e}")

        composition = alloy_data.get("composition", {}) or primary_compositions.get(alloy_name, {})
        has_own_comp = bool(alloy_data.get("composition"))

        comp_id = variant_id if has_own_comp else alloy_name
        comp_uri = mint_res(f"comp/{safe_iri(comp_id)}")
        g.add((comp_uri, RDF.type, NS.Composition))
        g.add((variant_uri, NS.hasComposition, comp_uri))

        if alloy_data.get("other_constituents"):
            g.add((comp_uri, NS.otherConstituents, Literal(alloy_data["other_constituents"])))

        for elem_symbol, elem_data in composition.items():
            if elem_symbol == "other":
                continue

            if elem_symbol not in elements_created:
                elem_uri = mint_res(f"element/{safe_iri(elem_symbol)}")
                g.add((elem_uri, RDF.type, NS.Element))
                g.add((elem_uri, RDFS.label, Literal(elem_symbol)))

                if elem_symbol in CHEBI_ELEMENT_MAP:
                    chebi_uri = CHEBI[CHEBI_ELEMENT_MAP[elem_symbol]]
                    g.add((elem_uri, OWL.sameAs, chebi_uri))

                elements_created.add(elem_symbol)

            add_composition_entry(g, comp_uri, elem_symbol, elem_data, comp_id)

        for prop_key in PROPERTY_MAP.keys():
            measurements = alloy_data.get(prop_key, [])
            if measurements:
                add_property_measurements(g, variant_uri, variant_id, prop_key, measurements)

    log.info(f"Graph built with {len(g)} triples")
    return g


def create_repo_if_missing():
    """Create GraphDB repository if it doesn't exist."""
    resp = requests.get(f"{GRAPHDB_URL}/repositories/{REPO_ID}",
                        headers={"Accept": "application/json"}, timeout=10)

    if resp.status_code == 200:
        log.info(f"Repository '{REPO_ID}' exists")
        return

    log.info(f"Creating repository '{REPO_ID}'...")

    repo_config_path = Path(__file__).parent / "repo-config.ttl"
    if not repo_config_path.exists():
        raise FileNotFoundError(f"Repository config not found: {repo_config_path}")

    with open(repo_config_path, "r") as f:
        config = f.read().replace('repositoryID> "AlloyGraph"', f'repositoryID> "{REPO_ID}"')

    files = {'config': ('config.ttl', config, 'text/turtle')}
    create_resp = requests.post(f"{GRAPHDB_URL}/rest/repositories", files=files, timeout=30)

    if create_resp.status_code // 100 != 2:
        if "already exists" in create_resp.text:
            log.info(f"Repository '{REPO_ID}' already exists (race condition)")
            return
        raise RuntimeError(f"Failed to create repository: {create_resp.status_code} {create_resp.text}")

    log.info(f"✅ Repository '{REPO_ID}' created")


def clear_graph(graph_uri: URIRef):
    """Clear a specific named graph."""
    log.info(f"Clearing graph {graph_uri}...")
    endpoint = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    params = {"context": f"<{graph_uri}>"}
    resp = requests.delete(endpoint, params=params, timeout=30)
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Failed to clear graph: {resp.status_code}")


def upload_graph(g: Graph, graph_uri: URIRef, clear_first: bool = True):
    """Upload RDF graph to GraphDB."""
    if clear_first:
        clear_graph(graph_uri)

    ttl_bytes = g.serialize(format="turtle").encode("utf-8")
    log.info(f"Uploading {len(g)} triples to {graph_uri}...")

    endpoint = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    params = {"context": f"<{graph_uri}>"}
    headers = {"Content-Type": "text/turtle; charset=UTF-8"}

    resp = requests.post(endpoint, params=params, data=ttl_bytes, headers=headers, timeout=120)
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")

    log.info(f"✅ Uploaded to {graph_uri}")


def upload_ontology():
    """Upload ontology to GraphDB."""
    if not Path(ONTOLOGY_FILE).exists():
        log.warning(f"Ontology file not found: {ONTOLOGY_FILE}")
        return

    log.info(f"Reading ontology: {ONTOLOGY_FILE}")
    with open(ONTOLOGY_FILE, "rb") as f:
        rdf_data = f.read()

    log.info(f"Uploading ontology ({len(rdf_data)} bytes)...")
    endpoint = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    params = {"context": f"<{ONTOLOGY_GRAPH}>"}
    headers = {"Content-Type": "application/rdf+xml; charset=UTF-8"}

    clear_graph(ONTOLOGY_GRAPH)

    resp = requests.post(endpoint, params=params, data=rdf_data, headers=headers, timeout=120)
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Ontology upload failed: {resp.status_code} {resp.text}")

    log.info(f"✅ Ontology uploaded to {ONTOLOGY_GRAPH}")


def main():
    """Main execution: create repo, upload ontology, then upload data."""
    log.info("=" * 60)
    log.info("AlloyGraph Data Ingestion")
    log.info("=" * 60)
    log.info(f"GraphDB:  {GRAPHDB_URL}")
    log.info(f"Repo:     {REPO_ID}")
    log.info(f"JSON:     {JSON_FILE}")
    log.info(f"Ontology: {ONTOLOGY_FILE}")
    log.info("=" * 60)

    create_repo_if_missing()
    upload_ontology()

    if not Path(JSON_FILE).exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")

    data_graph = build_data_graph(JSON_FILE)
    upload_graph(data_graph, DATA_GRAPH, clear_first=True)

    log.info("=" * 60)
    log.info("✅ Ingestion complete!")
    log.info(f"   Ontology graph: {ONTOLOGY_GRAPH}")
    log.info(f"   Data graph:     {DATA_GRAPH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
