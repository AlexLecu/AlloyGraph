import os
import logging
from collections import defaultdict

import weaviate
from weaviate.util import generate_uuid5
from SPARQLWrapper import SPARQLWrapper, JSON

# Config
GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
GRAPHDB_REPO = os.getenv("GRAPHDB_REPO", "AlloyGraph")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8081"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50052"))

# URI Namespaces (matching enrich_graphdb_new.py)
ONTOLOGY_BASE = "https://w3id.org/alloygraph/ont#"
RESOURCE_BASE = "https://w3id.org/alloygraph/res/"
BASE = ONTOLOGY_BASE  # For backward compatibility in SPARQL queries
NS = "alloygraph"  # Namespace for UUID generation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def uuid5(key):
    return str(generate_uuid5(NS, key))


def to_float(x):
    try:
        return float(x)
    except:
        return None


def to_bool(x):
    if x is None:
        return None
    return str(x).strip().lower() in ("true", "1")


def query_graphdb(sparql_query):
    endpoint = f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPO}"
    sp = SPARQLWrapper(endpoint)
    sp.setQuery(sparql_query)
    sp.setReturnFormat(JSON)
    sp.setMethod("POST")
    return sp.query().convert()


def connect_weaviate():
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )
    client.is_live()
    return client


def fetch_alloys():
    """Fetch all alloys with pagination."""
    alloys = {}
    offset = 0
    page_size = 100

    log.info("Fetching alloys from GraphDB...")

    while True:
        q = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ont: <{ONTOLOGY_BASE}>
        PREFIX res: <{RESOURCE_BASE}>
        SELECT ?alloy ?label ?uns ?comp ?others ?family ?density ?gammaPrime ?heatTreat ?processing
        WHERE {{
          ?alloy rdf:type ont:NickelBasedSuperalloy ;
                 ont:tradeDesignation ?label .
          OPTIONAL {{ ?alloy ont:unsNumber ?uns }}
          OPTIONAL {{ ?alloy ont:family ?family }}
          OPTIONAL {{ ?alloy ont:density ?density }}
          OPTIONAL {{ ?alloy ont:gammaPrimeVolPct ?gammaPrime }}
          OPTIONAL {{ ?alloy ont:typicalHeatTreatment ?heatTreat }}
          OPTIONAL {{ ?alloy ont:hasProcessingMethod ?pm . ?pm rdfs:label ?processing }}
          OPTIONAL {{
            ?alloy ont:hasComposition ?comp .
            OPTIONAL {{ ?comp ont:otherConstituents ?others }}
          }}
        }}
        ORDER BY ?label
        LIMIT {page_size} OFFSET {offset}
        """

        res = query_graphdb(q)
        results = res['results']['bindings']

        if not results:
            break

        for b in results:
            alloys[b["alloy"]["value"]] = {
                "label": b["label"]["value"],
                "uns": b.get("uns", {}).get("value"),
                "family": b.get("family", {}).get("value"),
                "density": to_float(b.get("density", {}).get("value")),
                "gammaPrime": to_float(b.get("gammaPrime", {}).get("value")),
                "heatTreat": b.get("heatTreat", {}).get("value"),
                "comp": b.get("comp", {}).get("value"),
                "others": b.get("others", {}).get("value"),
                "processing": b.get("processing", {}).get("value")
            }

        if len(results) < page_size:
            break
        offset += page_size

    log.info(f"Found {len(alloys)} alloys")
    return alloys


def fetch_composition(comp_iri):
    """Get composition entries for a composition."""
    q = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ont: <{ONTOLOGY_BASE}>
    PREFIX res: <{RESOURCE_BASE}>
    SELECT ?entry ?elem ?elemLabel ?isBal ?qty ?num ?min ?max ?nom ?unit ?approx ?qual ?raw
    WHERE {{
      <{comp_iri}> ont:hasComponent ?entry .
      ?entry ont:element ?elem .
      OPTIONAL {{ ?elem rdfs:label ?elemLabel }}
      OPTIONAL {{ ?entry ont:isBalanceRemainder ?isBal }}
      OPTIONAL {{
        ?entry ont:hasMassFraction ?qty .
        OPTIONAL {{ ?qty ont:numericValue ?num }}
        OPTIONAL {{ ?qty ont:minInclusive ?min }}
        OPTIONAL {{ ?qty ont:maxInclusive ?max }}
        OPTIONAL {{ ?qty ont:nominal ?nom }}
        OPTIONAL {{ ?qty ont:unitSymbol ?unit }}
        OPTIONAL {{ ?qty ont:isApproximate ?approx }}
        OPTIONAL {{ ?qty ont:qualifier ?qual }}
        OPTIONAL {{ ?qty ont:rawString ?raw }}
      }}
    }}
    """

    res = query_graphdb(q)
    entries = {}

    for b in res["results"]["bindings"]:
        entry_iri = b["entry"]["value"]
        elem_label = b.get("elemLabel", {}).get("value")
        elem_iri = b["elem"]["value"]

        symbol = elem_label if elem_label else elem_iri.split("#")[-1].replace("Element_", "")

        qty = None
        if b.get("qty"):
            qty = {
                "iri": b["qty"]["value"],
                "numericValue": to_float(b.get("num", {}).get("value")),
                "minInclusive": to_float(b.get("min", {}).get("value")),
                "maxInclusive": to_float(b.get("max", {}).get("value")),
                "nominal": to_float(b.get("nom", {}).get("value")),
                "unitSymbol": b.get("unit", {}).get("value"),
                "isApproximate": to_bool(b.get("approx", {}).get("value")),
                "qualifier": b.get("qual", {}).get("value"),
                "rawString": b.get("raw", {}).get("value"),
            }

        entries[entry_iri] = {
            "symbol": symbol,
            "is_balance": to_bool(b.get("isBal", {}).get("value")),
            "qty": qty,
        }

    return entries


def fetch_measurements(source_iri):
    """Get measurements for an alloy or variant."""
    measurements = []

    # Try direct measurements first
    q = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ont: <{ONTOLOGY_BASE}>
    PREFIX res: <{RESOURCE_BASE}>
    SELECT ?meas ?prop ?qty ?num ?min ?max ?nom ?unit ?approx ?qual ?raw
           ?tempCat ?heatTreat ?testTemp ?testTempNum ?testTempUnit
           ?stress ?lifeHours
    WHERE {{
      <{source_iri}> ont:hasMeasurement ?meas .
      ?meas ont:measures ?prop .
      OPTIONAL {{
        ?meas ont:hasQuantity ?qty .
        OPTIONAL {{ ?qty ont:numericValue ?num }}
        OPTIONAL {{ ?qty ont:minInclusive ?min }}
        OPTIONAL {{ ?qty ont:maxInclusive ?max }}
        OPTIONAL {{ ?qty ont:nominal ?nom }}
        OPTIONAL {{ ?qty ont:unitSymbol ?unit }}
        OPTIONAL {{ ?qty ont:isApproximate ?approx }}
        OPTIONAL {{ ?qty ont:qualifier ?qual }}
        OPTIONAL {{ ?qty ont:rawString ?raw }}
      }}
      OPTIONAL {{ ?meas ont:temperatureCategory ?tempCat }}
      OPTIONAL {{ ?meas ont:heatTreatmentCondition ?heatTreat }}
      OPTIONAL {{ ?meas ont:stress ?stress }}
      OPTIONAL {{ ?meas ont:lifeHours ?lifeHours }}
      OPTIONAL {{
        ?meas ont:hasTestTemperature ?testTemp .
        OPTIONAL {{ ?testTemp ont:numericValue ?testTempNum }}
        OPTIONAL {{ ?testTemp ont:unitSymbol ?testTempUnit }}
      }}
    }}
    """

    try:
        res = query_graphdb(q)
        for b in res["results"]["bindings"]:
            prop_type = b["prop"]["value"].split("#")[-1]

            qty = None
            if b.get("qty"):
                qty = {
                    "iri": b["qty"]["value"],
                    "numericValue": to_float(b.get("num", {}).get("value")),
                    "minInclusive": to_float(b.get("min", {}).get("value")),
                    "maxInclusive": to_float(b.get("max", {}).get("value")),
                    "nominal": to_float(b.get("nom", {}).get("value")),
                    "unitSymbol": b.get("unit", {}).get("value"),
                    "isApproximate": to_bool(b.get("approx", {}).get("value")),
                    "qualifier": b.get("qual", {}).get("value"),
                    "rawString": b.get("raw", {}).get("value"),
                }

            test_temp = None
            if b.get("testTemp"):
                test_temp = {
                    "iri": b["testTemp"]["value"],
                    "numericValue": to_float(b.get("testTempNum", {}).get("value")),
                    "unitSymbol": b.get("testTempUnit", {}).get("value"),
                }

            measurements.append({
                "iri": b["meas"]["value"],
                "prop_type": prop_type,
                "qty": qty,
                "temp_cat": b.get("tempCat", {}).get("value"),
                "heat_treatment": b.get("heatTreat", {}).get("value"),
                "stress": to_float(b.get("stress", {}).get("value")),
                "life_hours": to_float(b.get("lifeHours", {}).get("value")),
                "test_temp": test_temp,
            })
    except:
        pass

    # Try PropertySet structure
    q2 = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ont: <{ONTOLOGY_BASE}>
    PREFIX res: <{RESOURCE_BASE}>
    SELECT ?propSet ?prop ?meas ?qty ?num ?min ?max ?nom ?unit ?approx ?qual ?raw
           ?tempCat ?heatTreat ?testTemp ?testTempNum ?testTempUnit
           ?stress ?lifeHours
    WHERE {{
      <{source_iri}> ont:hasPropertySet ?propSet .
      ?propSet ont:measuresProperty ?prop .
      ?propSet ont:hasMeasurement ?meas .
      OPTIONAL {{
        ?meas ont:hasQuantity ?qty .
        OPTIONAL {{ ?qty ont:numericValue ?num }}
        OPTIONAL {{ ?qty ont:minInclusive ?min }}
        OPTIONAL {{ ?qty ont:maxInclusive ?max }}
        OPTIONAL {{ ?qty ont:nominal ?nom }}
        OPTIONAL {{ ?qty ont:unitSymbol ?unit }}
        OPTIONAL {{ ?qty ont:isApproximate ?approx }}
        OPTIONAL {{ ?qty ont:qualifier ?qual }}
        OPTIONAL {{ ?qty ont:rawString ?raw }}
      }}
      OPTIONAL {{ ?meas ont:temperatureCategory ?tempCat }}
      OPTIONAL {{ ?meas ont:heatTreatmentCondition ?heatTreat }}
      OPTIONAL {{ ?meas ont:stress ?stress }}
      OPTIONAL {{ ?meas ont:lifeHours ?lifeHours }}
      OPTIONAL {{
        ?meas ont:hasTestTemperature ?testTemp .
        OPTIONAL {{ ?testTemp ont:numericValue ?testTempNum }}
        OPTIONAL {{ ?testTemp ont:unitSymbol ?testTempUnit }}
      }}
    }}
    """

    try:
        res = query_graphdb(q2)
        for b in res["results"]["bindings"]:
            prop_type = b["prop"]["value"].split("#")[-1]

            qty = None
            if b.get("qty"):
                qty = {
                    "iri": b["qty"]["value"],
                    "numericValue": to_float(b.get("num", {}).get("value")),
                    "minInclusive": to_float(b.get("min", {}).get("value")),
                    "maxInclusive": to_float(b.get("max", {}).get("value")),
                    "nominal": to_float(b.get("nom", {}).get("value")),
                    "unitSymbol": b.get("unit", {}).get("value"),
                    "isApproximate": to_bool(b.get("approx", {}).get("value")),
                    "qualifier": b.get("qual", {}).get("value"),
                    "rawString": b.get("raw", {}).get("value"),
                }

            test_temp = None
            if b.get("testTemp"):
                test_temp = {
                    "iri": b["testTemp"]["value"],
                    "numericValue": to_float(b.get("testTempNum", {}).get("value")),
                    "unitSymbol": b.get("testTempUnit", {}).get("value"),
                }

            measurements.append({
                "iri": b["meas"]["value"],
                "prop_type": prop_type,
                "qty": qty,
                "temp_cat": b.get("tempCat", {}).get("value"),
                "heat_treatment": b.get("heatTreat", {}).get("value"),
                "stress": to_float(b.get("stress", {}).get("value")),
                "life_hours": to_float(b.get("lifeHours", {}).get("value")),
                "test_temp": test_temp,
            })
    except:
        pass

    return measurements


def batch_add_object(batch, collection_name, uuid, props):
    """Add object to batch."""
    props = {k: v for k, v in props.items() if v is not None}
    batch.add_object(
        collection=collection_name,
        uuid=uuid,
        properties=props
    )

def batch_add_ref(batch, from_coll, from_uuid, ref_prop, to_uuid):
    """Add reference to batch."""
    batch.add_reference(
        from_collection=from_coll,
        from_uuid=from_uuid,
        from_property=ref_prop,
        to=to_uuid
    )

def create_quantity(batch, qty):
    """Create quantity object and return UUID."""
    if not qty:
        return None

    q_uuid = uuid5(f"Quantity:{qty['iri']}")
    props = {}

    for k in ("numericValue", "minInclusive", "maxInclusive", "nominal",
              "unitSymbol", "isApproximate", "qualifier", "rawString"):
        if qty.get(k) is not None:
            props[k] = qty[k]

    batch_add_object(batch, "Quantity", q_uuid, props)
    return q_uuid


def fetch_variants(alloy_iri):
    """Fetch variants for a given alloy."""
    variants = []
    q = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ont: <{ONTOLOGY_BASE}>
    PREFIX res: <{RESOURCE_BASE}>
    SELECT ?var ?label ?proc ?form ?comp ?others ?density ?gammaPrime ?heatTreat
           ?mdAvg ?mdGamma ?vecAvg ?gammaPrimeEst ?densityCalc ?tcpRisk
           ?latticeMismatch ?sss ?sssCoeff ?precipHard ?creepRes ?ref ?gpWT ?oxRes
           ?alti ?altiAt ?crco ?crni ?mow ?gpAt ?altiInt ?cralInt
           ?apJson ?gammaJson ?gammaPrimeJson
    WHERE {{
      <{alloy_iri}> ont:hasVariant ?var .
      OPTIONAL {{ ?var rdfs:label ?label }}
      OPTIONAL {{ ?var ont:hasProcessingMethod ?pm . ?pm rdfs:label ?proc }}
      OPTIONAL {{ ?var ont:hasForm ?f . ?f ont:form ?form }}
      OPTIONAL {{ ?var ont:density ?density }}
      OPTIONAL {{ ?var ont:gammaPrimeVolPct ?gammaPrime }}
      OPTIONAL {{ ?var ont:typicalHeatTreatment ?heatTreat }}
      # Computed metallurgical features
      OPTIONAL {{ ?var ont:hasMdAverage ?mdAvg }}
      OPTIONAL {{ ?var ont:hasMdGamma ?mdGamma }}
      OPTIONAL {{ ?var ont:hasVECAvg ?vecAvg }}
      OPTIONAL {{ ?var ont:hasGammaPrimeEstimate ?gammaPrimeEst }}
      OPTIONAL {{ ?var ont:hasDensityCalculated ?densityCalc }}
      OPTIONAL {{ ?var ont:hasTcpRisk ?tcpRisk }}
      OPTIONAL {{ ?var ont:hasLatticeMismatchPct ?latticeMismatch }}
      OPTIONAL {{ ?var ont:hasSSSTotalWtPct ?sss }}
      OPTIONAL {{ ?var ont:hasSSSCoefficient ?sssCoeff }}
      OPTIONAL {{ ?var ont:hasPrecipitationHardeningCoeff ?precipHard }}
      OPTIONAL {{ ?var ont:hasCreepResistanceParam ?creepRes }}
      OPTIONAL {{ ?var ont:hasRefractoryTotalWtPct ?ref }}
      OPTIONAL {{ ?var ont:hasGPFormersWtPct ?gpWT }}
      OPTIONAL {{ ?var ont:hasOxidationResistance ?oxRes }}
      OPTIONAL {{ ?var ont:hasAlTiRatio ?alti }}
      OPTIONAL {{ ?var ont:hasAlTiAtRatio ?altiAt }}
      OPTIONAL {{ ?var ont:hasCrCoRatio ?crco }}
      OPTIONAL {{ ?var ont:hasCrNiRatio ?crni }}
      OPTIONAL {{ ?var ont:hasMoWRatio ?mow }}
      OPTIONAL {{ ?var ont:hasGPFormersAtPct ?gpAt }}
      OPTIONAL {{ ?var ont:hasAlTiInteraction ?altiInt }}
      OPTIONAL {{ ?var ont:hasCrAlInteraction ?cralInt }}
      # JSON composition fields
      OPTIONAL {{ ?var ont:hasAtomicCompositionJson ?apJson }}
      OPTIONAL {{ ?var ont:hasGammaCompositionJson ?gammaJson }}
      OPTIONAL {{ ?var ont:hasGammaPrimeCompositionJson ?gammaPrimeJson }}

      OPTIONAL {{
        ?var ont:hasComposition ?comp .
        OPTIONAL {{ ?comp ont:otherConstituents ?others }}
      }}
    }}
    """
    res = query_graphdb(q)
    for b in res["results"]["bindings"]:
        variants.append({
            "iri": b["var"]["value"],
            "label": b.get("label", {}).get("value"),
            "processing": b.get("proc", {}).get("value"),
            "form": b.get("form", {}).get("value"),
            "density": to_float(b.get("density", {}).get("value")),
            "gammaPrime": to_float(b.get("gammaPrime", {}).get("value")),
            "heatTreat": b.get("heatTreat", {}).get("value"),
            "comp": b.get("comp", {}).get("value"),
            "others": b.get("others", {}).get("value"),
            # Computed metallurgical features
            "mdAvg": to_float(b.get("mdAvg", {}).get("value")),
            "mdGamma": to_float(b.get("mdGamma", {}).get("value")),
            "vecAvg": to_float(b.get("vecAvg", {}).get("value")),
            "gammaPrimeEst": to_float(b.get("gammaPrimeEst", {}).get("value")),
            "densityCalc": to_float(b.get("densityCalc", {}).get("value")),
            "tcpRisk": b.get("tcpRisk", {}).get("value"),
            "latticeMismatchPct": to_float(b.get("latticeMismatch", {}).get("value")),
            "sssTotalWtPct": to_float(b.get("sss", {}).get("value")),
            "sssCoefficient": to_float(b.get("sssCoeff", {}).get("value")),
            "precipitationHardeningCoeff": to_float(b.get("precipHard", {}).get("value")),
            "creepResistanceParam": to_float(b.get("creepRes", {}).get("value")),
            "refractoryTotalWtPct": to_float(b.get("ref", {}).get("value")),
            "gpFormersWtPct": to_float(b.get("gpWT", {}).get("value")),
            "oxidationResistance": to_float(b.get("oxRes", {}).get("value")),
            "alTiRatio": to_float(b.get("alti", {}).get("value")),
            "alTiAtRatio": to_float(b.get("altiAt", {}).get("value")),
            "crCoRatio": to_float(b.get("crco", {}).get("value")),
            "crNiRatio": to_float(b.get("crni", {}).get("value")),
            "moWRatio": to_float(b.get("mow", {}).get("value")),
            "gpFormersAtPct": to_float(b.get("gpAt", {}).get("value")),
            "alTiInteraction": to_float(b.get("altiInt", {}).get("value")),
            "crAlInteraction": to_float(b.get("cralInt", {}).get("value")),
            # JSON composition fields
            "atomicCompositionJson": b.get("apJson", {}).get("value"),
            "gammaCompositionJson": b.get("gammaJson", {}).get("value"),
            "gammaPrimeCompositionJson": b.get("gammaPrimeJson", {}).get("value"),
        })
    return variants

def main():
    log.info("Starting GraphDB → Weaviate import (Hierarchical)")

    alloys = fetch_alloys()
    client = connect_weaviate()

    try:
        # Build lookups
        elem_lookup = {} 
        prop_lookup = {}
        
        log.info("Seeding Property Types...")
        PROPS = [
            "TensileStrength", "YieldStrength", "Elongation", "Hardness",
            "ElasticModulus", "Elasticity", "UTS", "CreepRupture", "ReductionOfArea"
        ]
        
        with client.batch.dynamic() as batch:
            for p in PROPS:
                p_uuid = uuid5(f"Property:{p}")
                batch_add_object(batch, "MechanicalProperty", p_uuid, {
                    "label": p,
                    "propertyType": p
                })
                prop_lookup[p] = p_uuid
        
        prop_col = client.collections.get("MechanicalProperty")
        for o in prop_col.iterator():
            pt = o.properties.get("propertyType")
            if pt: prop_lookup[pt] = o.uuid

        stats = defaultdict(int)

        log.info(f"Importing {len(alloys)} alloys with BATCHING...")

        with client.batch.dynamic() as batch:
            
            for idx, (a_iri, meta) in enumerate(alloys.items(), 1):
                if idx % 50 == 0:
                    log.info(f"  {idx}/{len(alloys)}...")

                a_uuid = uuid5(f"Alloy:{a_iri}")

                # Create Alloy
                batch_add_object(batch, "NickelBasedSuperalloy", a_uuid, {
                    "tradeDesignation": meta["label"],
                    "unsNumber": meta.get("uns"),
                    "family": meta.get("family")
                })
                stats["alloys"] += 1

                # Fetch Variants
                variants = fetch_variants(a_iri)
                
                for v in variants:
                    v_uuid = uuid5(f"Variant:{v['iri']}")
                    
                    # Compute Comp Summary for Variant
                    comp_summary = ""
                    comp_entries = {}
                    if v.get("comp"):
                        comp_entries = fetch_composition(v["comp"])
                        parts = []
                        for e in comp_entries.values():
                            sym = e.get("symbol", "")
                            val = 0.0
                            if e.get("qty"):
                                val = e["qty"].get("numericValue") or e["qty"].get("nominal") or 0.0
                            if sym and val > 0:
                                parts.append(f"{sym}: {val}%")
                        comp_summary = ", ".join(parts)

                    batch_add_object(batch, "Variant", v_uuid, {
                        "name": v["label"],
                        "processingMethod": v["processing"],
                        "density": v["density"],
                        "gammaPrimeVolPct": v["gammaPrime"],
                        "typicalHeatTreatment": v["heatTreat"],
                        "compositionSummary": comp_summary,
                        # Computed metallurgical features
                        "mdAverage": v["mdAvg"],
                        "mdGamma": v["mdGamma"],
                        "vecAvg": v["vecAvg"],
                        "gammaPrimeEstimate": v["gammaPrimeEst"],
                        "densityCalculated": v["densityCalc"],
                        "tcpRisk": v["tcpRisk"],
                        "latticeMismatchPct": v["latticeMismatchPct"],
                        "sssTotalWtPct": v["sssTotalWtPct"],
                        "sssCoefficient": v["sssCoefficient"],
                        "precipitationHardeningCoeff": v["precipitationHardeningCoeff"],
                        "creepResistanceParam": v["creepResistanceParam"],
                        "refractoryTotalWtPct": v["refractoryTotalWtPct"],
                        "gpFormersWtPct": v["gpFormersWtPct"],
                        "oxidationResistance": v["oxidationResistance"],
                        "alTiRatio": v["alTiRatio"],
                        "alTiAtRatio": v["alTiAtRatio"],
                        "crCoRatio": v["crCoRatio"],
                        "crNiRatio": v["crNiRatio"],
                        "moWRatio": v["moWRatio"],
                        "gpFormersAtPct": v["gpFormersAtPct"],
                        "alTiInteraction": v["alTiInteraction"],
                        "crAlInteraction": v["crAlInteraction"],
                        # JSON composition fields
                        "atomicCompositionJson": v["atomicCompositionJson"],
                        "gammaCompositionJson": v["gammaCompositionJson"],
                        "gammaPrimeCompositionJson": v["gammaPrimeCompositionJson"],
                    })
                    batch_add_ref(batch, "NickelBasedSuperalloy", a_uuid, "hasVariant", v_uuid)

                    # FormType (Shared)
                    if v.get("form"):
                        f_uuid = uuid5(f"FormType:{v['form']}")
                        batch_add_object(batch, "FormType", f_uuid, {"formTypeName": v["form"]})
                        batch_add_ref(batch, "Variant", v_uuid, "hasFormType", f_uuid)
                    stats["variants"] += 1

                    # Create Composition
                    if v.get("comp"):
                        comp_uuid = uuid5(f"Composition:{v['comp']}")
                        batch_add_object(batch, "Composition", comp_uuid, {
                            "otherConstituents": v.get("others")
                        })
                        batch_add_ref(batch, "Variant", v_uuid, "hasComposition", comp_uuid)
                        stats["compositions"] += 1

                        for e_iri, e in comp_entries.items():
                            e_uuid = uuid5(f"CompEntry:{e_iri}")
                            batch_add_object(batch, "CompositionEntry", e_uuid, {"isBalanceRemainder": e["is_balance"]})
                            batch_add_ref(batch, "Composition", comp_uuid, "hasComponent", e_uuid)

                            sym = e["symbol"]
                            el_uuid = elem_lookup.get(sym)
                            if not el_uuid:
                                el_uuid = uuid5(f"Element:{sym}")
                                batch_add_object(batch, "Element", el_uuid, {"symbol": sym, "label": sym})
                                elem_lookup[sym] = el_uuid
                            batch_add_ref(batch, "CompositionEntry", e_uuid, "hasElement", el_uuid)

                            if e.get("qty"):
                                q_uuid = create_quantity(batch, e["qty"])
                                if q_uuid:
                                    batch_add_ref(batch, "CompositionEntry", e_uuid, "hasMassFraction", q_uuid)
                                    stats["quantities"] += 1
                            stats["entries"] += 1

                    # Processing Route (Shared)
                    if v.get("processing"):
                        pr_uuid = uuid5(f"ProcessingRoute:{v['processing']}")
                        batch_add_object(batch, "ProcessingRoute", pr_uuid, {"processingDescription": v["processing"]})
                        batch_add_ref(batch, "Variant", v_uuid, "hasProcessingRoute", pr_uuid)

                    # Properties (Using Variant IRI)
                    variant_meas = fetch_measurements(v['iri'])
                    by_prop = defaultdict(list)
                    for m in variant_meas:
                        by_prop[m["prop_type"]].append(m)

                    for prop_type, measurements in by_prop.items():
                        ps_uuid = uuid5(f"PropertySet:{v['iri']}:{prop_type}")
                        batch_add_object(batch, "PropertySet", ps_uuid, {})
                        batch_add_ref(batch, "Variant", v_uuid, "hasPropertySet", ps_uuid)
                        stats["property_sets"] += 1
                        
                        mp_uuid = prop_lookup.get(prop_type)
                        if mp_uuid:
                            batch_add_ref(batch, "PropertySet", ps_uuid, "measuresProperty", mp_uuid)

                        for m in measurements:
                            m_uuid = uuid5(f"Measurement:{m['iri']}")
                            batch_add_object(batch, "Measurement", m_uuid, {
                                "stress": m.get("stress"), "lifeHours": m.get("life_hours")
                            })
                            batch_add_ref(batch, "PropertySet", ps_uuid, "hasMeasurement", m_uuid)
                            stats["measurements"] += 1

                            if m.get("qty"):
                                q_uuid = create_quantity(batch, m["qty"])
                                if q_uuid: batch_add_ref(batch, "Measurement", m_uuid, "hasQuantity", q_uuid)

                            if m.get("temp_cat") or m.get("heat_treatment") or m.get("test_temp"):
                                tc_uuid = uuid5(f"TestCond:{m['iri']}")
                                batch_add_object(batch, "TestCondition", tc_uuid, {
                                    "temperatureCategory": m.get("temp_cat"),
                                    "heatTreatmentCondition": m.get("heat_treatment")
                                })
                                batch_add_ref(batch, "Measurement", m_uuid, "hasTestCondition", tc_uuid)
                                
                                if m.get("test_temp"):
                                    temp_q_uuid = create_quantity(batch, m["test_temp"])
                                    if temp_q_uuid: batch_add_ref(batch, "TestCondition", tc_uuid, "hasTemperature", temp_q_uuid)

        log.info("\nImport complete!")
        for key, value in sorted(stats.items()):
            log.info(f"  {key}: {value}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
