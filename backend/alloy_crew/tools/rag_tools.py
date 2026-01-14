from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Type, Dict, Any
import weaviate
from weaviate.classes.query import QueryReference
import os
import json
import ast
import logging
import hashlib

logger = logging.getLogger(__name__)

class AlloySearchInput(BaseModel):
    """Input for Weaviate search."""
    composition: Dict[str, float] = Field(
        ...,
        description=(
            "Target composition to find similar alloys for as a dict of wt% values, "
            "e.g. {'Ni': 60.0, 'Al': 5.0}. "
            "If you only have a JSON string, pass it and it will be parsed."
        ),
    )
    limit: int = Field(3, description="Number of results to return.")

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_style_inputs(cls, data: Any) -> Any:
        """
        CrewAI sometimes shows tool args as a schema-like dict:
          {"composition": {"description": "...", "type": "dict[str, float]"}, "limit": {...}}
        Some models will incorrectly echo that structure back as the actual tool input.
        This normalizes those cases before regular field validation runs.
        """
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        comp = normalized.get("composition")
        if isinstance(comp, dict):
            for key in ("value", "composition", "description"):
                if key in comp:
                    normalized["composition"] = comp[key]
                    break

        lim = normalized.get("limit")
        if isinstance(lim, dict):
            for key in ("value", "limit", "default"):
                if key in lim:
                    normalized["limit"] = lim[key]
                    break
            else:
                normalized.pop("limit", None)

        return normalized

    @field_validator("composition", mode="before")
    @classmethod
    def _parse_composition(cls, value: Any) -> Dict[str, float]:
        if isinstance(value, dict):
            if "description" in value and isinstance(value["description"], str):
                value = value["description"]
            else:
                return {str(k): float(v) for k, v in value.items()}

        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = json.loads(text.replace("'", '"'))
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception as e:
                    raise ValueError("Invalid composition format; expected dict or JSON string.") from e

            if not isinstance(parsed, dict):
                raise ValueError("Invalid composition format; expected a dict.")
            return {str(k): float(v) for k, v in parsed.items()}

        raise ValueError("Invalid composition format; expected dict or JSON string.")

    @field_validator("limit", mode="before")
    @classmethod
    def _parse_limit(cls, value: Any) -> int:
        if value is None:
            return 3
        if isinstance(value, dict):
            for key in ("value", "limit", "default"):
                if key in value:
                    value = value[key]
                    break
            else:
                return 3
        return int(value)


# ============================================================
# KG SEARCH CACHING
# ============================================================
def _create_cache_key(composition: Dict[str, float], limit: int) -> str:
    """
    Create stable hash key for composition + limit.

    Normalizes composition to handle rounding variations:
    - {Ni: 60.0, Al: 5.0} and {Ni: 60.01, Al: 4.99} → same key
    """
    # Normalize to 2 decimal places and sort for consistent hashing
    normalized = {k: round(v, 2) for k, v in composition.items() if v > 0}
    sorted_comp = json.dumps(normalized, sort_keys=True)
    cache_str = f"{sorted_comp}|{limit}"
    return hashlib.md5(cache_str.encode()).hexdigest()

_kg_search_cache = {}
_cache_hits = 0
_cache_misses = 0


def _get_cached_search(composition: Dict[str, float], limit: int) -> tuple:
    """
    Get cached KG search result if available.
    """
    global _cache_hits, _cache_misses

    cache_key = _create_cache_key(composition, limit)

    if cache_key in _kg_search_cache:
        _cache_hits += 1
        logging.debug(f"✓ KG Cache HIT (hits={_cache_hits}, misses={_cache_misses})")
        return _kg_search_cache[cache_key], True
    else:
        _cache_misses += 1
        logging.debug(f"✗ KG Cache MISS (hits={_cache_hits}, misses={_cache_misses})")
        return None, False


def _store_cached_search(composition: Dict[str, float], limit: int, result: str):
    """Store KG search result in cache."""
    global _kg_search_cache

    cache_key = _create_cache_key(composition, limit)

    # Simple LRU: if cache exceeds 128 entries, remove oldest
    if len(_kg_search_cache) >= 128:
        # Remove first (oldest) entry
        oldest_key = next(iter(_kg_search_cache))
        _kg_search_cache.pop(oldest_key)

    _kg_search_cache[cache_key] = result


def get_cache_stats() -> dict:
    """Get KG search cache statistics."""
    global _cache_hits, _cache_misses

    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0

    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate_pct": round(hit_rate, 1),
        "cache_size": len(_kg_search_cache)
    }


class AlloySearchTool(BaseTool):
    name: str = "AlloyKnowledgeGraphSearch"
    description: str = (
        "Searches the Weaviate Knowledge Graph for existing alloys similar to the input composition. "
        "Returns a list of matching alloys with their properties, composition, and processing family. "
        "Use this for Research and Benchmarking. "
        "IMPORTANT: pass real values, not the schema. Example Action Input: "
        "{'composition': {'Ni': 58.0, 'Cr': 19.5}, 'limit': 3}."
    )
    args_schema: Type[BaseModel] = AlloySearchInput

    def _run(self, composition: Dict[str, float], limit: int = 3, **kwargs: Any) -> Any:
        # Check cache first
        cached_result, cache_hit = _get_cached_search(composition, limit)
        if cache_hit:
            return cached_result

        client = None
        try:
            WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'localhost')
            WEAVIATE_PORT = int(os.getenv('WEAVIATE_PORT', 8081))
            WEAVIATE_GRPC_PORT = int(os.getenv('WEAVIATE_GRPC_PORT', 50052))
            
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
            
            if not client.is_live():
                return "Error: Weaviate instance is not reachable."

            # 2. Query Strategy: Wide Net + Reranking
            collection = client.collections.get("Variant")
            
            query_parts = [f"{v}% {k}" for k, v in composition.items() if v > 1.0]
            query_str = "Superalloy with " + ", ".join(query_parts)

            response = collection.query.hybrid(
                query=query_str,
                limit=50,
                return_properties=[
                    "name", "processingMethod", "densityCalculated", "gammaPrimeEstimate", 
                    "mdAverage", "tcpRisk", "sssTotalWtPct", "refractoryTotalWtPct", 
                    "gpFormersWtPct", "atomicCompositionJson"
                ],
                return_references=[
                    QueryReference(
                        link_on="hasComposition",
                        return_references=[
                            QueryReference(
                                link_on="hasComponent",
                                return_references=[
                                    QueryReference(link_on="hasElement", return_properties=["symbol"]),
                                    QueryReference(link_on="hasMassFraction", return_properties=["numericValue", "nominal"])
                                ]
                            )
                        ]
                    )
                ]
            )
            
            if not response.objects:
                return "No similar alloys found in Knowledge Graph."

            candidates = []
            
            for obj in response.objects:
                props = obj.properties
                candidate = {
                    "uuid": obj.uuid,
                    "name": props.get("name", "Unknown"),
                    "processing": props.get("processingMethod", "Unknown"),
                    "density": props.get("densityCalculated", "N/A"),
                    "gamma_prime": props.get("gammaPrimeEstimate", "N/A"),
                    "md_avg": props.get("mdAverage", "N/A"),
                    "tcp_risk": props.get("tcpRisk", "N/A"),
                    "sss_wt_pct": props.get("sssTotalWtPct", "N/A"),
                    "refractory_wt_pct": props.get("refractoryTotalWtPct", "N/A"),
                    "gp_formers_wt_pct": props.get("gpFormersWtPct", "N/A"),
                    "composition": {},
                    "properties": {} 
                }

                try:
                    if obj.references and "hasComposition" in obj.references:
                        comp_ref_obj = obj.references["hasComposition"].objects[0]
                        if comp_ref_obj.references and "hasComponent" in comp_ref_obj.references:
                             for comp_node in comp_ref_obj.references["hasComponent"].objects:
                                 if "hasElement" in comp_node.references and "hasMassFraction" in comp_node.references:
                                     sym = comp_node.references["hasElement"].objects[0].properties["symbol"]
                                     mf_node = comp_node.references["hasMassFraction"].objects[0]
                                     wt = mf_node.properties.get("numericValue")
                                     if wt is None: wt = mf_node.properties.get("nominal", 0.0)
                                     if wt is not None:
                                         candidate["composition"][sym] = float(wt)
                except Exception:
                    pass

                ac_json = props.get("atomicCompositionJson")
                if ac_json and isinstance(ac_json, str):
                    try:
                        candidate["_atomic_comp"] = json.loads(ac_json)
                    except Exception:
                        pass

                # Normalize compositions for fair comparison
                def normalize_comp(c):
                    total = sum(c.values())
                    if total == 0: return c
                    return {k: (v/total)*100 for k, v in c.items()}
                
                test_norm = normalize_comp(composition)
                cand_norm = normalize_comp(candidate["composition"])
                
                # Calculate distance on normalized compositions
                dist = 0.0
                all_keys = set(test_norm.keys()) | set(cand_norm.keys())
                for k in all_keys:
                    v1 = test_norm.get(k, 0.0)
                    v2 = cand_norm.get(k, 0.0)
                    dist += (v1 - v2) ** 2
                dist = dist ** 0.5
                
                candidate["_distance"] = dist
                candidates.append(candidate)

                if len(candidates) <= 3:
                     logger.debug(f"Distance calc: {candidate.get('name', 'Unknown')}: {dist:.4f} | Test(norm): {test_norm} | Cand(norm): {cand_norm}")

            candidates.sort(key=lambda x: x["_distance"])
            top_results = candidates[:limit]

            top_summaries = [f"{c.get('name')} (d={c.get('_distance', 'N/A')})" for c in top_results]
            logger.info(f"Top {limit} KG Results: {top_summaries}")

            for cand in top_results:
                try:
                    props = _fetch_properties_for_uuid(client, cand["uuid"])
                    cand["properties"] = props
                except Exception as e:
                    logging.debug(f"Failed to fetch properties for {cand.get('name')}: {e}")

            final_output = []
            for c in top_results:
                serialized = {
                    "name": c.get("name"),
                    "processing": c.get("processing"),
                    "_distance": c.get("_distance", 999.0),
                    "metallurgy": {
                        "md_avg": c.get("md_avg"),
                        "tcp_risk": c.get("tcp_risk"),
                        "density_gcm3": c.get("density"),
                        "gamma_prime_vol": c.get("gamma_prime"),
                        "sss_wt_pct": c.get("sss_wt_pct"),
                        "refractory_wt_pct": c.get("refractory_wt_pct")
                    },
                    "properties": {}
                }
                
                if "composition" in c:
                    serialized["composition_wt_pct"] = {str(k): round(float(v), 2) for k, v in c["composition"].items() if v is not None and v > 0}
                
                if "properties" in c:
                    for prop, measurements in c["properties"].items():
                        summary_list = []
                        for m in measurements:
                            val = round(m.get("val") or 0.0, 0)
                            temp = round(m.get("temp_c") or 20.0, 0)
                            unit = m.get("unit") or ""
                            
                            if "stress" in m:
                                stress = round(m.get("stress") or 0.0, 0)
                                summary_list.append(f"Life: {val}{unit}, Temp: {temp}C, Stress: {stress}MPa")
                            else:
                                if unit:
                                    summary_list.append(f"{val} {unit} @ {temp}C")
                                else:
                                    summary_list.append(f"{val} @ {temp}C")
                        serialized["properties"][prop] = ", ".join(summary_list)
                
                final_output.append(serialized)

            result_json = json.dumps(final_output, indent=2)
            _store_cached_search(composition, limit, result_json)

            return result_json

        except Exception as e:
            logger.error(f"RAG Search failed: {e}")
            return f"Error executing RAG search: {str(e)}"
        finally:
            if client:
                client.close()


def _fetch_properties_for_uuid(client, uuid):
    """Fetches mechanical properties for a given Variant UUID (Direct Query)."""
    variant_coll = client.collections.get("Variant")
    
    # Get the Variant properties directly
    v_obj = variant_coll.query.fetch_object_by_id(
        uuid,
        return_references=[
            QueryReference(
                link_on="hasPropertySet",
                return_references=[
                    QueryReference(
                        link_on="measuresProperty",
                        return_properties=["propertyType"]
                    ),
                    QueryReference(
                        link_on="hasMeasurement",
                        return_properties=["stress", "lifeHours"],
                        return_references=[
                            QueryReference(
                                link_on="hasQuantity", 
                                return_properties=["numericValue", "unitSymbol"]
                            ),
                            QueryReference(
                                link_on="hasTestCondition",
                                return_references=[
                                    QueryReference(
                                        link_on="hasTemperature",
                                        return_properties=["numericValue"]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )
    
    if not v_obj:
        return {}
    
    # 2. Results Container
    properties = {} 

    # 3. Process the single variant object
    if not v_obj.references or "hasPropertySet" not in v_obj.references:
        return {}
            
    for pset in v_obj.references["hasPropertySet"].objects:
        p_name = "Unknown"
        if pset.references and "measuresProperty" in pset.references:
            p_name = pset.references["measuresProperty"].objects[0].properties.get("propertyType", "Unknown")
        
        if pset.references and "hasMeasurement" in pset.references:
            for meas in pset.references["hasMeasurement"].objects:
                stress = meas.properties.get("stress")
                life_hours = meas.properties.get("lifeHours")
                
                val = 0.0
                unit = ""
                temp_c = 20
                
                if life_hours is not None:
                    val = life_hours 
                    unit = "h"
                elif meas.references and "hasQuantity" in meas.references:
                    quant = meas.references["hasQuantity"].objects[0]
                    val = quant.properties.get("numericValue", 0.0)
                    unit = quant.properties.get("unitSymbol", "")
                
                if meas.references and "hasTestCondition" in meas.references:
                    tc = meas.references["hasTestCondition"].objects[0]
                    if tc.references and "hasTemperature" in tc.references:
                        temp_c = tc.references["hasTemperature"].objects[0].properties.get("numericValue", 20)
                        
                if p_name not in properties:
                    properties[p_name] = []
                    
                entry = {"val": val, "temp_c": temp_c, "unit": unit}
                if stress is not None:
                    entry["stress"] = stress
                
                properties[p_name].append(entry)
    
    return properties
