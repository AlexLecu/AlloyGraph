from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Type, Dict, Any, Optional
import weaviate
from weaviate.classes.query import QueryReference
import os
import json
import ast
import logging
import hashlib

logger = logging.getLogger(__name__)

class AlloySearchInput(BaseModel):
    """Input for Weaviate search — composition-based or text-based."""
    model_config = {"extra": "ignore"}

    composition: str = Field(
        default="{}",
        description=(
            "JSON string of composition in wt%, e.g. '{\"Ni\": 60.0, \"Al\": 5.0}'. "
            "Pass '{}' for text search."
        ),
    )
    query: str = Field(
        "",
        description=(
            "Text search query — alloy name ('Waspaloy'), class ('disc alloys'), "
            "or description ('high strength wrought'). Pass empty string for composition search."
        ),
    )
    target_temperature_c: float = Field(
        0.0,
        description="Target temperature in °C. 0 = ignore. When set, adds a property_summary with closest measurement."
    )
    limit: int = Field(3, description="Number of results to return.")
    processing: str = Field(
        "",
        description=(
            "Processing route: 'wrought', 'cast', or empty. "
            "When set, results prefer alloys with matching processing."
        ),
    )

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

        # Unwrap composition (could be schema-echo, nested dict, or null)
        comp = normalized.get("composition")
        if comp is None:
            normalized["composition"] = {}
        elif isinstance(comp, dict):
            for key in ("value", "composition", "description"):
                if key in comp:
                    normalized["composition"] = comp[key]
                    break

        # Unwrap query (could be schema-echo)
        q = normalized.get("query")
        if isinstance(q, dict):
            for key in ("value", "query", "default"):
                if key in q:
                    normalized["query"] = q[key]
                    break
            else:
                normalized.pop("query", None)

        # Unwrap target_temperature_c (could be schema-echo)
        ttc = normalized.get("target_temperature_c")
        if isinstance(ttc, dict):
            for key in ("value", "default"):
                if key in ttc:
                    normalized["target_temperature_c"] = ttc[key]
                    break
            else:
                normalized.pop("target_temperature_c", None)

        # Unwrap limit (could be schema-echo)
        lim = normalized.get("limit")
        if isinstance(lim, dict):
            for key in ("value", "limit", "default"):
                if key in lim:
                    normalized["limit"] = lim[key]
                    break
            else:
                normalized.pop("limit", None)

        proc = normalized.get("processing")
        if isinstance(proc, dict):
            for key in ("value", "processing", "default"):
                if key in proc:
                    normalized["processing"] = proc[key]
                    break
            else:
                normalized.pop("processing", None)

        return normalized

    @field_validator("composition", mode="before")
    @classmethod
    def _parse_composition(cls, value: Any) -> str:
        if value is None or value == {} or value == "{}":
            return "{}"

        if isinstance(value, dict):
            if "description" in value and isinstance(value["description"], str):
                value = value["description"]
            else:
                return json.dumps({str(k): float(v) for k, v in value.items()})

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return "{}"
            try:
                parsed = json.loads(text.replace("'", '"'))
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception as e:
                    raise ValueError("Invalid composition format; expected dict or JSON string.") from e

            if not isinstance(parsed, dict):
                raise ValueError("Invalid composition format; expected a dict.")
            return json.dumps({str(k): float(v) for k, v in parsed.items()})

        raise ValueError("Invalid composition format; expected dict or JSON string.")


# ============================================================
# KG SEARCH CACHING
# ============================================================
def _create_cache_key(composition: Dict[str, float], limit: int,
                      target_temperature_c: float = 0.0, processing: str = "") -> str:
    """
    Create stable hash key for composition + limit + temperature + processing.

    Normalizes composition to handle rounding variations:
    - {Ni: 60.0, Al: 5.0} and {Ni: 60.01, Al: 4.99} → same key
    """
    # Normalize to 2 decimal places and sort for consistent hashing
    normalized = {k: round(v, 2) for k, v in composition.items() if v > 0}
    sorted_comp = json.dumps(normalized, sort_keys=True)
    cache_str = f"{sorted_comp}|{limit}|{target_temperature_c:.0f}|{processing}"
    return hashlib.md5(cache_str.encode()).hexdigest()

_kg_search_cache = {}
_cache_hits = 0
_cache_misses = 0


def _get_cached_search(composition: Dict[str, float], limit: int,
                       target_temperature_c: float = 0.0, processing: str = "") -> tuple:
    """
    Get cached KG search result if available.
    """
    global _cache_hits, _cache_misses

    cache_key = _create_cache_key(composition, limit, target_temperature_c, processing)

    if cache_key in _kg_search_cache:
        _cache_hits += 1
        logger.debug("KG Cache HIT (hits=%d, misses=%d)", _cache_hits, _cache_misses)
        return _kg_search_cache[cache_key], True
    else:
        _cache_misses += 1
        logger.debug("KG Cache MISS (hits=%d, misses=%d)", _cache_hits, _cache_misses)
        return None, False


def _store_cached_search(composition: Dict[str, float], limit: int, result: str,
                         target_temperature_c: float = 0.0, processing: str = ""):
    """Store KG search result in cache."""
    global _kg_search_cache

    cache_key = _create_cache_key(composition, limit, target_temperature_c, processing)

    # FIFO eviction: if cache exceeds 128 entries, remove oldest
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
    name: str = "AlloySearchTool"
    description: str = (
        "Searches the Weaviate Knowledge Graph for alloys. Two modes:\n"
        "1. Composition search: pass composition={'Ni': 58.0, 'Cr': 19.5} to find similar alloys.\n"
        "2. Text search: pass query='Waspaloy' or query='disc alloys' to search by name/class.\n"
        "Optionally set target_temperature_c to get property values at a specific temperature."
    )
    args_schema: Type[BaseModel] = AlloySearchInput

    def _run(
        self,
        composition: Any = None,
        query: str = "",
        target_temperature_c: float = 0.0,
        limit: int = 3,
        processing: str = "",
        **kwargs: Any,
    ) -> Any:
        if isinstance(composition, str):
            composition = json.loads(composition) if composition and composition != "{}" else {}
        if not composition and not query:
            return json.dumps({"error": "Provide either 'composition' or 'query'.", "results": []})

        # Composition mode: check cache
        if composition and not query:
            cached_result, cache_hit = _get_cached_search(composition, limit, target_temperature_c, processing)
            if cache_hit:
                return cached_result

        client = None
        try:
            client = _connect_weaviate()
            if not client.is_live():
                return json.dumps({"error": "Weaviate instance is not reachable.", "results": []})

            collection = client.collections.get("Variant")

            # Build query string
            if query:
                query_str = query
                fetch_limit = limit * 2  # text search needs less overfetch
            else:
                query_parts = [f"{v}% {k}" for k, v in composition.items() if v > 1.0]
                query_str = "Superalloy with " + ", ".join(query_parts)
                fetch_limit = 50  # composition needs wide net for distance reranking

            # Shared return schema — includes property refs for text mode (eager fetch)
            return_props = [
                "name", "processingMethod", "densityCalculated", "gammaPrimeEstimate",
                "mdAverage", "tcpRisk", "sssTotalWtPct", "refractoryTotalWtPct",
                "gpFormersWtPct"
            ]
            comp_refs = [
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
            # Text mode: fetch properties inline (eager) to avoid N+1 queries
            if query:
                comp_refs.append(
                    QueryReference(
                        link_on="hasPropertySet",
                        return_references=[
                            QueryReference(link_on="measuresProperty", return_properties=["propertyType"]),
                            QueryReference(
                                link_on="hasMeasurement",
                                return_properties=["stress", "lifeHours"],
                                return_references=[
                                    QueryReference(link_on="hasQuantity", return_properties=["numericValue", "unitSymbol"]),
                                    QueryReference(
                                        link_on="hasTestCondition",
                                        return_references=[
                                            QueryReference(link_on="hasTemperature", return_properties=["numericValue"])
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                )

            response = collection.query.hybrid(
                query=query_str,
                limit=fetch_limit,
                return_properties=return_props,
                return_references=comp_refs,
            )

            if not response.objects:
                return json.dumps({"error": f"No alloys found for '{query or query_str}'.", "results": []})

            # --- Build candidates ---
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
                    "properties": {},
                }

                # Extract composition from graph refs
                _extract_composition(obj, candidate)

                # Composition mode: compute Euclidean distance for reranking
                if composition:
                    candidate["_distance"] = _composition_distance(composition, candidate["composition"])
                else:
                    candidate["_distance"] = 0.0  # text mode: keep hybrid relevance order

                # Text mode: extract properties inline (eager)
                if query:
                    _extract_properties_inline(obj, candidate)

                candidates.append(candidate)

            # Composition mode: rerank by distance
            if composition:
                candidates.sort(key=lambda x: x["_distance"])

            if processing and composition:
                proc_lower = processing.lower()

                def _proc_compatible(cand):
                    cand_proc = (cand.get("processing") or "").lower()
                    return proc_lower in cand_proc or cand_proc in proc_lower

                compatible = [c for c in candidates if _proc_compatible(c)]
                incompatible = [c for c in candidates if not _proc_compatible(c)]

                if compatible:
                    candidates = compatible + incompatible
                    logger.info(
                        "KG: Processing filter '%s': %d compatible, %d incompatible of %d total",
                        processing, len(compatible), len(incompatible), len(compatible) + len(incompatible)
                    )

            top_results = candidates[:limit]
            logger.info("Top %d KG: %s", limit, [(c.get("name"), round(c["_distance"], 2)) for c in top_results])

            # Composition mode: lazy-fetch properties via UUID
            if composition and not query:
                for cand in top_results:
                    try:
                        cand["properties"] = _fetch_properties_for_uuid(client, cand["uuid"])
                    except Exception as e:
                        logger.debug("Failed to fetch properties for %s: %s", cand.get('name'), e)

            # --- Serialize output ---
            final_output = []
            for c in top_results:
                serialized = {
                    "name": c.get("name"),
                    "processing": c.get("processing"),
                    "_distance": round(c.get("_distance", 999.0), 2),
                    "metallurgy": {
                        "md_avg": c.get("md_avg"),
                        "tcp_risk": c.get("tcp_risk"),
                        "density_gcm3": c.get("density"),
                        "gamma_prime_vol": c.get("gamma_prime"),
                        "sss_wt_pct": c.get("sss_wt_pct"),
                        "refractory_wt_pct": c.get("refractory_wt_pct"),
                    },
                    "properties": {},
                }

                if c.get("composition"):
                    serialized["composition_wt_pct"] = {
                        str(k): round(float(v), 2)
                        for k, v in c["composition"].items() if v is not None and v > 0
                    }

                # Format properties
                if c.get("properties"):
                    for prop_name, measurements in c["properties"].items():
                        serialized["properties"][prop_name] = _format_measurements(measurements)

                # Temperature-matched summary (when target_temperature_c is set)
                if target_temperature_c and c.get("properties"):
                    serialized["property_summary"] = _pick_closest_temp(
                        c["properties"], target_temperature_c
                    )

                final_output.append(serialized)

            result_json = json.dumps(final_output, indent=2)

            # Cache composition-mode results only
            if composition and not query:
                _store_cached_search(composition, limit, result_json, target_temperature_c, processing)

            return result_json

        except Exception as e:
            logger.error("KG Search failed: %s", e)
            return json.dumps({"error": str(e), "results": []})
        finally:
            if client:
                client.close()


# ============================================================
# SHARED HELPERS
# ============================================================

def _connect_weaviate():
    """Create a Weaviate client from environment variables."""
    return weaviate.connect_to_local(
        host=os.getenv('WEAVIATE_HOST', 'localhost'),
        port=int(os.getenv('WEAVIATE_PORT', 8081)),
        grpc_port=int(os.getenv('WEAVIATE_GRPC_PORT', 50052)),
    )


def _extract_composition(obj, candidate: dict):
    """Extract composition from Weaviate graph references into candidate dict."""
    try:
        if obj.references and "hasComposition" in obj.references:
            comp_ref = obj.references["hasComposition"].objects[0]
            if comp_ref.references and "hasComponent" in comp_ref.references:
                for comp_node in comp_ref.references["hasComponent"].objects:
                    if "hasElement" in comp_node.references and "hasMassFraction" in comp_node.references:
                        sym = comp_node.references["hasElement"].objects[0].properties["symbol"]
                        mf_node = comp_node.references["hasMassFraction"].objects[0]
                        wt = mf_node.properties.get("numericValue")
                        if wt is None:
                            wt = mf_node.properties.get("nominal", 0.0)
                        if wt is not None:
                            candidate["composition"][sym] = float(wt)
    except Exception as e:
        logger.warning("Failed to extract composition for %s: %s", candidate.get("name", "?"), e)


def _extract_property_sets(pset_objects) -> dict:
    """Extract properties from a list of PropertySet graph objects.

    Shared by text-mode (eager inline fetch) and composition-mode (lazy UUID fetch).
    Returns {prop_type: [{val, temp_c, unit, ?stress}, ...]}.
    """
    properties = {}
    for pset in pset_objects:
        prop_type = "Unknown"
        if pset.references and "measuresProperty" in pset.references:
            prop_type = pset.references["measuresProperty"].objects[0].properties.get("propertyType", "Unknown")

        if not (pset.references and "hasMeasurement" in pset.references):
            continue

        for meas in pset.references["hasMeasurement"].objects:
            val, unit, temp_c = 0.0, "", 20.0
            stress = meas.properties.get("stress")
            life_hours = meas.properties.get("lifeHours")

            if life_hours is not None:
                val, unit = life_hours, "h"
            elif meas.references and "hasQuantity" in meas.references:
                quant = meas.references["hasQuantity"].objects[0]
                val = quant.properties.get("numericValue", 0.0)
                unit = quant.properties.get("unitSymbol", "")

            if meas.references and "hasTestCondition" in meas.references:
                tc = meas.references["hasTestCondition"].objects[0]
                if tc.references and "hasTemperature" in tc.references:
                    temp_c = tc.references["hasTemperature"].objects[0].properties.get("numericValue", 20.0)

            if prop_type not in properties:
                properties[prop_type] = []
            entry = {"val": val, "temp_c": temp_c, "unit": unit}
            if stress is not None:
                entry["stress"] = stress
            properties[prop_type].append(entry)
    return properties


def _extract_properties_inline(obj, candidate: dict):
    """Extract properties from eagerly-fetched graph refs (text search mode)."""
    if not (obj.references and "hasPropertySet" in obj.references):
        return
    candidate["properties"] = _extract_property_sets(obj.references["hasPropertySet"].objects)


def _composition_distance(target: Dict[str, float], candidate_comp: Dict[str, float]) -> float:
    """Euclidean distance on normalized compositions."""
    def normalize(c):
        total = sum(c.values())
        return {k: (v / total) * 100 for k, v in c.items()} if total > 0 else c

    t_norm = normalize(target)
    c_norm = normalize(candidate_comp)
    all_keys = set(t_norm) | set(c_norm)
    return sum((t_norm.get(k, 0) - c_norm.get(k, 0)) ** 2 for k in all_keys) ** 0.5


def _format_measurements(measurements: list) -> str:
    """Format a list of measurement dicts into a compact summary string."""
    parts = []
    for m in measurements:
        val = round(m.get("val") or 0.0)
        temp = round(m.get("temp_c") or 20.0)
        unit = m.get("unit") or ""
        if "stress" in m:
            stress = round(m.get("stress") or 0.0)
            parts.append(f"Life: {val}{unit}, Temp: {temp}C, Stress: {stress}MPa")
        elif unit:
            parts.append(f"{val} {unit} @ {temp}C")
        else:
            parts.append(f"{val} @ {temp}C")
    return ", ".join(parts)


def _pick_closest_temp(properties: dict, target_temp: float) -> dict:
    """For each property, pick the measurement closest to target temperature."""
    summary = {}
    for prop_type, measurements in properties.items():
        best, best_diff = None, float("inf")
        for m in measurements:
            diff = abs((m.get("temp_c") or 20.0) - target_temp)
            if diff < best_diff:
                best_diff = diff
                best = m
        if best and best.get("val") is not None:
            summary[prop_type] = {
                "value": round(best["val"], 1),
                "unit": best.get("unit", ""),
                "at_temperature_c": best.get("temp_c", target_temp),
            }
    return summary


def _fetch_properties_for_uuid(client, uuid):
    """Fetch properties for a Variant UUID via separate query (composition-mode lazy fetch)."""
    variant_coll = client.collections.get("Variant")
    v_obj = variant_coll.query.fetch_object_by_id(
        uuid,
        return_references=[
            QueryReference(
                link_on="hasPropertySet",
                return_references=[
                    QueryReference(link_on="measuresProperty", return_properties=["propertyType"]),
                    QueryReference(
                        link_on="hasMeasurement",
                        return_properties=["stress", "lifeHours"],
                        return_references=[
                            QueryReference(link_on="hasQuantity", return_properties=["numericValue", "unitSymbol"]),
                            QueryReference(
                                link_on="hasTestCondition",
                                return_references=[
                                    QueryReference(link_on="hasTemperature", return_properties=["numericValue"])
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )
    if not v_obj or not v_obj.references or "hasPropertySet" not in v_obj.references:
        return {}
    return _extract_property_sets(v_obj.references["hasPropertySet"].objects)
