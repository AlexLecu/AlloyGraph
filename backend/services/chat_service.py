import json
import re
import unicodedata
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Optional

from .alloy_retriever import AlloyRetriever, AlloyData
from .config import (
    LLMConfig,
    SearchConfig,
    HistoryConfig,
    Prompts,
    PROPERTY_SEARCH_TERMS,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _map_property_term(prop_target: str) -> str:
    """Map user property terms to search keywords (first match wins)."""
    prop_lower = prop_target.lower()
    for key, value in PROPERTY_SEARCH_TERMS:
        if key in prop_lower:
            return value
    return prop_lower


def _extract_property_value(
    alloy: AlloyData, search_term: str, prefer_room_temp: bool = True
) -> Optional[float]:
    """Extract best property value from an alloy for the given search term."""
    if "density" in search_term and alloy.density_gcm3:
        return alloy.density_gcm3

    if not alloy.properties:
        return None

    room_temp_val = None
    best_non_rt = None       # (temperature_distance_from_RT, value)

    for p in alloy.properties:
        p_type = p.property_type.lower()

        is_match = False
        if "yield" in search_term and ("yield" in p_type or "0.2%" in p_type):
            is_match = True
        elif "tensile" in search_term and ("ultimate" in p_type or "tensile" in p_type or "uts" in p_type):
            if "yield" not in p_type and "0.2%" not in p_type:
                is_match = True
        elif "elongation" in search_term and "elongation" in p_type:
            is_match = True
        elif "elastic" in search_term and ("elastic" in p_type or "modulus" in p_type):
            is_match = True
        elif "creep" in search_term and ("creep" in p_type or "rupture" in p_type):
            is_match = True
        elif "hardness" in search_term and "hardness" in p_type:
            is_match = True
        elif search_term in p_type:
            is_match = True

        if is_match and p.value is not None:
            # Treat None temperature as room temperature (most datasheets
            # omit explicit temperature when reporting RT values)
            temp = p.temperature_c if p.temperature_c is not None else 22.0
            is_room = 20 <= temp <= 25

            if is_room and prefer_room_temp:
                if room_temp_val is None or p.value > room_temp_val:
                    room_temp_val = p.value
            else:
                # Prefer the value closest to room temperature
                dist = abs(temp - 22.0)
                if best_non_rt is None or dist < best_non_rt[0]:
                    best_non_rt = (dist, p.value)

    if room_temp_val is not None:
        return room_temp_val
    return best_non_rt[1] if best_non_rt is not None else None


def _format_history(history: list, depth: int, max_content: int = 100) -> str:
    """Format conversation history for LLM context."""
    hist_str = ""
    for msg in reversed(history[-depth:]):
        role = msg.get("role", "")
        content = msg.get("content", "")[:max_content]
        if role in ("user", "assistant"):
            hist_str = f"{role}: {content}\n" + hist_str
    return hist_str


def _normalize_text(text: str) -> str:
    """Normalize text by removing accents and converting to lowercase."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn").lower()


def _extract_core_name(alloy_name: str) -> str:
    """Extract the core alloy name, removing processing suffixes."""
    name = alloy_name.lower().replace("*", "").strip()
    name = re.sub(r"\(forged\)|\(cast\)", "", name)
    # Word boundaries prevent partial matches inside names like "Nimocast"
    for word in ["wrought", "cast", "bar", "sheet", "plate", "forged"]:
        name = re.sub(rf"\b{word}\b", "", name)
    return name.strip()


def find_best_match(target_name: str, search_results: list[AlloyData]) -> Optional[AlloyData]:
    """Find the best matching alloy from search results based on similarity."""
    if not search_results:
        return None

    clean_target = _normalize_text(target_name.strip())

    scored = []
    for alloy in search_results:
        clean_alloy_name = _normalize_text(_extract_core_name(alloy.name))

        ratio = SequenceMatcher(None, clean_target, clean_alloy_name).ratio()

        if clean_target in clean_alloy_name:
            ratio += SearchConfig.SUBSTRING_MATCH_BOOST

        scored.append((ratio, alloy))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_alloy = scored[0]

    if best_score >= SearchConfig.MIN_SIMILARITY_THRESHOLD:
        return best_alloy
    return None


# ── Focused context builder ─────────────────────────────────────────────

def _format_focused_context(alloys: list[AlloyData]) -> str:
    """Build LLM context with all available data for each alloy.

    Always includes every non-empty section so the LLM never misses
    relevant information.  The per-alloy overhead is small (~300 tokens)
    and far outweighs the risk of hiding data behind keyword heuristics.
    """
    if not alloys:
        return "No matching alloys found in the knowledge graph."

    results = []
    for alloy in alloys:
        lines = [
            f"\n{'='*50}",
            f"Alloy: {alloy.name}",
            f"Processing: {alloy.processing_method}",
            f"{'='*50}",
        ]

        # ── Composition ──────────────────────────────────────────────
        if alloy.composition:
            lines.append("\nComposition (wt%):")
            for el, val in sorted(alloy.composition.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {el}: {val:.2f}%")

        if alloy.atomic_composition:
            lines.append("\nAtomic Composition (at%):")
            for el, val in sorted(alloy.atomic_composition.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {el}: {val:.2f}%")

        if alloy.gamma_composition:
            lines.append("\nGamma (Matrix) Phase (at%):")
            for el, val in sorted(alloy.gamma_composition.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {el}: {val:.2f}%")

        if alloy.gamma_prime_composition:
            lines.append("\nGamma Prime (Precipitate) Phase (at%):")
            for el, val in sorted(alloy.gamma_prime_composition.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {el}: {val:.2f}%")

        # ── Physical Properties ──────────────────────────────────────
        phys = []
        if alloy.density_gcm3:
            phys.append(f"Density: {alloy.density_gcm3:.2f} g/cm³")
        if alloy.gamma_prime_vol_pct:
            phys.append(f"γ' Volume Fraction: {alloy.gamma_prime_vol_pct:.1f}%")
        if alloy.lattice_mismatch_pct is not None:
            phys.append(f"Lattice Mismatch: {alloy.lattice_mismatch_pct:.3f}%")
        if phys:
            lines.append("\nPhysical Properties:")
            for p in phys:
                lines.append(f"  {p}")

        # ── Phase Stability ──────────────────────────────────────────
        stab = []
        if alloy.md_avg is not None:
            stab.append(f"Md (avg): {alloy.md_avg:.3f}")
        if alloy.md_gamma is not None:
            stab.append(f"Md (γ matrix): {alloy.md_gamma:.3f}")
        if alloy.vec_avg is not None:
            stab.append(f"VEC (avg): {alloy.vec_avg:.2f}")
        if alloy.tcp_risk:
            stab.append(f"TCP Risk: {alloy.tcp_risk}")
        if stab:
            lines.append("\nPhase Stability:")
            for s in stab:
                lines.append(f"  {s}")

        # ── Strengthening Mechanisms ─────────────────────────────────
        sp = []
        if alloy.sss_wt_pct is not None:
            sp.append(f"SSS Elements: {alloy.sss_wt_pct:.1f} wt%")
        if alloy.sss_coefficient is not None:
            sp.append(f"SSS Coefficient: {alloy.sss_coefficient:.4f}")
        if alloy.precipitation_hardening_coeff is not None:
            sp.append(f"Precipitation Hardening: {alloy.precipitation_hardening_coeff:.4f}")
        if alloy.creep_resistance_param is not None:
            sp.append(f"Creep Resistance: {alloy.creep_resistance_param:.2f}")
        if sp:
            lines.append("\nStrengthening Mechanisms:")
            for s in sp:
                lines.append(f"  {s}")

        # ── Element Ratios ───────────────────────────────────────────
        rats = []
        if alloy.al_ti_ratio is not None:
            rats.append(f"Al/Ti (wt): {alloy.al_ti_ratio:.2f}")
        if alloy.al_ti_at_ratio is not None:
            rats.append(f"Al/Ti (at): {alloy.al_ti_at_ratio:.2f}")
        if alloy.cr_co_ratio is not None:
            rats.append(f"Cr/Co: {alloy.cr_co_ratio:.2f}")
        if alloy.cr_ni_ratio is not None:
            rats.append(f"Cr/Ni: {alloy.cr_ni_ratio:.3f}")
        if alloy.mo_w_ratio is not None:
            rats.append(f"Mo/W: {alloy.mo_w_ratio:.2f}")
        if rats:
            lines.append("\nElement Ratios:")
            for r in rats:
                lines.append(f"  {r}")

        # ── Composition Metrics ──────────────────────────────────────
        cm = []
        if alloy.refractory_wt_pct is not None:
            cm.append(f"Refractory Elements: {alloy.refractory_wt_pct:.1f} wt%")
        if alloy.gp_formers_wt_pct is not None:
            cm.append(f"γ' Formers: {alloy.gp_formers_wt_pct:.1f} wt%")
        if alloy.oxidation_resistance is not None:
            cm.append(f"Oxidation Resistance Index: {alloy.oxidation_resistance:.2f}")
        if cm:
            lines.append("\nComposition Metrics:")
            for c in cm:
                lines.append(f"  {c}")

        # ── Mechanical Properties ────────────────────────────────────
        if alloy.properties:
            lines.append("\nMechanical Properties:")
            prop_groups: dict[str, list] = {}
            for prop in alloy.properties:
                if prop.property_type not in prop_groups:
                    prop_groups[prop.property_type] = []
                prop_groups[prop.property_type].append(prop)
            for prop_type, measurements in prop_groups.items():
                lines.append(f"  {prop_type}:")
                for m in measurements:
                    lines.append(f"    • {m.format()}")

        results.append("\n".join(lines))

    return "\n\n".join(results)


# ── Routing (single LLM call) ───────────────────────────────────────────

def route_query(prompt: str, history: list) -> dict:
    """Classify intent AND extract alloy names in a single LLM call.

    Returns: {"intent": str, "alloys": list[str], "params": dict}
    """
    client = LLMConfig.get_client()
    if not client:
        return {"intent": "SEARCH", "alloys": [], "params": {}}

    try:
        hist_str = _format_history(
            history, HistoryConfig.EXTRACTION_CONTEXT_DEPTH, max_content=200,
        )

        completion = client.chat.completions.create(
            model=LLMConfig.MODEL,
            messages=[
                {"role": "system", "content": Prompts.ROUTE_AND_EXTRACT},
                {"role": "user", "content": f"History:\n{hist_str}\nQuery: {prompt}"},
            ],
            temperature=LLMConfig.ROUTING_TEMPERATURE,
            max_tokens=LLMConfig.ROUTING_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        data = json.loads(completion.choices[0].message.content)
        alloys = data.get("alloys", [])
        if isinstance(alloys, str):
            alloys = [a.strip() for a in alloys.split(",") if a.strip() and a.strip().lower() != "none"]
        return {
            "intent": data.get("intent", "SEARCH"),
            "alloys": alloys,
            "params": data.get("params", {}),
        }
    except Exception as e:
        print(f"Warning: route_query failed: {e}")
        return {"intent": "SEARCH", "alloys": [], "params": {}}


# ── Deduplication helper ─────────────────────────────────────────────────

def _deduplicate_by_name(scored_alloys: list[tuple]) -> list[tuple]:
    """Keep only the best-scoring variant per core alloy name.

    Input: list of tuples where the first element is an AlloyData object.
    Returns: filtered list with duplicates removed (first occurrence wins).
    """
    seen_cores: set[str] = set()
    result = []
    for item in scored_alloys:
        core = _extract_core_name(item[0].name)
        if core not in seen_cores:
            seen_cores.add(core)
            result.append(item)
    return result


# ── Analytics / Target processors ────────────────────────────────────────

def process_analytics_query(params: dict, retriever: AlloyRetriever) -> list[AlloyData]:
    """Handle analytical queries by fetching candidates and sorting."""
    prop_target = params.get("property", "")
    direction = params.get("direction", "highest")
    limit = params.get("limit", 5)

    search_term = _map_property_term(prop_target)
    candidates = retriever.get_alloys_with_property(search_term, limit=100)

    scored = []
    for alloy in candidates:
        value = _extract_property_value(alloy, search_term)
        if value is not None:
            scored.append((alloy, value))

    scored.sort(key=lambda x: x[1], reverse=(direction == "highest"))
    scored = _deduplicate_by_name(scored)
    return [alloy for alloy, _ in scored[:limit]]


def process_target_query(params: dict, retriever: AlloyRetriever) -> list[AlloyData]:
    """Find alloys with properties closest to a target value."""
    prop_target = params.get("property", "")
    target_value = params.get("target_value", 0)
    tolerance_pct = params.get("tolerance_pct", 20)
    limit = params.get("limit", 3)

    if target_value is None:
        return []

    search_term = _map_property_term(prop_target)
    candidates = retriever.get_alloys_with_property(search_term, limit=100)

    scored = []
    for alloy in candidates:
        value = _extract_property_value(alloy, search_term)
        if value is not None:
            distance = abs(value - target_value)
            pct_diff = (distance / target_value) * 100 if target_value else 0
            scored.append((alloy, distance, pct_diff))

    scored.sort(key=lambda x: x[1])
    scored = _deduplicate_by_name(scored)

    # Drop extreme outliers: keep only alloys within tolerance range
    if tolerance_pct and len(scored) > limit:
        filtered = [(a, d, p) for a, d, p in scored if p <= tolerance_pct]
        if len(filtered) >= limit:
            scored = filtered

    return [alloy for alloy, _, _ in scored[:limit]]


# ── Main streaming generator ────────────────────────────────────────────

def stream_chat_response(prompt: str, session_id: str, history: list):
    """Generator that streams alloy data + LLM response as NDJSON chunks."""

    try:
        yield from _stream_chat_inner(prompt, session_id, history)
    except Exception as e:
        print(f"Stream error: {e}")
        yield json.dumps({"type": "error", "content": f"Something went wrong: {e}"}) + "\n"


def _stream_chat_inner(prompt: str, session_id: str, history: list):
    """Inner generator — separated so the outer can catch and yield errors."""

    with AlloyRetriever() as retriever:
        # ── 1. Single routing call (intent + alloy extraction) ───────
        route = route_query(prompt, history)
        intent = route["intent"]
        extracted_alloys = route["alloys"]
        params = route["params"]

        target_alloys: list[AlloyData] = []
        final_context = ""

        # ── 2. Handle each intent ───────────────────────────────────
        if intent == "CONVERSATION":
            yield json.dumps({"type": "data", "alloys": []}) + "\n"
            client = LLMConfig.get_client()
            if client:
                conv_prompt = (
                    "You are a friendly alloy research assistant. The user sent a casual/conversational message. "
                    "Respond briefly and warmly, then remind them you can help with: "
                    "looking up alloy compositions and properties, comparing alloys, "
                    "finding alloys with specific characteristics, or answering questions about superalloys. "
                    "Keep it short (1-2 sentences)."
                )
                stream = client.chat.completions.create(
                    model=LLMConfig.MODEL,
                    messages=[
                        {"role": "system", "content": conv_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=150,
                    temperature=0.7,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield json.dumps({"type": "chunk", "content": content}) + "\n"
            else:
                yield json.dumps({
                    "type": "chunk",
                    "content": "Hello! I'm an alloy research assistant. Ask me about alloy compositions, properties, or comparisons!",
                }) + "\n"
            return

        elif intent == "ANALYTICS":
            print(f"Analytics query: {params}")
            prop = params.get("property", "yield strength")
            direction = params.get("direction", "highest")

            if extracted_alloys:
                # Named alloys + ranking → look up each, then sort
                for name in extracted_alloys:
                    matches = retriever.search_alloys(name, limit=5)
                    best = find_best_match(name, matches)
                    if best and not any(a.name == best.name for a in target_alloys):
                        target_alloys.append(best)
                # Sort by the requested property
                search_term = _map_property_term(prop)
                scored = []
                for a in target_alloys:
                    val = _extract_property_value(a, search_term)
                    scored.append((a, val if val is not None else float('-inf')))
                scored.sort(key=lambda x: x[1], reverse=(direction == "highest"))
                target_alloys = [a for a, _ in scored]
                final_context = f"Comparison by {prop} ({direction}):\n"
            else:
                # No specific alloys → bulk search
                target_alloys = process_analytics_query(params, retriever)
                final_context = f"Top results for {prop} ({direction}):\n"

            final_context += _format_focused_context(target_alloys)

        elif intent == "TARGET":
            target_val = params.get("target_value")
            prop_name = params.get("property", "yield strength")

            reference_alloy_name = None
            if target_val is None and extracted_alloys:
                reference_alloy_name = extracted_alloys[0]
                search_term = _map_property_term(prop_name)
                ref_results = retriever.search_alloys(reference_alloy_name, limit=5)
                ref_alloy = find_best_match(reference_alloy_name, ref_results)
                if ref_alloy:
                    resolved = _extract_property_value(ref_alloy, search_term)
                    if resolved is not None:
                        target_val = resolved
                        params["target_value"] = target_val
                        print(f"Resolved target from {ref_alloy.name}: {prop_name} = {target_val}")

            print(f"Target query: {params}")

            if target_val is not None:
                target_alloys = process_target_query(params, retriever)

                if reference_alloy_name and target_alloys:
                    ref_core = _extract_core_name(reference_alloy_name)
                    target_alloys = [
                        a for a in target_alloys
                        if _extract_core_name(a.name) != ref_core
                    ]

                display_val = f"{target_val}"
                if reference_alloy_name:
                    final_context = f"Alloys with {prop_name} similar to {reference_alloy_name} ({display_val}):\n"
                else:
                    final_context = f"Alloys with {prop_name} closest to {display_val}:\n"
                final_context += _format_focused_context(target_alloys)
            else:
                print(f"TARGET fallback: no target_value, using ANALYTICS for {prop_name}")
                direction = "lowest" if "density" in prop_name.lower() else "highest"
                fallback_params = {"property": prop_name, "direction": direction, "limit": 5}
                target_alloys = process_analytics_query(fallback_params, retriever)
                final_context = f"Could not determine target value. Top alloys by {prop_name} ({direction}):\n"
                final_context += _format_focused_context(target_alloys)

        elif intent == "DESIGN":
            final_context = "User wants to design an alloy. Encourage them to use the Designer tool."

        else:
            # ── SEARCH intent ────────────────────────────────────────
            if extracted_alloys:
                found_names = set()
                for name in extracted_alloys:
                    matches = retriever.search_alloys(name, limit=5)
                    best = find_best_match(name, matches)
                    if best and not any(a.name == best.name for a in target_alloys):
                        target_alloys.append(best)
                        found_names.add(name)

                missing = [n for n in extracted_alloys if n not in found_names]
                if missing:
                    for name in missing:
                        extra = retriever.search_alloys(f"{name} alloy", limit=5)
                        best = find_best_match(name, extra)
                        if best and not any(a.name == best.name for a in target_alloys):
                            target_alloys.append(best)
            else:
                raw_results = retriever.search_alloys(prompt, limit=5)
                if raw_results:
                    target_alloys = raw_results

            final_context = _format_focused_context(target_alloys)

        # ── 3. Send alloy data chunk ─────────────────────────────────
        serialized = [asdict(a) for a in target_alloys] if target_alloys else []
        yield json.dumps({"type": "data", "alloys": serialized}) + "\n"

        if intent == "DESIGN":
            yield json.dumps({"type": "tool_suggestion", "tool": "designer"}) + "\n"

        if not target_alloys and intent != "DESIGN":
            # Fallback: answer using general LLM knowledge (discussion questions)
            client = LLMConfig.get_client()
            if client:
                fallback_prompt = (
                    "You are a materials science expert specialising in nickel-based superalloys. "
                    "The knowledge graph returned no matching alloys for this query, but it may be a "
                    "conceptual or discussion question about metallurgy. Answer using your expert knowledge. "
                    "If the question truly requires specific alloy data you don't have, say so."
                )
                messages = [{"role": "system", "content": fallback_prompt}]
                for msg in history[-HistoryConfig.MAX_CONTEXT_MESSAGES:]:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })
                messages.append({"role": "user", "content": prompt})
                stream = client.chat.completions.create(
                    model=LLMConfig.MODEL,
                    messages=messages,
                    max_tokens=LLMConfig.RESPONSE_MAX_TOKENS,
                    temperature=LLMConfig.RESPONSE_TEMPERATURE,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield json.dumps({"type": "chunk", "content": content}) + "\n"
            else:
                yield json.dumps({
                    "type": "chunk",
                    "content": "No matching alloys found in the knowledge graph.",
                }) + "\n"
            return

        # ── 4. Stream LLM response ──────────────────────────────────
        client = LLMConfig.get_client()
        if not client:
            return

        system_prompt = Prompts.CHAT_RESPONSE
        if intent == "ANALYTICS":
            system_prompt = Prompts.ANALYTICS_RESPONSE
        elif intent == "TARGET":
            system_prompt = Prompts.TARGET_RESPONSE

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-HistoryConfig.MAX_CONTEXT_MESSAGES:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        messages.append({"role": "user", "content": f"Context:\n{final_context}\n\nUser Query: {prompt}"})

        stream = client.chat.completions.create(
            model=LLMConfig.MODEL,
            messages=messages,
            max_tokens=LLMConfig.RESPONSE_MAX_TOKENS,
            temperature=LLMConfig.RESPONSE_TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield json.dumps({"type": "chunk", "content": content}) + "\n"
