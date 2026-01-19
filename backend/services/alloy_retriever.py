import os
import json
from typing import Optional
from dataclasses import dataclass
import weaviate
from weaviate.classes.query import QueryReference


@dataclass
class PropertyMeasurement:
    """A single property measurement with value, unit, and temperature."""
    property_type: str
    value: float
    unit: str
    temperature_c: Optional[float] = None
    stress_mpa: Optional[float] = None
    life_hours: Optional[float] = None
    
    def format(self) -> str:
        """Format for display."""
        temp_str = f" @ {self.temperature_c}°C" if self.temperature_c is not None else ""
        
        if self.stress_mpa and self.life_hours:
            return f"Stress: {self.stress_mpa:.0f} MPa, Life: {self.life_hours:.0f}h{temp_str}"
        
        return f"{self.value:.0f} {self.unit}{temp_str}"


@dataclass
class AlloyData:
    """Complete alloy data from Weaviate."""
    name: str
    processing_method: str    # Composition data
    composition: Optional[dict[str, float]] = None
    atomic_composition: Optional[dict[str, float]] = None
    gamma_composition: Optional[dict[str, float]] = None
    gamma_prime_composition: Optional[dict[str, float]] = None
    # Physical properties
    density_gcm3: Optional[float] = None
    gamma_prime_vol_pct: Optional[float] = None
    # Phase stability parameters
    md_avg: Optional[float] = None
    md_gamma: Optional[float] = None
    vec_avg: Optional[float] = None
    tcp_risk: Optional[str] = None
    lattice_mismatch_pct: Optional[float] = None
    # Strengthening parameters
    sss_wt_pct: Optional[float] = None
    sss_coefficient: Optional[float] = None
    precipitation_hardening_coeff: Optional[float] = None
    creep_resistance_param: Optional[float] = None
    # Composition metrics
    refractory_wt_pct: Optional[float] = None
    gp_formers_wt_pct: Optional[float] = None
    gp_formers_at_pct: Optional[float] = None
    oxidation_resistance: Optional[float] = None
    # Element ratios
    al_ti_ratio: Optional[float] = None
    al_ti_at_ratio: Optional[float] = None
    cr_co_ratio: Optional[float] = None
    cr_ni_ratio: Optional[float] = None
    mo_w_ratio: Optional[float] = None
    # Interaction terms
    al_ti_interaction: Optional[float] = None
    cr_al_interaction: Optional[float] = None
    # Mechanical properties
    properties: Optional[list[PropertyMeasurement]] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = []
        if self.composition is None:
            self.composition = {}
        if self.atomic_composition is None:
            self.atomic_composition = {}
        if self.gamma_composition is None:
            self.gamma_composition = {}
        if self.gamma_prime_composition is None:
            self.gamma_prime_composition = {}


class AlloyRetriever:
    """Clean interface to Weaviate Variant collection."""
    
    def __init__(self):
        """Initialize Weaviate client."""
        self.host = os.getenv('WEAVIATE_HOST', 'localhost')
        self.port = int(os.getenv('WEAVIATE_PORT', 8081))
        self.grpc_port = int(os.getenv('WEAVIATE_GRPC_PORT', 50052))
        self._client = None
    
    def _get_client(self) -> weaviate.WeaviateClient:
        """Get or create Weaviate client."""
        if self._client is None:
            self._client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port
            )
        return self._client
    
    def close(self):
        """Close Weaviate connection."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def search_alloys(self, query: str, limit: int = 5) -> list[AlloyData]:
        """
        Search for alloys using hybrid search.

        Args:
            query: Natural language search query
            limit: Maximum number of results

        Returns:
            List of AlloyData objects with properties and temperatures
        """
        client = self._get_client()

        if not client.is_live():
            raise ConnectionError("Weaviate instance is not reachable")

        collection = client.collections.get("Variant")

        response = collection.query.hybrid(
            query=query,
            limit=limit,
            return_properties=[
                # Basic info
                "name", "processingMethod",
                # Physical properties
                "densityCalculated", "gammaPrimeEstimate",
                # Phase stability
                "mdAverage", "mdGamma", "vecAvg", "tcpRisk", "latticeMismatchPct",
                # Strengthening
                "sssTotalWtPct", "sssCoefficient", "precipitationHardeningCoeff", "creepResistanceParam",
                # Composition metrics
                "refractoryTotalWtPct", "gpFormersWtPct", "gpFormersAtPct", "oxidationResistance",
                # Element ratios
                "alTiRatio", "alTiAtRatio", "crCoRatio", "crNiRatio", "moWRatio",
                # Interaction terms
                "alTiInteraction", "crAlInteraction",
                # JSON composition fields
                "atomicCompositionJson", "gammaCompositionJson", "gammaPrimeCompositionJson",
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
                ),
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
        
        # Parse results
        alloys = []
        for obj in response.objects:
            props = obj.properties
            
            alloy = AlloyData(
                name=str(props.get('name', 'Unknown')),
                processing_method=str(props.get('processingMethod', 'Unknown')),
                # Physical properties
                density_gcm3=props.get('densityCalculated'),
                gamma_prime_vol_pct=props.get('gammaPrimeEstimate'),
                # Phase stability
                md_avg=props.get('mdAverage'),
                md_gamma=props.get('mdGamma'),
                vec_avg=props.get('vecAvg'),
                tcp_risk=props.get('tcpRisk'),
                lattice_mismatch_pct=props.get('latticeMismatchPct'),
                # Strengthening
                sss_wt_pct=props.get('sssTotalWtPct'),
                sss_coefficient=props.get('sssCoefficient'),
                precipitation_hardening_coeff=props.get('precipitationHardeningCoeff'),
                creep_resistance_param=props.get('creepResistanceParam'),
                # Composition metrics
                refractory_wt_pct=props.get('refractoryTotalWtPct'),
                gp_formers_wt_pct=props.get('gpFormersWtPct'),
                gp_formers_at_pct=props.get('gpFormersAtPct'),
                oxidation_resistance=props.get('oxidationResistance'),
                # Element ratios
                al_ti_ratio=props.get('alTiRatio'),
                al_ti_at_ratio=props.get('alTiAtRatio'),
                cr_co_ratio=props.get('crCoRatio'),
                cr_ni_ratio=props.get('crNiRatio'),
                mo_w_ratio=props.get('moWRatio'),
                # Interaction terms
                al_ti_interaction=props.get('alTiInteraction'),
                cr_al_interaction=props.get('crAlInteraction'),
                # Initialize empty collections
                composition={},
                atomic_composition={},
                gamma_composition={},
                gamma_prime_composition={},
                properties=[]
            )

            # Parse atomic composition JSON
            try:
                ac_json = props.get('atomicCompositionJson')
                if ac_json:
                    data = json.loads(ac_json)
                    if "atomic_percent" in data:
                        alloy.atomic_composition = data["atomic_percent"]
                    else:
                        alloy.atomic_composition = data
            except Exception:
                pass

            # Parse gamma (matrix) composition JSON
            try:
                gc_json = props.get('gammaCompositionJson')
                if gc_json:
                    data = json.loads(gc_json)
                    alloy.gamma_composition = data
            except Exception:
                pass

            # Parse gamma prime (precipitate) composition JSON
            try:
                gp_json = props.get('gammaPrimeCompositionJson')
                if gp_json:
                    data = json.loads(gp_json)
                    alloy.gamma_prime_composition = data
            except Exception:
                pass
            
            # Extract composition
            if obj.references and "hasComposition" in obj.references:
                comp_obj = obj.references["hasComposition"].objects[0]
                if comp_obj.references and "hasComponent" in comp_obj.references:
                    for component in comp_obj.references["hasComponent"].objects:
                        if component.references and "hasElement" in component.references:
                            element_symbol = component.references["hasElement"].objects[0].properties.get("symbol")
                            if component.references and "hasMassFraction" in component.references:
                                mass_frac = component.references["hasMassFraction"].objects[0]
                                value = mass_frac.properties.get("numericValue") or mass_frac.properties.get("nominal")
                                if element_symbol and value is not None:
                                    alloy.composition[element_symbol] = float(value)
            
            # Extract properties with temperatures
            if obj.references and "hasPropertySet" in obj.references:
                for pset in obj.references["hasPropertySet"].objects:
                    prop_type = "Unknown"
                    if pset.references and "measuresProperty" in pset.references:
                        prop_type = pset.references["measuresProperty"].objects[0].properties.get("propertyType", "Unknown")
                    
                    if pset.references and "hasMeasurement" in pset.references:
                        for meas in pset.references["hasMeasurement"].objects:
                            val = 0.0
                            unit = ""
                            temp_c = None
                            stress = meas.properties.get("stress")
                            life_hours = meas.properties.get("lifeHours")
                            
                            # Get value and unit
                            if life_hours is not None:
                                val = life_hours
                                unit = "h"
                            elif meas.references and "hasQuantity" in meas.references:
                                quant = meas.references["hasQuantity"].objects[0]
                                val = quant.properties.get("numericValue", 0.0)
                                unit = quant.properties.get("unitSymbol", "")
                            
                            # Get temperature
                            if meas.references and "hasTestCondition" in meas.references:
                                tc = meas.references["hasTestCondition"].objects[0]
                                if tc.references and "hasTemperature" in tc.references:
                                    temp_c = tc.references["hasTemperature"].objects[0].properties.get("numericValue")
                            
                            alloy.properties.append(PropertyMeasurement(
                                property_type=prop_type,
                                value=val,
                                unit=unit,
                                temperature_c=temp_c,
                                stress_mpa=stress,
                                life_hours=life_hours
                            ))
            
            alloys.append(alloy)
        
        return alloys
    
    def get_alloys_with_property(self, property_name_part: str, limit: int = 50) -> list[AlloyData]:
        """
        Fetch a larger set of alloys that likely contain the requested property data.
        This is optimized for analytical queries where we do sorting in Python.
        """
        return self.search_alloys(property_name_part, limit=limit)

    @staticmethod
    def format_for_llm(alloys: list[AlloyData]) -> str:
        """
        Format alloy data for LLM consumption.

        Args:
            alloys: List of AlloyData objects

        Returns:
            Formatted string with all alloy information
        """
        if not alloys:
            return "No matching alloys found in the knowledge graph."
        
        results = []
        for alloy in alloys:
            lines = [
                f"\n{'='*70}",
                f"Alloy: {alloy.name}",
                f"Processing: {alloy.processing_method}",
                f"{'='*70}"
            ]
            
            if alloy.composition:
                lines.append("\nComposition (wt%):")
                sorted_comp = sorted(alloy.composition.items(), key=lambda x: x[1], reverse=True)
                for element, value in sorted_comp:
                    lines.append(f"  {element}: {value:.2f}%")
            
            if alloy.atomic_composition:
                lines.append("\nAtomic Composition (at%):")
                sorted_at = sorted(alloy.atomic_composition.items(), key=lambda x: x[1], reverse=True)
                for element, value in sorted_at:
                    lines.append(f"  {element}: {value:.2f}%")

            if alloy.gamma_composition:
                lines.append("\nGamma (Matrix) Phase Composition (at%):")
                sorted_gamma = sorted(alloy.gamma_composition.items(), key=lambda x: x[1], reverse=True)
                for element, value in sorted_gamma:
                    lines.append(f"  {element}: {value:.2f}%")

            if alloy.gamma_prime_composition:
                lines.append("\nGamma Prime (Precipitate) Phase Composition (at%):")
                sorted_gp = sorted(alloy.gamma_prime_composition.items(), key=lambda x: x[1], reverse=True)
                for element, value in sorted_gp:
                    lines.append(f"  {element}: {value:.2f}%")

            # Physical Properties
            phys_props = []
            if alloy.density_gcm3:
                phys_props.append(f"Density: {alloy.density_gcm3:.2f} g/cm³")
            if alloy.gamma_prime_vol_pct:
                phys_props.append(f"γ' Volume Fraction: {alloy.gamma_prime_vol_pct:.1f}%")
            if phys_props:
                lines.append("\nPhysical Properties:")
                for p in phys_props:
                    lines.append(f"  {p}")

            # Phase Stability Parameters
            stability_props = []
            if alloy.md_avg is not None:
                stability_props.append(f"Md (avg): {alloy.md_avg:.3f}")
            if alloy.md_gamma is not None:
                stability_props.append(f"Md (γ matrix): {alloy.md_gamma:.3f}")
            if alloy.vec_avg is not None:
                stability_props.append(f"VEC (avg): {alloy.vec_avg:.2f}")
            if alloy.tcp_risk:
                stability_props.append(f"TCP Risk: {alloy.tcp_risk}")
            if alloy.lattice_mismatch_pct is not None:
                stability_props.append(f"Lattice Mismatch: {alloy.lattice_mismatch_pct:.3f}%")
            if stability_props:
                lines.append("\nPhase Stability:")
                for p in stability_props:
                    lines.append(f"  {p}")

            # Strengthening Parameters
            strength_props = []
            if alloy.sss_wt_pct is not None:
                strength_props.append(f"SSS Elements: {alloy.sss_wt_pct:.1f} wt%")
            if alloy.sss_coefficient is not None:
                strength_props.append(f"SSS Coefficient: {alloy.sss_coefficient:.4f}")
            if alloy.precipitation_hardening_coeff is not None:
                strength_props.append(f"Precipitation Hardening: {alloy.precipitation_hardening_coeff:.4f}")
            if alloy.creep_resistance_param is not None:
                strength_props.append(f"Creep Resistance Parameter: {alloy.creep_resistance_param:.2f}")
            if strength_props:
                lines.append("\nStrengthening Mechanisms:")
                for p in strength_props:
                    lines.append(f"  {p}")

            # Composition Metrics
            comp_metrics = []
            if alloy.refractory_wt_pct is not None:
                comp_metrics.append(f"Refractory Elements: {alloy.refractory_wt_pct:.1f} wt%")
            if alloy.gp_formers_wt_pct is not None:
                comp_metrics.append(f"γ' Formers: {alloy.gp_formers_wt_pct:.1f} wt%")
            if alloy.gp_formers_at_pct is not None:
                comp_metrics.append(f"γ' Formers: {alloy.gp_formers_at_pct:.1f} at%")
            if alloy.oxidation_resistance is not None:
                comp_metrics.append(f"Oxidation Resistance Index: {alloy.oxidation_resistance:.2f}")
            if comp_metrics:
                lines.append("\nComposition Metrics:")
                for p in comp_metrics:
                    lines.append(f"  {p}")

            # Element Ratios
            ratios = []
            if alloy.al_ti_ratio is not None:
                ratios.append(f"Al/Ti (wt): {alloy.al_ti_ratio:.2f}")
            if alloy.al_ti_at_ratio is not None:
                ratios.append(f"Al/Ti (at): {alloy.al_ti_at_ratio:.2f}")
            if alloy.cr_co_ratio is not None:
                ratios.append(f"Cr/Co: {alloy.cr_co_ratio:.2f}")
            if alloy.cr_ni_ratio is not None:
                ratios.append(f"Cr/Ni: {alloy.cr_ni_ratio:.3f}")
            if alloy.mo_w_ratio is not None:
                ratios.append(f"Mo/W: {alloy.mo_w_ratio:.2f}")
            if ratios:
                lines.append("\nElement Ratios:")
                for r in ratios:
                    lines.append(f"  {r}")
            
            if alloy.properties:
                lines.append("\nMechanical Properties:")
                
                # Group by property type
                prop_groups: dict[str, list[PropertyMeasurement]] = {}
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
