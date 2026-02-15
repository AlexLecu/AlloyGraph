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
        if self.stress_mpa:
            return f"{self.value:.0f} {self.unit} at {self.stress_mpa:.0f} MPa{temp_str}"

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
                comp_objects = obj.references["hasComposition"].objects
                if comp_objects:
                    comp_obj = comp_objects[0]
                    if comp_obj.references and "hasComponent" in comp_obj.references:
                        for component in comp_obj.references["hasComponent"].objects:
                            if not (component.references and "hasElement" in component.references):
                                continue
                            el_objects = component.references["hasElement"].objects
                            if not el_objects:
                                continue
                            element_symbol = el_objects[0].properties.get("symbol")
                            if not (component.references and "hasMassFraction" in component.references):
                                continue
                            mf_objects = component.references["hasMassFraction"].objects
                            if not mf_objects:
                                continue
                            mass_frac = mf_objects[0]
                            value = mass_frac.properties.get("numericValue")
                            if value is None:
                                value = mass_frac.properties.get("nominal")
                            if element_symbol and value is not None:
                                alloy.composition[element_symbol] = float(value)
            
            # Extract properties with temperatures
            if obj.references and "hasPropertySet" in obj.references:
                for pset in obj.references["hasPropertySet"].objects:
                    prop_type = "Unknown"
                    if pset.references and "measuresProperty" in pset.references:
                        mp_objects = pset.references["measuresProperty"].objects
                        if mp_objects:
                            prop_type = mp_objects[0].properties.get("propertyType", "Unknown")

                    if pset.references and "hasMeasurement" in pset.references:
                        for meas in pset.references["hasMeasurement"].objects:
                            val = None
                            unit = ""
                            temp_c = None
                            stress = meas.properties.get("stress")
                            life_hours = meas.properties.get("lifeHours")

                            # Get value and unit
                            if life_hours is not None:
                                val = life_hours
                                unit = "h"
                            elif meas.references and "hasQuantity" in meas.references:
                                q_objects = meas.references["hasQuantity"].objects
                                if q_objects:
                                    quant = q_objects[0]
                                    val = quant.properties.get("numericValue")
                                    unit = quant.properties.get("unitSymbol", "")

                            # Skip measurements where we couldn't extract a value
                            if val is None:
                                continue

                            # Get temperature
                            if meas.references and "hasTestCondition" in meas.references:
                                tc_objects = meas.references["hasTestCondition"].objects
                                if tc_objects:
                                    tc = tc_objects[0]
                                    if tc.references and "hasTemperature" in tc.references:
                                        t_objects = tc.references["hasTemperature"].objects
                                        if t_objects:
                                            temp_c = t_objects[0].properties.get("numericValue")

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

