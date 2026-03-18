import os
from datetime import date
from owlready2 import (
    get_ontology, Thing, DataProperty, ObjectProperty,
    ConstrainedDatatype, sync_reasoner_hermit, AllDisjoint,
)
from rdflib import Namespace, URIRef, Literal, RDFS, XSD

BASE_IRI = "https://w3id.org/alloygraph/ont"
SAVE = os.getenv("SAVE_ONTOLOGY", "1") == "1"
OUT_PATH = os.getenv("ONTO_OUT", os.path.join(os.path.dirname(__file__), "..", "..", "ontology", "alloygraph.owl"))

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
EMMO = Namespace("http://emmo.info/emmo#")
CHEBI = Namespace("http://purl.obolibrary.org/obo/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
NS = Namespace(f"{BASE_IRI}#")

EMMO_MAPPINGS = {
    "Material": "EMMO_4207e895_8b83_4318_996a_72cfb32acd94",
    "MechanicalProperty": "EMMO_b7bcff25_ffc3_474e_9ab5_01b1664bd4ba",
    "ChemicalElement": "EMMO_4f40def1_3cd7_4067_9596_541e9a5134cf",
}


def build_ontology():
    """Build the ontology schema using owlready2."""
    onto = get_ontology(BASE_IRI)

    with onto:
        # Core Classes
        class Material(Thing):
            """Base class for all materials."""
            pass

        class NickelBasedSuperalloy(Material):
            """Nickel-based superalloy material."""
            pass

        class Variant(Thing):
            """Specific variant of an alloy (processing + form combination)."""
            pass

        class Composition(Thing):
            """Chemical composition of an alloy."""
            pass

        class CompositionEntry(Thing):
            """Single element entry in a composition (n-ary pattern)."""
            pass

        class Element(Thing):
            """Chemical element."""
            pass
            
        class ProcessingMethod(Thing):
            """Manufacturing/processing method."""
            pass

        class Form(Thing):
            """Physical form of the material."""
            pass

        # Phase Classes
        class Phase(Thing):
            """Thermodynamic phase in the alloy microstructure."""
            pass

        class GammaPhase(Phase):
            """FCC γ (gamma) matrix phase."""
            pass

        class GammaPrimePhase(Phase):
            """L12 γ' (gamma prime) precipitate phase."""
            pass

        class TCPPhase(Phase):
            """Topologically Close-Packed (TCP) phase."""
            pass

        class HeatTreatment(Thing):
            """Heat treatment applied to the alloy."""
            pass

        # Property Classes
        class PropertySet(Thing):
            """Collection of measurements for a property type."""
            pass

        class MechanicalProperty(Thing):
            """Base class for mechanical properties."""
            pass

        class YieldStrength(MechanicalProperty):
            """Yield strength (0.2% offset) in MPa."""
            pass

        class UTS(MechanicalProperty):
            """Ultimate Tensile Strength in MPa."""
            pass

        class Elongation(MechanicalProperty):
            """Elongation at break in percent."""
            pass

        class Elasticity(MechanicalProperty):
            """Elastic Modulus (Young's modulus) in GPa."""
            pass

        class Hardness(MechanicalProperty):
            """Hardness measurement."""
            pass

        class CreepRupture(MechanicalProperty):
            """Creep rupture life at given stress and temperature."""
            pass

        class ReductionOfArea(MechanicalProperty):
            """Reduction of area at fracture in percent."""
            pass

        class Density(MechanicalProperty):
            """Material density in g/cm³."""
            pass

        class Measurement(Thing):
            """Single measurement/data point."""
            pass

        class Quantity(Thing):
            """Numeric quantity with optional unit and range."""
            pass

        class Unit(Thing):
            """Unit of measurement."""
            pass

        # Object Properties
        class hasVariant(ObjectProperty):
            domain = [NickelBasedSuperalloy]
            range = [Variant]

        class isVariantOf(ObjectProperty):
            domain = [Variant]
            range = [NickelBasedSuperalloy]
            inverse_property = hasVariant

        class hasComposition(ObjectProperty):
            domain = [Variant]
            range = [Composition]

        class hasComponent(ObjectProperty):
            domain = [Composition]
            range = [CompositionEntry]

        class isComponentOf(ObjectProperty):
            domain = [CompositionEntry]
            range = [Composition]
            inverse_property = hasComponent

        class element(ObjectProperty):
            domain = [CompositionEntry]
            range = [Element]

        class hasMassFraction(ObjectProperty):
            domain = [CompositionEntry]
            range = [Quantity]

        class hasProcessingMethod(ObjectProperty):
            domain = [Variant]
            range = [ProcessingMethod]

        class hasForm(ObjectProperty):
            domain = [Variant]
            range = [Form]

        class hasHeatTreatment(ObjectProperty):
            domain = [Variant]
            range = [HeatTreatment]

        class hasPhase(ObjectProperty):
            domain = [Variant]
            range = [Phase]

        class hasPropertySet(ObjectProperty):
            domain = [Variant]
            range = [PropertySet]

        class measuresProperty(ObjectProperty):
            domain = [PropertySet]
            range = [MechanicalProperty]

        class hasMeasurement(ObjectProperty):
            domain = [PropertySet]
            range = [Measurement]

        class hasQuantity(ObjectProperty):
            domain = [Measurement]
            range = [Quantity]

        class hasTestTemperature(ObjectProperty):
            domain = [Measurement]
            range = [Quantity]

        class hasUnit(ObjectProperty):
            domain = [Quantity]
            range = [Unit]

        # Data Properties - Alloy Metadata
        class tradeDesignation(DataProperty):
            domain = [NickelBasedSuperalloy]
            range = [str]

        class unsNumber(DataProperty):
            domain = [NickelBasedSuperalloy]
            range = [str]

        class family(DataProperty):
            domain = [NickelBasedSuperalloy]
            range = [str]

        # Data Properties - Variant Metadata
        class processingMethod(DataProperty):
            domain = [Variant]
            range = [str]

        class form(DataProperty):
            domain = [Form]
            range = [str]

        class density(DataProperty):
            domain = [Variant]
            range = [float]

        class gammaPrimeVolPct(DataProperty):
            domain = [Variant]
            range = [float]

        class typicalHeatTreatment(DataProperty):
            domain = [Variant]
            range = [str]

        # Data Properties - Computed Metallurgical Features
        class hasMdAverage(DataProperty):
            domain = [Variant]
            range = [float]

        class hasMdGamma(DataProperty):
            domain = [Variant]
            range = [float]

        class hasVECAvg(DataProperty):
            domain = [Variant]
            range = [float]

        class hasGammaPrimeEstimate(DataProperty):
            domain = [Variant]
            range = [float]

        class hasDensityCalculated(DataProperty):
            domain = [Variant]
            range = [float]

        class hasTcpRisk(DataProperty):
            domain = [Variant]
            range = [str]

        class hasLatticeMismatchPct(DataProperty):
            domain = [Variant]
            range = [float]

        class hasSSSTotalWtPct(DataProperty):
            domain = [Variant]
            range = [float]

        class hasSSSCoefficient(DataProperty):
            domain = [Variant]
            range = [float]

        class hasPrecipitationHardeningCoeff(DataProperty):
            domain = [Variant]
            range = [float]

        class hasCreepResistanceParam(DataProperty):
            domain = [Variant]
            range = [float]

        class hasRefractoryTotalWtPct(DataProperty):
            domain = [Variant]
            range = [float]

        class hasGPFormersWtPct(DataProperty):
            domain = [Variant]
            range = [float]

        class hasOxidationResistance(DataProperty):
            domain = [Variant]
            range = [float]

        class hasAlTiRatio(DataProperty):
            domain = [Variant]
            range = [float]

        class hasAlTiAtRatio(DataProperty):
            domain = [Variant]
            range = [float]

        class hasCrCoRatio(DataProperty):
            domain = [Variant]
            range = [float]

        class hasCrNiRatio(DataProperty):
            domain = [Variant]
            range = [float]

        class hasMoWRatio(DataProperty):
            domain = [Variant]
            range = [float]

        class hasGPFormersAtPct(DataProperty):
            domain = [Variant]
            range = [float]

        class hasAlTiInteraction(DataProperty):
            domain = [Variant]
            range = [float]

        class hasCrAlInteraction(DataProperty):
            domain = [Variant]
            range = [float]

        class hasAtomicCompositionJson(DataProperty):
            domain = [Variant]
            range = [str]

        class hasGammaCompositionJson(DataProperty):
            domain = [Variant]
            range = [str]

        class hasGammaPrimeCompositionJson(DataProperty):
            domain = [Variant]
            range = [str]

        # Data Properties - Quantity
        class numericValue(DataProperty):
            domain = [Quantity]
            range = [float]

        class minInclusive(DataProperty):
            domain = [Quantity]
            range = [float]

        class maxInclusive(DataProperty):
            domain = [Quantity]
            range = [float]

        class unitSymbol(DataProperty):
            domain = [Quantity]
            range = [str]

        class isApproximate(DataProperty):
            domain = [Quantity]
            range = [bool]

        class qualifier(DataProperty):
            domain = [Quantity]
            range = [str]

        class rawString(DataProperty):
            domain = [Quantity]
            range = [str]

        # Data Properties - Composition
        class isBalanceRemainder(DataProperty):
            domain = [CompositionEntry]
            range = [bool]

        class otherConstituents(DataProperty):
            domain = [Composition]
            range = [str]

        # Data Properties - Measurement
        class temperatureCategory(DataProperty):
            domain = [Measurement]
            range = [str]

        class heatTreatmentCondition(DataProperty):
            domain = [Measurement]
            range = [str]

        class stress(DataProperty):
            domain = [Measurement]
            range = [float]

        class lifeHours(DataProperty):
            domain = [Measurement]
            range = [float]

        # Cardinality Constraints
        CompositionEntry.is_a.append(element.exactly(1, Element))
        CompositionEntry.is_a.append(hasMassFraction.max(1, Quantity))
        PropertySet.is_a.append(measuresProperty.exactly(1, MechanicalProperty))
        Measurement.is_a.append(hasQuantity.exactly(1, Quantity))

        # =================================================================
        # Classification Axioms — alloy class (SSS / γ' / SC-DS)
        # Thresholds match alloy_parameters.py
        # =================================================================
        class SolidSolutionAlloy(Variant):
            """SSS alloy: GP formers (Al+Ti+Ta+0.35×Nb) < 2 wt%."""
            equivalent_to = [
                Variant
                & hasGPFormersWtPct.some(
                    ConstrainedDatatype(float, max_exclusive=2.0)
                )
            ]

        class GammaPrimeAlloy(Variant):
            """γ′ precipitation-hardened alloy: GP formers ≥ 2 wt%."""
            equivalent_to = [
                Variant
                & hasGPFormersWtPct.some(
                    ConstrainedDatatype(float, min_inclusive=2.0)
                )
            ]

        AllDisjoint([SolidSolutionAlloy, GammaPrimeAlloy])

        # =================================================================
        # Classification Axioms — TCP phase stability risk
        # Based on Morinaga Md parameter (1984 thresholds)
        # =================================================================
        class TCPRiskLow(Variant):
            """Low TCP risk: Md_avg < 0.940."""
            equivalent_to = [
                Variant
                & hasMdAverage.some(
                    ConstrainedDatatype(float, max_exclusive=0.940)
                )
            ]

        class TCPRiskModerate(Variant):
            """Moderate TCP risk: 0.940 ≤ Md_avg < 0.960."""
            equivalent_to = [
                Variant
                & hasMdAverage.some(
                    ConstrainedDatatype(float, min_inclusive=0.940, max_exclusive=0.960)
                )
            ]

        class TCPRiskElevated(Variant):
            """Elevated TCP risk: 0.960 ≤ Md_avg < 0.985."""
            equivalent_to = [
                Variant
                & hasMdAverage.some(
                    ConstrainedDatatype(float, min_inclusive=0.960, max_exclusive=0.985)
                )
            ]

        class TCPRiskCritical(Variant):
            """Critical TCP risk: Md_avg ≥ 0.985."""
            equivalent_to = [
                Variant
                & hasMdAverage.some(
                    ConstrainedDatatype(float, min_inclusive=0.985)
                )
            ]

        AllDisjoint([TCPRiskLow, TCPRiskModerate, TCPRiskElevated, TCPRiskCritical])

        # =================================================================
        # Data Range Restrictions — physical plausibility bounds
        # HermiT flags violations during consistency checking
        # =================================================================
        Variant.is_a.append(
            hasDensityCalculated.only(
                ConstrainedDatatype(float, min_inclusive=7.0, max_inclusive=10.0)
            )
        )
        Variant.is_a.append(
            hasGammaPrimeEstimate.only(
                ConstrainedDatatype(float, min_inclusive=0.0, max_inclusive=85.0)
            )
        )
        Variant.is_a.append(
            hasMdAverage.only(
                ConstrainedDatatype(float, min_inclusive=0.70, max_inclusive=1.05)
            )
        )
        Variant.is_a.append(
            hasLatticeMismatchPct.only(
                ConstrainedDatatype(float, min_inclusive=-2.0, max_inclusive=2.0)
            )
        )

        # Ontology Metadata
        onto.metadata.label.append("AlloyGraph Ontology")
        onto.metadata.comment.append(
            "Ontology for Ni-based superalloys with composition, processing, "
            "mechanical properties, computed metallurgical features, "
            "and OWL DL classification axioms for alloy type and TCP risk."
        )
        onto.metadata.versionInfo.append("1.2.0")

    return onto


def run_reasoner(onto):
    """Run HermiT OWL reasoner to infer implicit facts."""
    print("  Running HermiT reasoner...")
    try:
        with onto:
            sync_reasoner_hermit(infer_property_values=True)
    except Exception as e:
        print(f"  Warning: Reasoner failed: {e}")


def populate_and_reason(onto, json_path: str):
    """
    Load alloy data into owlready2 instances, run HermiT, and report
    before/after classification statistics.
    """
    import json
    try:
        from backend.alloy_crew.models.feature_engineering import compute_alloy_features
    except ModuleNotFoundError:
        from alloy_crew.models.feature_engineering import compute_alloy_features

    # Load alloy records
    alloys = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("==>"):
                try:
                    alloys.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\n  Loading {len(alloys)} alloy variants into ontology...")

    # Get class references from the ontology
    Variant = onto["Variant"]
    hasGPFormersWtPct = onto["hasGPFormersWtPct"]
    hasMdAverage = onto["hasMdAverage"]
    hasDensityCalculated = onto["hasDensityCalculated"]
    hasGammaPrimeEstimate = onto["hasGammaPrimeEstimate"]
    hasLatticeMismatchPct = onto["hasLatticeMismatchPct"]

    seen_alloys = set()

    with onto:
        for alloy_data in alloys:
            name = alloy_data.get("alloy", "")
            if not name:
                continue

            processing = alloy_data.get("processing", "")
            variant_id = f"{name}_{processing}".replace(" ", "_").replace("-", "_")

            if variant_id in seen_alloys:
                continue
            seen_alloys.add(variant_id)

            # Compute features from composition
            try:
                features = compute_alloy_features(alloy_data)
            except Exception:
                continue

            # Create owlready2 individual
            v = Variant(variant_id)
            v.hasGPFormersWtPct = [features.get("GP_formers_wt_pct", 0.0)]
            v.hasMdAverage = [features.get("Md_avg", 0.0)]
            v.hasDensityCalculated = [features.get("density_calculated_gcm3", 0.0)]
            v.hasGammaPrimeEstimate = [features.get("gamma_prime_estimated_vol_pct", 0.0)]
            v.hasLatticeMismatchPct = [features.get("lattice_mismatch_pct", 0.0)]

    print(f"  Created {len(seen_alloys)} Variant individuals")

    # --- Before reasoning: count unclassified ---
    SolidSolutionAlloy = onto["SolidSolutionAlloy"]
    GammaPrimeAlloy = onto["GammaPrimeAlloy"]
    TCPRiskLow = onto["TCPRiskLow"]
    TCPRiskModerate = onto["TCPRiskModerate"]
    TCPRiskElevated = onto["TCPRiskElevated"]
    TCPRiskCritical = onto["TCPRiskCritical"]

    before = {
        "SolidSolutionAlloy": len(list(SolidSolutionAlloy.instances())),
        "GammaPrimeAlloy": len(list(GammaPrimeAlloy.instances())),
        "TCPRiskLow": len(list(TCPRiskLow.instances())),
        "TCPRiskModerate": len(list(TCPRiskModerate.instances())),
        "TCPRiskElevated": len(list(TCPRiskElevated.instances())),
        "TCPRiskCritical": len(list(TCPRiskCritical.instances())),
    }

    print(f"\n  Before HermiT reasoning:")
    for cls_name, count in before.items():
        print(f"    {cls_name}: {count}")

    # --- Run HermiT ---
    print("\n  Running HermiT reasoner on populated ontology...")
    try:
        with onto:
            sync_reasoner_hermit(infer_property_values=True)
        print("  HermiT completed successfully (ontology is consistent)")
    except Exception as e:
        print(f"  HermiT found inconsistencies: {e}")
        return before, before  # Return before stats twice if reasoning fails

    # --- After reasoning: count classified ---
    after = {
        "SolidSolutionAlloy": len(list(SolidSolutionAlloy.instances())),
        "GammaPrimeAlloy": len(list(GammaPrimeAlloy.instances())),
        "TCPRiskLow": len(list(TCPRiskLow.instances())),
        "TCPRiskModerate": len(list(TCPRiskModerate.instances())),
        "TCPRiskElevated": len(list(TCPRiskElevated.instances())),
        "TCPRiskCritical": len(list(TCPRiskCritical.instances())),
    }

    print(f"\n  After HermiT reasoning:")
    for cls_name, count in after.items():
        diff = after[cls_name] - before[cls_name]
        marker = f" (+{diff} inferred)" if diff > 0 else ""
        print(f"    {cls_name}: {count}{marker}")

    total_classified = after["SolidSolutionAlloy"] + after["GammaPrimeAlloy"]
    total_tcp = after["TCPRiskLow"] + after["TCPRiskModerate"] + after["TCPRiskElevated"] + after["TCPRiskCritical"]
    print(f"\n  Summary:")
    print(f"    Alloy classification: {total_classified}/{len(seen_alloys)} variants classified")
    print(f"      SSS: {after['SolidSolutionAlloy']}, γ': {after['GammaPrimeAlloy']}")
    print(f"    TCP risk assessment: {total_tcp}/{len(seen_alloys)} variants assessed")
    print(f"      Low: {after['TCPRiskLow']}, Moderate: {after['TCPRiskModerate']}, "
          f"Elevated: {after['TCPRiskElevated']}, Critical: {after['TCPRiskCritical']}")

    return before, after


def add_external_alignments(owl_file: str):
    """Add external ontology alignments using rdflib."""
    from rdflib import Graph, OWL

    g = Graph()
    g.parse(owl_file, format="xml")

    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("emmo", EMMO)
    g.bind("chebi", CHEBI)
    g.bind("qudt", QUDT)
    g.bind("owl", OWL)
    g.bind("ns", NS)

    onto_uri = URIRef(BASE_IRI)

    g.add((onto_uri, DCTERMS.creator, Literal("AlloyGraph Project")))
    g.add((onto_uri, DCTERMS.created, Literal(date.today().isoformat(), datatype=XSD.date)))
    g.add((onto_uri, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    g.add((onto_uri, RDFS.seeAlso, URIRef("https://github.com/AlexLecu/AlloyGraph")))

    g.add((NS.Material, SKOS.closeMatch, EMMO[EMMO_MAPPINGS["Material"]]))
    g.add((NS.Element, SKOS.closeMatch, EMMO[EMMO_MAPPINGS["ChemicalElement"]]))
    g.add((NS.MechanicalProperty, SKOS.closeMatch, EMMO[EMMO_MAPPINGS["MechanicalProperty"]]))
    g.add((NS.Element, SKOS.closeMatch, CHEBI.CHEBI_33250))

    qudt_quantity_value = URIRef("http://qudt.org/schema/qudt/QuantityValue")
    qudt_unit = URIRef("http://qudt.org/schema/qudt/Unit")
    g.add((NS.Quantity, SKOS.closeMatch, qudt_quantity_value))
    g.add((NS.Unit, SKOS.closeMatch, qudt_unit))

    # =========================================================
    # rdfs:label and rdfs:comment for all classes and properties
    # =========================================================
    CLASS_LABELS = {
        "Material": ("Material", "Base class for all materials."),
        "NickelBasedSuperalloy": ("Nickel-Based Superalloy", "A nickel-based superalloy material."),
        "Variant": ("Variant", "A specific variant of an alloy defined by its processing route and product form."),
        "Composition": ("Composition", "The chemical composition of an alloy expressed as a set of element entries."),
        "CompositionEntry": ("Composition Entry", "A single element entry in a composition, mapping an element to its weight-percent quantity via an n-ary relation pattern."),
        "Element": ("Chemical Element", "A chemical element present in an alloy composition."),
        "ProcessingMethod": ("Processing Method", "The manufacturing or processing route applied to an alloy (e.g., wrought, cast)."),
        "Form": ("Product Form", "The physical product form of an alloy variant (e.g., bar, sheet, forged)."),
        "Phase": ("Phase", "A thermodynamic phase in the alloy microstructure."),
        "GammaPhase": ("Gamma Phase", "The face-centred cubic (FCC) gamma matrix phase."),
        "GammaPrimePhase": ("Gamma Prime Phase", "The ordered L1_2 gamma prime precipitate phase (Ni3Al-type)."),
        "TCPPhase": ("TCP Phase", "A topologically close-packed phase detrimental to mechanical properties (e.g., sigma, mu, Laves)."),
        "HeatTreatment": ("Heat Treatment", "A heat treatment applied to an alloy variant."),
        "PropertySet": ("Property Set", "A collection of measurements for a specific mechanical property type."),
        "MechanicalProperty": ("Mechanical Property", "Base class for mechanical properties of materials."),
        "YieldStrength": ("Yield Strength", "The 0.2% offset yield strength, measured in MPa."),
        "UTS": ("Ultimate Tensile Strength", "The ultimate tensile strength, measured in MPa."),
        "Elongation": ("Elongation", "The elongation at break, measured in percent."),
        "Elasticity": ("Elastic Modulus", "The elastic modulus (Young's modulus), measured in GPa."),
        "Hardness": ("Hardness", "A hardness measurement (e.g., Vickers, Rockwell)."),
        "CreepRupture": ("Creep Rupture", "The creep rupture life at a given stress and temperature."),
        "ReductionOfArea": ("Reduction of Area", "The reduction of area at fracture, measured in percent."),
        "Density": ("Density", "The material density, measured in g/cm3."),
        "Measurement": ("Measurement", "A single measurement data point recording a property value at a specific test condition."),
        "Quantity": ("Quantity", "A numeric quantity with an optional unit, range, and qualifier."),
        "Unit": ("Unit", "A unit of measurement."),
        "SolidSolutionAlloy": ("Solid Solution Strengthened Alloy", "An alloy classified as solid-solution strengthened: gamma-prime former content (Al+Ti+Ta+0.35*Nb) below 2.0 wt%."),
        "GammaPrimeAlloy": ("Gamma Prime Precipitation Hardened Alloy", "An alloy classified as gamma-prime precipitation hardened: gamma-prime former content (Al+Ti+Ta+0.35*Nb) at or above 2.0 wt%."),
        "TCPRiskLow": ("TCP Risk Low", "Alloy variant with low TCP phase formation risk: Md_avg below 0.940."),
        "TCPRiskModerate": ("TCP Risk Moderate", "Alloy variant with moderate TCP phase formation risk: Md_avg between 0.940 and 0.960."),
        "TCPRiskElevated": ("TCP Risk Elevated", "Alloy variant with elevated TCP phase formation risk: Md_avg between 0.960 and 0.985."),
        "TCPRiskCritical": ("TCP Risk Critical", "Alloy variant with critical TCP phase formation risk: Md_avg at or above 0.985."),
    }

    OBJECT_PROPERTY_LABELS = {
        "hasVariant": ("has variant", "Links a superalloy to one of its processing/form variants."),
        "isVariantOf": ("is variant of", "Inverse of hasVariant; links a variant back to its parent superalloy."),
        "hasComposition": ("has composition", "Links a variant to its chemical composition."),
        "hasComponent": ("has component", "Links a composition to one of its element entries."),
        "isComponentOf": ("is component of", "Inverse of hasComponent; links an element entry back to its composition."),
        "element": ("element", "Links a composition entry to the chemical element it represents."),
        "hasMassFraction": ("has mass fraction", "Links a composition entry to its weight-percent quantity."),
        "hasProcessingMethod": ("has processing method", "Links a variant to its manufacturing processing method."),
        "hasForm": ("has form", "Links a variant to its physical product form."),
        "hasHeatTreatment": ("has heat treatment", "Links a variant to its heat treatment."),
        "hasPhase": ("has phase", "Links a variant to a phase present in its microstructure."),
        "hasPropertySet": ("has property set", "Links a variant to a set of measurements for a property type."),
        "measuresProperty": ("measures property", "Links a property set to the type of mechanical property it measures."),
        "hasMeasurement": ("has measurement", "Links a property set to an individual measurement data point."),
        "hasQuantity": ("has quantity", "Links a measurement to its numeric quantity value."),
        "hasTestTemperature": ("has test temperature", "Links a measurement to the temperature at which it was performed."),
        "hasUnit": ("has unit", "Links a quantity to its unit of measurement."),
    }

    DATA_PROPERTY_LABELS = {
        "tradeDesignation": ("trade designation", "The commercial trade name of a superalloy."),
        "unsNumber": ("UNS number", "The Unified Numbering System identifier for the alloy."),
        "family": ("alloy family", "The alloy family classification."),
        "processingMethod": ("processing method (literal)", "The processing route as a string literal (denormalised convenience property)."),
        "form": ("product form (literal)", "The product form as a string literal."),
        "density": ("measured density", "The experimentally measured density in g/cm3."),
        "gammaPrimeVolPct": ("measured gamma prime vol%", "The experimentally measured gamma prime volume fraction in percent."),
        "typicalHeatTreatment": ("typical heat treatment", "Description of the typical heat treatment applied."),
        "hasMdAverage": ("Md average (bulk)", "The Morinaga d-electron parameter averaged over the bulk alloy composition."),
        "hasMdGamma": ("Md gamma (matrix)", "The Morinaga d-electron parameter computed from the estimated gamma matrix composition."),
        "hasVECAvg": ("VEC average", "The valence electron concentration averaged over the alloy composition."),
        "hasGammaPrimeEstimate": ("estimated gamma prime vol%", "The estimated gamma prime volume fraction from the solubility-based model, in percent."),
        "hasDensityCalculated": ("calculated density", "The alloy density calculated from the inverse rule of mixtures, in g/cm3."),
        "hasTcpRisk": ("TCP risk classification", "The TCP phase formation risk level (Low, Moderate, Elevated, or Critical)."),
        "hasLatticeMismatchPct": ("lattice mismatch %", "The lattice mismatch between gamma and gamma prime phases, in percent."),
        "hasSSSTotalWtPct": ("SSS total wt%", "Total weight percent of solid-solution strengthening elements (Mo+W+Nb+Ta+Re)."),
        "hasSSSCoefficient": ("SSS coefficient", "The solid solution strengthening coefficient computed via the Labusch-Nabarro model."),
        "hasPrecipitationHardeningCoeff": ("precipitation hardening coefficient", "The precipitation hardening coefficient proportional to f^0.5 x (Al+Ti)."),
        "hasCreepResistanceParam": ("creep resistance parameter", "A creep resistance parameter weighted by Re, Ru, and W content."),
        "hasRefractoryTotalWtPct": ("refractory total wt%", "Total weight percent of refractory elements (Mo+W+Ta+Re+Nb+Hf)."),
        "hasGPFormersWtPct": ("GP formers wt%", "Total weight percent of gamma-prime forming elements (Al+Ti+Ta+Nb)."),
        "hasOxidationResistance": ("oxidation resistance", "An oxidation resistance metric derived from Cr and Al content."),
        "hasAlTiRatio": ("Al/Ti ratio (wt%)", "The aluminium-to-titanium weight percent ratio."),
        "hasAlTiAtRatio": ("Al/Ti ratio (at%)", "The aluminium-to-titanium atomic percent ratio."),
        "hasCrCoRatio": ("Cr/Co ratio", "The chromium-to-cobalt weight percent ratio."),
        "hasCrNiRatio": ("Cr/Ni ratio", "The chromium-to-nickel weight percent ratio."),
        "hasMoWRatio": ("Mo/W ratio", "The molybdenum-to-tungsten weight percent ratio."),
        "hasGPFormersAtPct": ("GP formers at%", "Total atomic percent of gamma-prime forming elements (Al+Ti+Ta+Nb)."),
        "hasAlTiInteraction": ("Al x Ti interaction", "The Al-Ti interaction term (product of atomic percentages)."),
        "hasCrAlInteraction": ("Cr x Al interaction", "The Cr-Al interaction term (product of weight percentages)."),
        "hasAtomicCompositionJson": ("atomic composition (JSON)", "The full alloy composition in atomic percent, serialised as a JSON string."),
        "hasGammaCompositionJson": ("gamma composition (JSON)", "The estimated gamma matrix composition in atomic percent, serialised as a JSON string."),
        "hasGammaPrimeCompositionJson": ("gamma prime composition (JSON)", "The estimated gamma prime composition in atomic percent, serialised as a JSON string."),
        "numericValue": ("numeric value", "The numeric value of a quantity."),
        "minInclusive": ("minimum value (inclusive)", "The minimum bound of a quantity range."),
        "maxInclusive": ("maximum value (inclusive)", "The maximum bound of a quantity range."),
        "unitSymbol": ("unit symbol", "The symbol of the unit of measurement (e.g., MPa, GPa, %)."),
        "isApproximate": ("is approximate", "Whether the quantity value is approximate."),
        "qualifier": ("qualifier", "A qualifier for the quantity (e.g., minimum, typical)."),
        "rawString": ("raw string", "The original string representation of the quantity as parsed from the source data."),
        "isBalanceRemainder": ("is balance remainder", "Whether the composition entry represents the balance element (typically Ni)."),
        "otherConstituents": ("other constituents", "Description of other minor or trace constituents in the composition."),
        "temperatureCategory": ("temperature category", "A categorical label for the test temperature range (e.g., room temperature, elevated)."),
        "heatTreatmentCondition": ("heat treatment condition", "The heat treatment condition under which the measurement was taken."),
        "stress": ("stress", "The applied stress for creep rupture tests, in MPa."),
        "lifeHours": ("life hours", "The creep rupture life in hours."),
    }

    for local_name, (label, comment) in CLASS_LABELS.items():
        uri = NS[local_name]
        g.add((uri, RDFS.label, Literal(label, lang="en")))
        g.add((uri, RDFS.comment, Literal(comment, lang="en")))

    for local_name, (label, comment) in OBJECT_PROPERTY_LABELS.items():
        uri = NS[local_name]
        g.add((uri, RDFS.label, Literal(label, lang="en")))
        g.add((uri, RDFS.comment, Literal(comment, lang="en")))

    for local_name, (label, comment) in DATA_PROPERTY_LABELS.items():
        uri = NS[local_name]
        g.add((uri, RDFS.label, Literal(label, lang="en")))
        g.add((uri, RDFS.comment, Literal(comment, lang="en")))

    return g


def main():
    JSON_PATH = os.getenv("ALLOY_JSON", "../alloy_crew/models/training_data/train_77alloys.jsonl")

    print("=" * 60)
    print("AlloyGraph Ontology Builder")
    print("=" * 60)
    print(f"  Base IRI: {BASE_IRI}")
    print(f"  Output:   {OUT_PATH}")
    print()

    # Step 1: Build schema
    print("Step 1: Building ontology schema...")
    onto = build_ontology()

    # Step 2: Validate schema consistency
    print("\nStep 2: Validating schema consistency...")
    run_reasoner(onto)

    # Step 3: Populate with alloy data and run classification reasoning
    if os.path.exists(JSON_PATH):
        print(f"\nStep 3: Populating ontology with alloy data...")
        print(f"  Data source: {JSON_PATH}")
        populate_and_reason(onto, JSON_PATH)
    else:
        print(f"\nStep 3: Skipped (data file not found: {JSON_PATH})")

    # Step 4: Save OWL file (schema only, without instance data)
    if SAVE:
        out_dir = os.path.dirname(OUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Rebuild clean schema (without instance data) for the OWL file
        print("\nStep 4: Saving ontology schema...")
        clean_onto = build_ontology()
        temp_path = OUT_PATH + ".tmp"
        clean_onto.save(file=temp_path, format="rdfxml")

        print("  Adding external alignments...")
        g = add_external_alignments(temp_path)

        g.serialize(destination=OUT_PATH, format="xml")
        os.remove(temp_path)

        print(f"\n✅ Saved to: {OUT_PATH}")
        print(f"   Total triples: {len(g)}")

    print("\nExternal Alignments:")
    print("  Material      -> EMMO:Material")
    print("  Element       -> EMMO:ChemicalElement, ChEBI:33250")
    print("  MechProperty  -> EMMO:MechanicalProperty")
    print("  Quantity/Unit -> QUDT (closeMatch)")
    print("\nDone!")


if __name__ == "__main__":
    main()
