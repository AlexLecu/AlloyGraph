import os
from datetime import date
from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty, sync_reasoner_hermit
from rdflib import Namespace, URIRef, Literal, RDFS, XSD

BASE_IRI = "https://w3id.org/alloygraph/ont"
SAVE = os.getenv("SAVE_ONTOLOGY", "1") == "1"
OUT_PATH = os.getenv("ONTO_OUT", "../Data/Ontology/alloygraph.owl")

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

        # Ontology Metadata
        onto.metadata.label.append("AlloyGraph Ontology")
        onto.metadata.comment.append(
            "Ontology for Ni-based superalloys with composition, processing, "
            "mechanical properties, and computed metallurgical features."
        )
        onto.metadata.versionInfo.append("1.1.0")

    return onto


def run_reasoner(onto):
    """Run HermiT OWL reasoner to infer implicit facts."""
    print("  Running HermiT reasoner...")
    try:
        with onto:
            sync_reasoner_hermit(infer_property_values=True)
    except Exception as e:
        print(f"  Warning: Reasoner failed: {e}")


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

    return g


def main():
    print("=" * 60)
    print("AlloyGraph Ontology Builder")
    print("=" * 60)
    print(f"  Base IRI: {BASE_IRI}")
    print(f"  Output:   {OUT_PATH}")
    print()

    print("Building ontology schema...")
    onto = build_ontology()

    run_reasoner(onto)

    if SAVE:
        out_dir = os.path.dirname(OUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        temp_path = OUT_PATH + ".tmp"
        onto.save(file=temp_path, format="rdfxml")
        print("  Base ontology saved")

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
