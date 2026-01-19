import logging
import time
import os

import weaviate
from weaviate.classes.config import Property, DataType, Configure, ReferenceProperty
from weaviate.util import generate_uuid5

logger = logging.getLogger(__name__)

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8081"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50052"))

# URI Namespaces (matching enrich_graphdb_new.py)
ONTOLOGY_BASE = "https://w3id.org/alloygraph/ont#"
RESOURCE_BASE = "https://w3id.org/alloygraph/res/"
NS = "alloygraph"  # namespace for deterministic UUIDv5


def sid(ns: str, key: str) -> str:
    """Generate stable UUID5 from namespace and key"""
    return str(generate_uuid5(ns, key))


def connect():
    """Connect to local Weaviate instance"""
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )

    # Wait for Weaviate to be ready
    for _ in range(30):
        try:
            client.is_live()
            break
        except Exception:
            time.sleep(1)
    return client


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    client = connect()

    try:
        # Wipe existing schema if any
        try:
            client.collections.delete_all()
            logger.info("✓ Deleted existing schema.")
        except Exception as e:
            logger.warning(f"Schema delete warning: {e}")

        vec_config = Configure.Vectorizer.text2vec_transformers()

        # ========================================================================
        # STEP 1: Create collections WITHOUT references first
        # ========================================================================

        # Core Alloy class
        client.collections.create(
            name="NickelBasedSuperalloy",
            properties=[
                Property(name="tradeDesignation", data_type=DataType.TEXT),
                Property(name="unsNumber", data_type=DataType.TEXT),
                Property(name="family", data_type=DataType.TEXT),
                Property(name="density", data_type=DataType.NUMBER),
                Property(name="gammaPrimeVolPct", data_type=DataType.NUMBER),
                Property(name="typicalHeatTreatment", data_type=DataType.TEXT),
                Property(name="compositionSummary", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: NickelBasedSuperalloy")

        # Composition
        client.collections.create(
            name="Composition",
            properties=[
                Property(name="otherConstituents", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: Composition")

        client.collections.create(
            name="CompositionEntry",
            properties=[
                Property(name="isBalanceRemainder", data_type=DataType.BOOL),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # No text to vectorize
        )
        logger.info("✓ Created: CompositionEntry")

        # Element (fixed vocabulary)
        client.collections.create(
            name="Element",
            properties=[
                Property(name="symbol", data_type=DataType.TEXT),
                Property(name="label", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: Element")

        # Quantity (base class for all numeric values with units)
        client.collections.create(
            name="Quantity",
            properties=[
                Property(name="numericValue", data_type=DataType.NUMBER),
                Property(name="minInclusive", data_type=DataType.NUMBER),
                Property(name="maxInclusive", data_type=DataType.NUMBER),
                Property(name="nominal", data_type=DataType.NUMBER),
                Property(name="unitSymbol", data_type=DataType.TEXT),
                Property(name="unitQUDT", data_type=DataType.TEXT),  # QUDT URI
                Property(name="isApproximate", data_type=DataType.BOOL),
                Property(name="qualifier", data_type=DataType.TEXT),  # <=, >=, etc.
                Property(name="rawString", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: Quantity")

        # NEW: Variant (different forms/processing of same alloy)
        client.collections.create(
            name="Variant",
            properties=[
                Property(name="name", data_type=DataType.TEXT),  # "RENE 41_wrought_bar"
                Property(name="processingMethod", data_type=DataType.TEXT),
                Property(name="density", data_type=DataType.NUMBER),
                Property(name="gammaPrimeVolPct", data_type=DataType.NUMBER),
                Property(name="typicalHeatTreatment", data_type=DataType.TEXT),
                Property(name="compositionSummary", data_type=DataType.TEXT),
                Property(name="sourceUrl", data_type=DataType.TEXT),
                # Computed metallurgical features
                Property(name="mdAverage", data_type=DataType.NUMBER),
                Property(name="mdGamma", data_type=DataType.NUMBER),
                Property(name="vecAvg", data_type=DataType.NUMBER),
                Property(name="gammaPrimeEstimate", data_type=DataType.NUMBER),
                Property(name="densityCalculated", data_type=DataType.NUMBER),
                Property(name="tcpRisk", data_type=DataType.TEXT),
                Property(name="latticeMismatchPct", data_type=DataType.NUMBER),
                Property(name="sssTotalWtPct", data_type=DataType.NUMBER),
                Property(name="sssCoefficient", data_type=DataType.NUMBER),
                Property(name="precipitationHardeningCoeff", data_type=DataType.NUMBER),
                Property(name="creepResistanceParam", data_type=DataType.NUMBER),
                Property(name="refractoryTotalWtPct", data_type=DataType.NUMBER),
                Property(name="gpFormersWtPct", data_type=DataType.NUMBER),
                Property(name="oxidationResistance", data_type=DataType.NUMBER),
                Property(name="alTiRatio", data_type=DataType.NUMBER),
                Property(name="alTiAtRatio", data_type=DataType.NUMBER),
                Property(name="crCoRatio", data_type=DataType.NUMBER),
                Property(name="crNiRatio", data_type=DataType.NUMBER),
                Property(name="moWRatio", data_type=DataType.NUMBER),
                Property(name="gpFormersAtPct", data_type=DataType.NUMBER),
                Property(name="alTiInteraction", data_type=DataType.NUMBER),
                Property(name="crAlInteraction", data_type=DataType.NUMBER),
                # JSON composition fields
                Property(name="atomicCompositionJson", data_type=DataType.TEXT),
                Property(name="gammaCompositionJson", data_type=DataType.TEXT),
                Property(name="gammaPrimeCompositionJson", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: Variant")

        # NEW: ProcessingRoute (manufacturing process)
        client.collections.create(
            name="ProcessingRoute",
            properties=[
                Property(name="processingDescription", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: ProcessingRoute")

        # NEW: FormType (bar, plate, sheet, etc.)
        client.collections.create(
            name="FormType",
            properties=[
                Property(name="formTypeName", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: FormType")

        # NEW: PropertySet (groups measurements of same property type)
        client.collections.create(
            name="PropertySet",
            properties=[],  # references only
            vectorizer_config=Configure.Vectorizer.none(),  # No properties to vectorize
        )
        logger.info("✓ Created: PropertySet")

        # MechanicalProperty (fixed vocabulary - property types)
        client.collections.create(
            name="MechanicalProperty",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="propertyType", data_type=DataType.TEXT),  # TensileStrength, etc.
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: MechanicalProperty")

        # NEW: TestCondition (reusable test environment)
        client.collections.create(
            name="TestCondition",
            properties=[
                Property(name="temperatureCategory", data_type=DataType.TEXT),  # RT, Low, High
                Property(name="heatTreatmentCondition", data_type=DataType.TEXT),
                Property(name="testStandard", data_type=DataType.TEXT),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: TestCondition")

        # Measurement
        client.collections.create(
            name="Measurement",
            properties=[
                Property(name="stress", data_type=DataType.NUMBER),
                Property(name="lifeHours", data_type=DataType.NUMBER),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # No properties to vectorize
        )
        logger.info("✓ Created: Measurement")

        # NEW: MaterialIndex (flattened view for fast queries)
        client.collections.create(
            name="MaterialIndex",
            properties=[
                Property(name="indexAlloyName", data_type=DataType.TEXT),
                Property(name="indexVariantName", data_type=DataType.TEXT),
                Property(name="indexProcessing", data_type=DataType.TEXT),
                # Room temperature properties
                Property(name="rtTensileStrength", data_type=DataType.NUMBER),
                Property(name="rtYieldStrength", data_type=DataType.NUMBER),
                Property(name="rtElongation", data_type=DataType.NUMBER),
                Property(name="rtHardness", data_type=DataType.NUMBER),
                Property(name="rtElasticModulus", data_type=DataType.NUMBER),
                # Key composition
                Property(name="nickelContent", data_type=DataType.NUMBER),
                Property(name="chromiumContent", data_type=DataType.NUMBER),
                Property(name="molybdenumContent", data_type=DataType.NUMBER),
            ],
            vectorizer_config=vec_config,
        )
        logger.info("✓ Created: MaterialIndex")

        # ========================================================================
        # STEP 2: Add reference properties
        # ========================================================================

        logger.info("\nAdding reference properties...")

        # NickelBasedSuperalloy references
        client.collections.get("NickelBasedSuperalloy").config.add_reference(
            ReferenceProperty(name="hasComposition", target_collection="Composition")
        )
        client.collections.get("NickelBasedSuperalloy").config.add_reference(
            ReferenceProperty(name="hasVariant", target_collection="Variant")
        )
        # Direct references for flat model
        client.collections.get("NickelBasedSuperalloy").config.add_reference(
            ReferenceProperty(name="hasPropertySet", target_collection="PropertySet")
        )
        client.collections.get("NickelBasedSuperalloy").config.add_reference(
            ReferenceProperty(name="hasProcessingRoute", target_collection="ProcessingRoute")
        )
        logger.info("  → NickelBasedSuperalloy references")

        # Composition references
        client.collections.get("Composition").config.add_reference(
            ReferenceProperty(name="hasComponent", target_collection="CompositionEntry")
        )
        logger.info("  → Composition references")

        # CompositionEntry references
        client.collections.get("CompositionEntry").config.add_reference(
            ReferenceProperty(name="hasElement", target_collection="Element")
        )
        client.collections.get("CompositionEntry").config.add_reference(
            ReferenceProperty(name="hasMassFraction", target_collection="Quantity")
        )
        logger.info("  → CompositionEntry references")

        # Variant references
        client.collections.get("Variant").config.add_reference(
            ReferenceProperty(name="hasComposition", target_collection="Composition")
        )
        client.collections.get("Variant").config.add_reference(
            ReferenceProperty(name="hasFormType", target_collection="FormType")
        )
        client.collections.get("Variant").config.add_reference(
            ReferenceProperty(name="hasPropertySet", target_collection="PropertySet")
        )
        client.collections.get("Variant").config.add_reference(
            ReferenceProperty(name="hasProcessingRoute", target_collection="ProcessingRoute")
        )
        client.collections.get("Variant").config.add_reference(
            ReferenceProperty(name="hasMaterialIndex", target_collection="MaterialIndex")
        )
        logger.info("  → Variant references")

        # ProcessingRoute references
        client.collections.get("ProcessingRoute").config.add_reference(
            ReferenceProperty(name="hasFormType", target_collection="FormType")
        )
        logger.info("  → ProcessingRoute references")

        # PropertySet references
        client.collections.get("PropertySet").config.add_reference(
            ReferenceProperty(name="measuresProperty", target_collection="MechanicalProperty")
        )
        client.collections.get("PropertySet").config.add_reference(
            ReferenceProperty(name="hasMeasurement", target_collection="Measurement")
        )
        logger.info("  → PropertySet references")

        # Measurement references
        client.collections.get("Measurement").config.add_reference(
            ReferenceProperty(name="hasQuantity", target_collection="Quantity")
        )
        client.collections.get("Measurement").config.add_reference(
            ReferenceProperty(name="hasTestCondition", target_collection="TestCondition")
        )
        logger.info("  → Measurement references")

        # TestCondition references
        client.collections.get("TestCondition").config.add_reference(
            ReferenceProperty(name="hasTemperature", target_collection="Quantity")
        )
        logger.info("  → TestCondition references")

        logger.info("\n✓ All reference properties added")

        # ========================================================================
        # STEP 3: Seed fixed vocabulary (idempotent with UUIDv5)
        # ========================================================================

        logger.info("\nSeeding fixed vocabularies...")

        # Seed MechanicalProperty types
        prop_types = {
            "TensileStrength": "Tensile Strength",
            "YieldStrength": "Yield Strength",
            "Elongation": "Elongation",
            "Hardness": "Hardness",
            "ElasticModulus": "Elastic Modulus",
            "CreepRupture": "Creep Rupture",
            "ReductionOfArea": "Reduction of Area",
        }

        mp_col = client.collections.get("MechanicalProperty")
        existing_props = {
            (obj.properties or {}).get("propertyType")
            for obj in mp_col.iterator()
        }

        for prop_type, prop_name in prop_types.items():
            if prop_type not in existing_props:
                mp_col.data.insert(
                    uuid=sid(NS, f"MechanicalProperty:{prop_type}"),
                    properties={
                        "name": prop_name,
                        "propertyType": prop_type
                    }
                )
        logger.info(f"  ✓ Seeded {len(prop_types)} MechanicalProperty types")

        # Seed Element vocabulary
        elements = [
            "Ni", "Cr", "Co", "Mo", "W", "Al", "Ti", "Fe", "Nb", "Ta",
            "C", "B", "Si", "Mn", "Cu", "Nb+Ta", "S", "P"
        ]

        el_col = client.collections.get("Element")
        existing_elements = {
            (obj.properties or {}).get("symbol")
            for obj in el_col.iterator()
        }

        for symbol in elements:
            if symbol not in existing_elements:
                el_col.data.insert(
                    uuid=sid(NS, f"Element:{symbol}"),
                    properties={
                        "symbol": symbol,
                        "label": symbol
                    }
                )
        logger.info(f"  ✓ Seeded {len(elements)} Element types")

        # Seed common FormTypes
        form_types = ["bar", "plate", "sheet", "rod", "wire", "tube", "forging"]

        ft_col = client.collections.get("FormType")
        existing_forms = {
            (obj.properties or {}).get("formTypeName")
            for obj in ft_col.iterator()
        }

        for form in form_types:
            if form not in existing_forms:
                ft_col.data.insert(
                    uuid=sid(NS, f"FormType:{form}"),
                    properties={"formTypeName": form}
                )
        logger.info(f"  ✓ Seeded {len(form_types)} FormType values")

        # ========================================================================
        # Summary
        # ========================================================================

        logger.info("\n" + "=" * 70)
        logger.info("Schema Setup Complete!")
        logger.info("=" * 70)
        logger.info("\nCollections created:")
        for col in client.collections.list_all():
            logger.info(f"  • {col}")

        logger.info("\nKey improvements from v9.0:")
        logger.info("  ✓ Variant class (separates alloy from specific forms)")
        logger.info("  ✓ PropertySet (groups measurements by property type)")
        logger.info("  ✓ TestCondition (reusable test environment)")
        logger.info("  ✓ ProcessingRoute + FormType (manufacturing details)")
        logger.info("  ✓ MaterialIndex (fast query optimization)")
        logger.info("\nReady for data ingestion!")

    except Exception as e:
        logger.error(f"✗ Setup failed: {e}", exc_info=True)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()