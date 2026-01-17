from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Dict, Optional, Literal, List, Any, Union


# =============================================================================
# Optimizations Tool Schemas
# =============================================================================

class ElementSuggestion(BaseModel):
    """Single element adjustment suggestion."""
    element: str
    current_wt: float
    suggested_wt: float
    delta_wt: float
    reason: str
    expected_md_change: float = 0.0
    expected_gp_change: float = 0.0
    trade_offs: str = ""


class SuggestionGroup(BaseModel):
    """Group of related suggestions to address a specific issue."""
    issue: str
    priority: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    suggestions: List[ElementSuggestion]
    rationale: str


class CompositionSensitivityInput(BaseModel):
    """Input for composition sensitivity analysis."""
    composition: Dict[str, float] = Field(..., description="Current alloy composition in wt%")
    target_properties: Dict[str, float] = Field(..., description="Target properties to achieve")
    current_properties: Dict[str, float] = Field(..., description="Current predicted properties")
    failure_reasons: List[str] = Field(..., description="List of failure reasons from validation")
    processing: Literal["cast", "wrought"] = Field("cast", description="Processing route")


# =============================================================================
# Design Output Schemas
# =============================================================================


class AlloyCompositionSchema(BaseModel):
    """
    Scientific Data Contract for an Alloy Composition.
    Enforces mass balance (Sum = 100%) and valid inputs.
    """
    elements: Dict[str, float] = Field(
        ...,
        description="Dictionary of elements and their weight percentages (e.g., {'Ni': 60.5, 'Al': 5.0})."
    )
    process_route: Literal["cast", "wrought"] = Field(
        ...,
        description="The processing method used to manufacture this alloy (strictly 'cast' or 'wrought')."
    )
    
    @model_validator(mode='after')
    def check_sum(self):
        elements = self.elements
        if elements:
            total = sum(elements.values())
            # Tolerance of 0.1% for floating point/rounding flexibility
            if not (99.9 <= total <= 100.1):
                raise ValueError(f"Composition must sum to 100% (+/- 0.1%). Current sum: {total:.2f}%")
        return self

    @field_validator('elements')
    def check_positive(cls, v):
        for el, wt in v.items():
            if wt < 0:
                raise ValueError(f"Element {el} cannot have negative weight percent.")
            if wt > 100:
                raise ValueError(f"Element {el} cannot exceed 100%.")
        return v

class AlloyPropertySchema(BaseModel):
    """
    Scientific Data Contract for a Mechanical Property.
    Captures uncertainty and domain validity.
    """
    value: float = Field(..., description="The predicted or measured value.")
    unit: str = Field(..., description="Unit of measurement (e.g., 'MPa', '%').")
    uncertainty_interval: Optional[float] = Field(None, description="Confidence interval (+/- step).")
    in_domain_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0-1) indicating if this alloy is within the ML model's training domain."
    )
    notes: Optional[str] = Field(None, description="Any warnings or context (e.g., 'Extrapolated').")

class FullAlloyReportSchema(BaseModel):
    """
    The final output object passed between agents.
    """
    alloy_name: str = Field(..., description="Name or ID of the alloy.")
    composition: AlloyCompositionSchema
    properties: Dict[str, AlloyPropertySchema] = Field(
        ..., 
        description="Keyed by property name (e.g., 'Yield Strength')."
    )
    physics_audit_passed: bool = Field(False, description="True only if the Physicist approves.")
    audit_penalties: list[str] = Field(default_factory=list, description="List of penalty reasons if any.")

class DesignOutput(BaseModel):
    """
    Output structure for the Alloy Designer agent.
    """
    reasoning: str = Field(..., description="Metallurgical reasoning for the proposed composition.")
    composition: Dict[str, float] = Field(..., description="The proposed alloy composition (elements summing to 100%).")
    processing: Literal["cast", "wrought"] = Field("cast", description="The processing route (cast or wrought).")


class ValidationOutput(BaseModel):
    status: Literal["OK", "FAIL"]
    temperature_c: int
    composition_wt_percent: Dict[str, float]
    ml_prediction: Dict[str, Any]
    errors: List[str] = []

class FusionMeta(BaseModel):
    kg_similarity_max: float = 0.0
    ml_weight: float = 0.0
    kg_weight: float = 0.0
    data_conflict: bool = False
    is_kg_anchored: bool = False

class ArbitrationOutput(BaseModel):
    status: str
    summary: str = Field("", description="Summary of Data Fusion (e.g. contains 'Anchoring')")
    processing: Literal["cast", "wrought", "unknown"] = Field(..., description="Alloy processing type (cast/wrought)")
    penalty_score: float
    tcp_risk: str
    properties: Dict[str, Any]
    property_intervals: Dict[str, Any] = Field(default_factory=dict, description="Uncertainty intervals for properties")
    metallurgy_metrics: Dict[str, Any] = {}
    fusion_meta: FusionMeta
    confidence: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = []

class AuditPenalty(BaseModel):
    name: str
    value: Union[float, str]
    reason: str

class OptimizationOutput(BaseModel):
    """Output from the Optimization Advisor agent."""
    status: Literal["OK", "ERROR"] = "OK"
    suggestion_groups: List[Dict[str, Any]] = Field(default_factory=list, description="Groups of optimization suggestions")
    summary: str = Field("", description="Summary of optimization analysis")
    recommended_actions: List[str] = Field(default_factory=list, description="Prioritized list of recommended adjustments")

class PhysicsAuditOutput(BaseModel):
    status: Literal["PASS", "REJECT", "FAIL"]
    processing: str = Field(..., description="Alloy processing type (cast/wrought/unknown)")
    penalty_score: float = 0.0
    tcp_risk: str = "LOW"
    properties: Dict[str, Any]
    property_intervals: Dict[str, Any] = Field(default_factory=dict, description="Uncertainty intervals for properties")
    metallurgy_metrics: Dict[str, Any]
    audit_penalties: List[AuditPenalty] = []
    recommended_repairs: List[str] = []
    errors: List[str] = []
    confidence: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""


class PropertyCorrection(BaseModel):
    """Single property correction with reasoning."""
    property_name: str = Field(..., description="Property that was corrected")
    original_value: float = Field(..., description="Original ML/fusion prediction")
    corrected_value: float = Field(..., description="Physics-corrected value")
    correction_reason: str = Field(..., description="Why this correction was applied")
    physics_constraint: str = Field("", description="The physics rule or constraint applied")


class CorrectedPropertiesOutput(BaseModel):
    """Output from physics corrections agent."""
    status: Literal["PASS", "REJECT", "FAIL"]
    processing: str = Field(..., description="Alloy processing type")
    penalty_score: float = 0.0
    tcp_risk: str = "LOW"
    properties: Dict[str, Any] = Field(..., description="FINAL corrected properties")
    property_intervals: Dict[str, Any] = Field(default_factory=dict)
    metallurgy_metrics: Dict[str, Any]
    audit_penalties: List[AuditPenalty] = []
    recommended_repairs: List[str] = []
    errors: List[str] = []
    confidence: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = Field("", description="Metallurgical analysis from Physicist")
    corrections_applied: List[PropertyCorrection] = Field(
        default_factory=list,
        description="List of corrections applied with reasoning"
    )
    corrections_explanation: str = Field(
        "",
        description="Overall explanation of why corrections were needed and their implications"
    )
