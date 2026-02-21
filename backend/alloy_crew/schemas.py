import json

from pydantic import BaseModel, BeforeValidator, Field
from typing import Annotated, Dict, Literal, List, Any, Union


def normalize_composition_to_str(value: Any) -> str:
    """Normalize composition input to a JSON string.ead."""
    if isinstance(value, dict):
        return json.dumps({str(k): float(v) for k, v in value.items()})
    if isinstance(value, str):
        return value
    return json.dumps(value)


CompositionStr = Annotated[str, BeforeValidator(normalize_composition_to_str)]


class AuditPenalty(BaseModel):
    name: str
    value: Union[float, str]
    reason: str


class PropertyCorrection(BaseModel):
    """Single property correction with reasoning."""
    property_name: str = Field(..., description="Property that was corrected")
    original_value: float = Field(..., description="Original ML/fusion prediction")
    corrected_value: float = Field(..., description="Physics-corrected value")
    correction_reason: str = Field(..., description="Why this correction was applied")
    physics_constraint: str = Field("", description="The physics rule or constraint applied")


class PhysicsAuditWithCorrectionsOutput(BaseModel):
    """Combined physics audit + corrections output with investigation support."""
    status: Literal["PASS", "REJECT", "FAIL"]
    processing: str = Field(..., description="Alloy processing type (cast/wrought/unknown)")
    penalty_score: float = 0.0
    tcp_risk: str = "LOW"
    properties: Dict[str, Any] = Field(..., description="FINAL corrected properties after physics adjustments")
    property_intervals: Dict[str, Any] = Field(default_factory=dict, description="Uncertainty intervals for properties")
    metallurgy_metrics: Dict[str, Any] = Field(default_factory=dict, description="Computed metrics (Md, mismatch, γ', etc.)")
    audit_penalties: List[AuditPenalty] = Field(default_factory=list, description="Physics violations found")
    errors: List[str] = Field(default_factory=list, description="Critical errors encountered")
    confidence: Dict[str, Any] = Field(default_factory=dict, description="Confidence scores (similarity, level)")
    explanation: str = Field("", description="3-5 sentence metallurgical analysis")
    # Corrections fields
    corrections_applied: List[PropertyCorrection] = Field(
        default_factory=list,
        description="List of physics-based corrections applied (SSS, γ' temp degradation, etc.)"
    )
    corrections_explanation: str = Field(
        "",
        description="Summary of corrections: why needed, implications, and final property validity"
    )
    # Investigation fields (for agent-driven corrections)
    investigation_findings: str = Field(
        "",
        description="What the agent learned from KG search when investigating discrepancies"
    )
    source_reliability: str = Field(
        "",
        description="Which source (ML/Physics/KG) the agent determined to be most reliable and why"
    )
    # Multi-agent reasoning fields (Analyst + Reviewer architecture)
    analyst_reasoning: str = Field(
        "",
        description="Analyst's reasoning chain: how sources were compared, what evidence supports the final values"
    )
    reviewer_assessment: str = Field(
        "",
        description="Reviewer's critical assessment: what's sound, what's weak, what was overlooked"
    )
