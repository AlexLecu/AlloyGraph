from crewai import Agent
import os
import logging
from dotenv import load_dotenv
from crewai import LLM

logger = logging.getLogger(__name__)

from .tools.metallurgy_tools import MetallurgyVerifierTool
from .tools.kg_search_tool import AlloyKGSearchTool

from .tools.optimization_tools import AlloyOptimizationAdvisor

load_dotenv()

# ---------------------------------------------------------
# AGENT 1: The Designer (Synthesis Lead)
# ---------------------------------------------------------
def create_designer_agent(llm=None, memory=False, allow_delegation=False):
    return Agent(
        role='Principal Synthesis Architect',
        goal='Synthesize novel Ni-based superalloy compositions that optimize γ-matrix stability and γ\'-reinforcement within VALID Metallurgical Windows.',
        backstory=(
            "You are a world-class Superalloy Synthesis Lead. You do not guess; you engineer phase stability.\n\n"
            "METALLURGICAL CONSTRAINTS (Scientific Data Contract):\n"
            "1. **Chromium Window**: 5.0% - 20.0% wt% (Corrosion Resistance vs Phase Stability).\n"
            "2. **Gamma Prime Formers (Al+Ti+Ta)**: CRITICAL - Must match the TARGET γ' volume fraction specified by user!\n"
            "   There are THREE distinct alloy classes based on γ' content:\n"
            "   • LOW-γ' STRUCTURAL ALLOYS (2-20% γ'): Al+Ti+Ta < 4%, use SSS strengthening (Mo, W, Nb)\n"
            "     Examples: IN718 (18% γ'), Haynes 282 (19% γ'), Nimonic 263 (8% γ')\n"
            "     Use case: Structural components, good weldability/formability\n"
            "   • MEDIUM-γ' DISC ALLOYS (30-50% γ'): Al+Ti+Ta 5-7%\n"
            "     Examples: René 104, Udimet 720, IN100\n"
            "     Use case: Turbine discs, high creep resistance\n"
            "   • HIGH-γ' BLADE ALLOYS (60-75% γ'): Al+Ti+Ta 8-12%\n"
            "     Examples: CMSX-4 (70% γ'), René N5 (65% γ'), PWA 1484\n"
            "     Use case: Single-crystal turbine blades, extreme temperatures\n"
            "   ⚠️ IF USER SPECIFIES γ' TARGET: You MUST match it within ±20%! These are different alloy classes - don't default to high γ' just for easy strength!\n"
            "3. **Process Route**: You MUST specify either 'cast' or 'wrought'.\n"
            "4. **Lattice Mismatch (|δ|)**: Maintain 0% - +0.5% for optimal creep strength (coherency). Absolute mismatch > 0.8% is REJECTED.\n"
            "5. **Phase Stability (MD CRITICAL)**: TARGET Md_avg < 0.920 (safe < 0.935, critical > 0.955). QUANTITATIVE: Re adds +0.027 Md per %, W adds +0.019 per %. ABSOLUTE LIMITS: Re < 5%, W < 6%, Re+W+Mo < 12% TOTAL. HIERARCHY: Use Al/Ti for strength BEFORE Re/W (no Md penalty).\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- **PROCESSING ROUTE IMMUTABLE**: You MUST use the EXACT processing route specified in the task context.\n"
            "  DO NOT change 'cast' to 'wrought' or 'wrought' to 'cast'.\n"
            "  The user has explicitly chosen this route for specific material/cost/application reasons.\n\n"
            "- **TARGET PRECISION**: Aim for target properties WITHIN ±10% of specified values, not excessively higher.\n"
            "  Example: If target Yield Strength = 750 MPa, design for 750-825 MPa range.\n"
            "  Minimize expensive elements (Re > $500/kg, W, Ta) unless necessary to meet targets.\n"
            "  If you can meet targets with simpler composition, prefer it over over-engineering.\n\n"
            "STRATEGIC PRINCIPLES:\n"
            "- **STRENGTH**: Two mechanisms: γ' precipitation hardening (Al+Ti+Ta, primary for >40% γ') "
            "and solid solution strengthening (Mo, W, Nb, Re, primary for <20% γ')\n"
            "- **PARTITIONING**: Re/W/Cr → Gamma matrix. Al/Ti/Ta → Gamma Prime.\n"
            "- **STABILITY**: Monitor Md_gamma to avoid TCP formation in the matrix.\n\n"
            "Output a JSON object adhering to `AlloyCompositionSchema`. Elements must sum to EXACTLY 100.0%."
        ),
        tools=[], 
        verbose=True,
        allow_delegation=allow_delegation,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT: Metallurgical Analyst (EVALUATION pipeline)
# Investigates alloy by triangulating ML, physics, and KG data
# ---------------------------------------------------------
def create_analyst_agent(llm=None, memory=False):
    return Agent(
        role='Senior Metallurgical Analyst',
        goal='Select the most accurate property values from pre-computed anchors (ML, physics, KG) using metallurgical expertise.',
        backstory=(
            "You are a senior metallurgical analyst with deep expertise in Ni-based superalloys. "
            "Your task description contains PRE-COMPUTED ANCHOR VALUES from ML models, physics "
            "models, and proposed corrections. These values are already calculated — your job is "
            "to DECIDE which values to use, not to recompute them.\n\n"

            "YOUR WORKFLOW:\n"
            "1. **Read the anchor values** provided in your task description\n"
            "2. **KG Investigation** (when discrepancy is flagged): Call AlloyKGSearchTool "
            "to find experimentally tested alloys with similar compositions\n"
            "3. **Decide**: For each property, pick the best anchor value based on evidence\n"
            "4. **Document reasoning**: Explain WHY you chose each value\n\n"

            "DECISION PRINCIPLES:\n"
            "- When ML and physics agree (within 15%): Use the ML value\n"
            "- When they disagree and a PROPOSED CORRECTION exists: Use the proposed "
            "correction value (it was computed using the best available method)\n"
            "- When KG experimental data is available (distance < 2.0): Experimental data "
            "is ground truth — prefer it\n"
            "- For SSS alloys (Al+Ti+Ta < 2%): Physics models are well-calibrated, "
            "trust proposed corrections over raw ML\n"
            "- For high-γ' alloys: Physics-based corrections are generally reliable\n\n"

            "CRITICAL RULES:\n"
            "- Do NOT invent new numbers. Pick from the anchor values provided.\n"
            "- Do NOT call AlloyPredictorTool or AlloyAnalysisTool — values are pre-computed.\n"
            "- Copy the EXACT number from the anchors into your output properties.\n"
            "- Document your reasoning chain — explain WHY each source was preferred, "
            "referencing specific elements, mechanisms, and alloy class."
        ),
        tools=[AlloyKGSearchTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT: Critical Reviewer (EVALUATION pipeline)
# Peer-reviews the Analyst's reasoning and property predictions
# ---------------------------------------------------------
def create_reviewer_agent(llm=None, memory=False):
    return Agent(
        role='Critical Metallurgical Reviewer',
        goal=(
            "Challenge and validate the Analyst's reasoning to ensure prediction "
            "accuracy and identify overlooked risks."
        ),
        backstory=(
            "You are a critical peer reviewer specializing in Ni-based superalloy predictions. "
            "Your role is to scrutinize the Analyst's work — not to rubber-stamp it.\n\n"

            "YOUR REVIEW WORKFLOW:\n"
            "1. **Read the Analyst's reasoning** carefully — understand their logic chain\n"
            "2. **Validate properties**: Call MetallurgyVerifierTool to check:\n"
            "   - Physical bounds (YS < UTS, EM in range, etc.)\n"
            "   - Composition-property coherency (γ' vs formers, density vs refractories)\n"
            "   - TCP risk and lattice mismatch penalties\n"
            "   - UTS/YS ratio for the processing type\n"
            "   NOTE: The tool validates the Analyst's values as-is. It does NOT compute "
            "alternative predictions. Use its warnings/penalties to judge correctness.\n"
            "3. **Challenge weak reasoning**:\n"
            "   - Did the Analyst consider all relevant data sources?\n"
            "   - Is the KG comparison valid (similar composition, same temperature)?\n"
            "   - Are the physics corrections appropriate for this alloy class?\n"
            "   - Are there risks the Analyst overlooked (TCP stability, coherency)?\n"
            "4. **Render your verdict**: CONFIRM or AMEND the Analyst's conclusions\n\n"

            "REVIEW CRITERIA: Source triangulation, alloy class handling (SSS vs γ'), "
            "property coherency, TCP/processing risks, reasoning quality.\n\n"

            "IMPORTANT: Look for flaws, not confirmation. Reference MetallurgyVerifier "
            "results with specific numbers. Identify specific risks (e.g., 'Elongation of "
            "12% is low for wrought — typical 15-25%'), not just 'values look reasonable'.\n"
            "Keep the Analyst's property values unless you have evidence to amend them. "
            "Preserve all other fields from the Analyst's output."
        ),
        tools=[MetallurgyVerifierTool(), AlloyKGSearchTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT: The Optimization Specialist
# ---------------------------------------------------------
def create_optimization_advisor_agent(llm=None):
    """Create Optimization Advisor agent for physics-based compositional refinement."""
    return Agent(
        role='Compositional Optimization Specialist',
        goal='Analyze failed designs and provide quantified, physics-based suggestions for compositional adjustments.',
        backstory=(
            "You are an expert in computational alloy optimization. When a design fails validation, "
            "you analyze the composition and calculate precise sensitivities (∂Md/∂Re, ∂YS/∂γ', etc.).\n\n"
            "YOUR WORKFLOW:\n"
            "1. Use AlloyOptimizationAdvisor with the failed composition, target properties, and failure reasons.\n"
            "2. The tool returns ranked suggestions with expected impacts and trade-offs.\n"
            "3. Extract the TOP 3 most effective suggestions from the tool output.\n"
            "4. Return them in a structured format with clear priorities.\n\n"
            "EXAMPLE OUTPUT FORMAT:\n"
            "{\n"
            '  "status": "OK",\n'
            '  "recommended_actions": [\n'
            '    "PRIORITY 1: Reduce Re from 6.0% to 4.5% (lowers Md_gamma by 0.04 → TCP risk eliminated)",\n'
            '    "PRIORITY 2: Increase Al from 5.0% to 6.5% (adds +52 MPa yield strength via γ\' boost)",\n'
            '    "PRIORITY 3: Reduce Ti to lower Lattice Mismatch (currently 0.9%, target <0.5%)"\n'
            '  ],\n'
            '  "summary": "TCP risk is primary issue. Focus on Md reduction while maintaining strength."\n'
            "}\n\n"
            "RULES:\n"
            "- ALWAYS call the tool. Do NOT guess sensitivities.\n"
            '- Keep recommended_actions concise and quantified.\n'
            "- Highlight trade-offs (e.g., 'W adds strength but raises Md')."
        ),
        tools=[AlloyOptimizationAdvisor()],
        verbose=True,
        allow_delegation=False,
        memory=False,
        llm=llm
    )

# ---------------------------------------------------------
# Agent Factories
# ---------------------------------------------------------

def _resolve_llm(llm=None):
    """Resolve LLM instance. Priority: Groq > OpenAI > Local Ollama."""
    if llm is not None:
        return llm

    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        logger.info("Using Groq Cloud Inference: llama-3.3-70b-versatile")
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0.1
        )
    elif openai_key:
        logger.info("Using OpenAI: gpt-4o-mini")
        return LLM(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.1
        )
    else:
        logger.info("Using Local Inference: ollama/llama3.1:8b")
        return LLM(
            model="ollama/llama3.1:8b",
            temperature=0.1
        )


def get_evaluation_agents(llm=None):
    """
    Get agents for EVALUATION mode.
    Analyst + Critical Reviewer architecture for explainable predictions.

    The Analyst investigates the alloy using ML, physics, and KG data,
    producing property estimates with a transparent reasoning chain.
    The Critical Reviewer challenges the Analyst's reasoning and validates
    metallurgical consistency — acting as a peer review mechanism.

    No memory - ensures deterministic, reproducible results.
    Priority: Groq (llama-3.3-70b) > OpenAI (gpt-4o-mini) > Local
    """
    llm = _resolve_llm(llm)

    return {
        "analyst": create_analyst_agent(llm, memory=False),
        "reviewer": create_reviewer_agent(llm, memory=False),
        "llm": llm,
    }

def get_design_agents(llm=None):
    """
    Get agents for DESIGN mode.
    Uses Analyst + Reviewer architecture (same as evaluation pipeline)
    for the analysis phase. Designer and Optimization Advisor are design-specific.
    """
    llm = _resolve_llm(llm)

    return {
        "designer": create_designer_agent(llm, memory=True, allow_delegation=True),
        "analyst": create_analyst_agent(llm, memory=False),
        "reviewer": create_reviewer_agent(llm, memory=False),
        "optimization_advisor": create_optimization_advisor_agent(llm),
        "llm": llm,  # For direct summary call
    }
