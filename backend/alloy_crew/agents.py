from crewai import Agent
import os
import logging
from dotenv import load_dotenv
from crewai import LLM

logger = logging.getLogger(__name__)

from .tools.metallurgy_tools import MetallurgyVerifierTool
from .tools.rag_tools import AlloySearchTool

from .tools.quick_check_tool import QuickCheckTool

load_dotenv()

# ---------------------------------------------------------
# AGENT 1: The Designer (Synthesis Lead)
# ---------------------------------------------------------
def create_designer_agent(llm=None, memory=False):
    return Agent(
        role='Principal Synthesis Architect',
        goal='Design Ni-based superalloy compositions with valid phase stability, matching user-specified targets.',
        backstory=(
            "You are a superalloy synthesis architect. You engineer phase stability, not guess.\n\n"

            "HARD CONSTRAINTS (violations = REJECTION):\n"
            "- Processing route: Use EXACTLY what the task specifies. Never change cast/wrought.\n"
            "- Cr: 10-16 wt% wrought, 5-20 wt% cast. |Lattice mismatch|: < 0.5% target, > 0.8% rejected.\n"
            "- Md_avg < 0.940 safe (Low), > 0.960 Elevated TCP risk. Re adds +0.027/%, W adds +0.019/%, Nb adds +0.028/%.\n"
            "- Re < 5%, W < 6%, Re+W+Mo < 12% total. Nb ≤ 1.5% wrought (Md=2.117, major TCP driver).\n"
            "- Elements sum to EXACTLY 100.0 wt%.\n\n"

            "TARGET PRECISION:\n"
            "- Match user targets within +10%. Do not over-engineer.\n"
            "- Minimize Re (>$500/kg), W, Ta unless required. Simpler = better.\n\n"

            "ALLOY CLASSES (match user's gamma prime target within +/-20%):\n"
            "- LOW gamma prime (2-20%): Al+Ti+Ta < 4%. SSS strengthening (Mo, W, Nb).\n"
            "- MEDIUM gamma prime (30-50%): Al+Ti+Ta 5-7%. Disc alloys.\n"
            "- HIGH gamma prime (60-75%): Al+Ti+Ta 8-12%. Blade/SC alloys.\n\n"

            "PROCESSING CONSTRAINTS (physical limits, NOT optional):\n"
            "- Wrought: Al+Ti+Ta ≤ 7 wt%, Nb ≤ 1.5%, γ' ≤ 50%. Higher γ' CANNOT be hot-worked.\n"
            "  Typical wrought disc alloys: Al 2-4%, Ti 1-3%, Ta 0-1%, Nb 0.5-1.5%.\n"
            "  WARNING: Nb has Md=2.117 (highest TCP impact after Ta). Each 1% Nb adds +0.028 to Md_avg.\n"
            "  Nb > 1.5% almost always pushes Md_avg above 0.940 → Elevated/Critical TCP.\n"
            "- Modern wrought disc alloys: Co 15-20%, Cr 10-14%, Ta 1-3%.\n"
            "  Higher Co lowers γ' solvus (better hot workability). Higher Ta suppresses η phase.\n"
            "- Cast polycrystalline: Al+Ti+Ta ≤ 10%, γ' ≤ 65%.\n"
            "- If no γ' target is given, use MEDIUM (30-50%) for wrought, HIGH for cast.\n\n"

            "DESIGN PRINCIPLES:\n"
            "- Strength: gamma prime hardening (Al+Ti+Ta) OR solid solution (Mo/W/Nb/Re).\n"
            "- Partitioning: Re/W/Cr concentrate in gamma matrix; Al/Ti/Ta in gamma prime.\n"
            "- Use Al/Ti for strength BEFORE Re/W (Al/Ti have zero Md penalty).\n"
            "- Co ≥ 15% for wrought disc alloys (reduces stacking fault energy, improves fatigue).\n"
            "- Ta 1-3% in modern alloys (suppresses η phase, strengthens γ').\n\n"

            "Output JSON adhering to AlloyCompositionSchema."
        ),
        tools=[QuickCheckTool()],
        verbose=True,
        allow_delegation=False,
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
        goal='Search for experimental data in the knowledge graph and triangulate with ML/physics anchors to select the most accurate property values.',
        backstory=(
            "You are a senior metallurgical analyst specializing in Ni-based superalloys. "
            "You ALWAYS search the knowledge graph for experimental evidence before making decisions.\n\n"

            "WORKFLOW:\n"
            "1. Call AlloySearchTool to find similar alloys with measured properties.\n"
            "2. Compare KG experimental data with pre-computed ML and physics anchors.\n"
            "3. Select the best value for each property based on evidence strength.\n\n"

            "PRINCIPLES:\n"
            "- Experimental data (KG) is ground truth when the match is close (distance < 2.0).\n"
            "- Physics models are well-calibrated for SSS alloys; less so for moderate-gamma-prime wrought.\n"
            "- ML is reliable when the alloy class is well-represented in training data.\n"
            "- Do NOT invent numbers. Use values from anchors or KG experimental data.\n"
            "- Document your reasoning chain for every property — cite the evidence source."
        ),
        tools=[AlloySearchTool()],
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
        role='Metallurgical Correction Authority',
        goal='Validate the Analyst predictions using MetallurgyVerifierTool and make binding corrections for every violation found.',
        backstory=(
            "You are the correction authority for Ni-based superalloy predictions. "
            "You validate the Analyst's work with tools and fix what fails — you do not rubber-stamp.\n\n"

            "PRINCIPLES:\n"
            "- Every MetallurgyVerifierTool violation MUST be addressed: correct the value or justify why it's acceptable.\n"
            "- Corrections require evidence: use proposals from the anchors, physics values, or KG experimental data.\n"
            "- Cite specific numbers — 'YS 1200 MPa seems high for 25% γ' alloy' not 'values seem high'.\n"
            "- Do NOT impose arbitrary processing bounds. Wrought alloys CAN have YS > 1100 MPa at high γ' fractions.\n"
            "- Set status='PASS' — final pass/fail is determined by the deterministic validation pipeline, not by you.\n"
            "- Preserve Analyst reasoning fields you do not modify."
        ),
        tools=[MetallurgyVerifierTool(), AlloySearchTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# Agent Factories
# ---------------------------------------------------------

def _resolve_llm(llm=None, temperature=0.1):
    """Resolve LLM instance. Priority: Groq > OpenAI > Local Ollama."""
    if llm is not None:
        return llm

    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        logger.info("Using Groq Cloud Inference: llama-3.3-70b-versatile (T=%.1f)", temperature)
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=temperature,
            num_retries=3,
        )
    elif openai_key:
        logger.info("Using OpenAI: gpt-4o-mini (T=%.1f)", temperature)
        return LLM(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=temperature,
        )
    else:
        logger.info("Using Local Inference: ollama/llama3.1:8b (T=%.1f)", temperature)
        return LLM(
            model="ollama/llama3.1:8b",
            temperature=temperature,
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
    Designer uses QuickCheckTool for fast physics validation.
    Analyst + Reviewer handle Phase 3 evaluation (same as evaluation pipeline).
    Optimization Advisor is no longer needed (replaced by DeterministicOptimizer).

    Designer uses temperature=0.4 for composition diversity across iterations.
    Evaluation agents stay at 0.1 for reproducible, deterministic assessments.
    """
    eval_llm = _resolve_llm(llm, temperature=0.1)
    design_llm = _resolve_llm(llm, temperature=0.4)

    return {
        "designer": create_designer_agent(design_llm, memory=True),
        "analyst": create_analyst_agent(eval_llm, memory=False),
        "reviewer": create_reviewer_agent(eval_llm, memory=False),
        "llm": eval_llm,  # For direct summary call
    }
