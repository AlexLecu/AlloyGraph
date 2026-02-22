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
            "- Elements sum to EXACTLY 100.0 wt%.\n"
            "- |Lattice mismatch| < 0.5% target, > 0.8% rejected.\n"
            "- Md_avg < 0.940 safe, > 0.960 Elevated TCP. Re+0.027/%, W+0.019/%, Nb+0.028/% per wt%.\n"
            "- Re < 5%, W < 6%, Re+W+Mo < 12%. Nb ≤ 1.5% wrought.\n\n"

            "PROPERTY FORMULAS (use to compute required γ'):\n"
            "- Wrought: YS ≈ 520+13×γ'%, EL ≈ 28-0.28×γ'%. Cast: YS ≈ 400+10×γ'%, EL ≈ 18-0.25×γ'%.\n"
            "- UTS ≈ YS × 1.3-1.5 (wrought), × 1.1-1.3 (cast). EM ≈ Reuss bound; W(411), Mo(329) boost it.\n"
            "- Match ALL targets within ±10%. Do not over-engineer.\n\n"

            "ALLOY CLASSES:\n"
            "- LOW γ' (2-20%): Al+Ti+Ta < 4%. MEDIUM (30-50%): 5-7%. HIGH (60-75%): 8-12%.\n"
            "- Wrought limits: Al+Ti+Ta ≤ 7%, Nb ≤ 1.5%, γ' ≤ 50%. Cast: Al+Ti+Ta ≤ 10%, γ' ≤ 65%.\n\n"

            "COMPOSITION GUIDELINES:\n"
            "- Wrought disc: Cr 10-14%, Co 15-20%, Mo 2-4%, Ta 1.5-3%, Al 2-4%, Ti 1-3%, Nb 0.5-1.5%.\n"
            "- Cast: Cr 5-20%. Prefer Al/Ti for strength before Re/W (zero Md penalty).\n"
            "- Ta ≥ 1.5% in modern alloys (η phase suppression, γ' strengthening).\n"
            "- Polycrystalline alloys benefit from small C, B, Zr additions for grain boundary strength.\n\n"

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
