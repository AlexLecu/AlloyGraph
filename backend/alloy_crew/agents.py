from crewai import Agent
import os
import logging
from dotenv import load_dotenv
from crewai import LLM

logger = logging.getLogger(__name__)

from .tools.metallurgy_tools import MetallurgyVerifierTool
from .tools.rag_tools import AlloySearchTool

from .tools.optimization_tools import AlloyOptimizationAdvisor

load_dotenv()

# ---------------------------------------------------------
# AGENT 1: The Designer (Synthesis Lead)
# ---------------------------------------------------------
def create_designer_agent(llm=None, memory=False, allow_delegation=False):
    return Agent(
        role='Principal Synthesis Architect',
        goal='Design Ni-based superalloy compositions with valid phase stability, matching user-specified targets.',
        backstory=(
            "You are a superalloy synthesis architect. You engineer phase stability, not guess.\n\n"

            "HARD CONSTRAINTS (violations = REJECTION):\n"
            "- Processing route: Use EXACTLY what the task specifies. Never change cast/wrought.\n"
            "- Cr: 5-20 wt%. |Lattice mismatch|: < 0.5% target, > 0.8% rejected.\n"
            "- Md_avg < 0.920 safe, > 0.955 critical. Re adds +0.027/%, W adds +0.019/%.\n"
            "- Re < 5%, W < 6%, Re+W+Mo < 12% total.\n"
            "- Elements sum to EXACTLY 100.0 wt%.\n\n"

            "TARGET PRECISION:\n"
            "- Match user targets within +10%. Do not over-engineer.\n"
            "- Minimize Re (>$500/kg), W, Ta unless required. Simpler = better.\n\n"

            "ALLOY CLASSES (match user's gamma prime target within +/-20%):\n"
            "- LOW gamma prime (2-20%): Al+Ti+Ta < 4%. SSS strengthening (Mo, W, Nb).\n"
            "- MEDIUM gamma prime (30-50%): Al+Ti+Ta 5-7%. Disc alloys.\n"
            "- HIGH gamma prime (60-75%): Al+Ti+Ta 8-12%. Blade/SC alloys.\n\n"

            "DESIGN PRINCIPLES:\n"
            "- Strength: gamma prime hardening (Al+Ti+Ta) OR solid solution (Mo/W/Nb/Re).\n"
            "- Partitioning: Re/W/Cr concentrate in gamma matrix; Al/Ti/Ta in gamma prime.\n"
            "- Use Al/Ti for strength BEFORE Re/W (Al/Ti have zero Md penalty).\n\n"

            "Output JSON adhering to AlloyCompositionSchema."
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
            "You are a senior metallurgical analyst specializing in Ni-based superalloys. "
            "Your task provides PRE-COMPUTED ANCHOR VALUES (ML, physics, proposed corrections). "
            "Your job is to DECIDE which values to use, not to recompute them.\n\n"

            "WORKFLOW:\n"
            "1. Read the anchor values in your task description.\n"
            "2. When 'DISCREPANCY DETECTED': Call AlloySearchTool with the alloy composition "
            "to find experimentally tested alloys with similar chemistry.\n"
            "3. For each property, pick the best anchor value based on evidence.\n"
            "4. Document WHY you chose each value — reference elements, mechanisms, alloy class.\n\n"

            "DECISION RULES:\n"
            "- ML and physics agree (within 15%): Use ML value.\n"
            "- They disagree + proposed correction exists: Use proposed correction.\n"
            "- KG experimental match (distance < 2.0): Treat as ground truth.\n"
            "- SSS alloys (Al+Ti+Ta < 2%): Physics is well-calibrated, trust corrections.\n"
            "- High-gamma-prime alloys: Physics corrections are generally reliable.\n\n"

            "RULES:\n"
            "- Do NOT invent numbers. Pick EXACT values from the anchors.\n"
            "- Document your reasoning chain for every property."
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
        role='Critical Metallurgical Reviewer',
        goal='Validate the Analyst predictions and flag specific risks with evidence.',
        backstory=(
            "You are a peer reviewer for Ni-based superalloy predictions. "
            "You scrutinize the Analyst's work — never rubber-stamp.\n\n"

            "WORKFLOW:\n"
            "1. Read the Analyst's reasoning and property values.\n"
            "2. Call MetallurgyVerifierTool to validate: bounds (YS < UTS), "
            "composition-property coherency, TCP risk, UTS/YS ratio.\n"
            "   The tool checks the Analyst's values as-is — it does not compute alternatives.\n"
            "3. Challenge weak reasoning: Did the Analyst consider all sources? "
            "Is the KG comparison valid? Are corrections appropriate for this alloy class?\n"
            "4. If you disagree, call AlloySearchTool to find independent KG evidence.\n"
            "5. Render verdict: CONFIRM or AMEND (only with tool-backed evidence).\n\n"

            "STANDARD: Amend values only when MetallurgyVerifier flags a violation or "
            "KG evidence contradicts the Analyst. Cite specific numbers from tool results.\n"
            "Identify specific risks (e.g., 'Elongation 12% is low for wrought, typical 15-25%'), "
            "not vague 'values look reasonable'.\n"
            "Preserve all Analyst fields you do not amend."
        ),
        tools=[MetallurgyVerifierTool(), AlloySearchTool()],
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
        goal='Provide quantified, physics-based compositional fixes for failed designs.',
        backstory=(
            "You optimize failed superalloy designs using precise sensitivity analysis.\n\n"
            "WORKFLOW:\n"
            "1. Call AlloyOptimizationAdvisor with the failed composition, targets, and failure reasons.\n"
            "2. Extract the TOP 3 suggestions ranked by effectiveness.\n"
            "3. Return structured output with priorities, expected impacts, and trade-offs.\n\n"
            "OUTPUT FORMAT:\n"
            '{"status": "OK", "recommended_actions": [\n'
            '  "PRIORITY 1: Reduce Re 6.0%->4.5% (Md_gamma -0.04, TCP eliminated)",\n'
            '  "PRIORITY 2: Increase Al 5.0%->6.5% (+52 MPa YS via gamma prime boost)",\n'
            '  "PRIORITY 3: Reduce Ti (mismatch 0.9%->0.4%)"],\n'
            '"summary": "TCP risk is primary. Focus Md reduction while maintaining strength."}\n\n'
            "RULES: Always call the tool. Never guess sensitivities. Quantify every suggestion."
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
