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

#: Cap on tool-use iterations per agent. CrewAI defaults to 25, which lets the
#: Analyst and Reviewer together reach ~56 LLM calls on a single row -- observed
#: at ~2M prompt tokens, 32x the median row and 76% of projected campaign spend.
#: The runaway is stochastic rather than tied to particular alloys, so a hard cap
#: is the only reliable bound. Healthy rows use 8-12 calls across both agents,
#: so 8 per agent leaves normal work untouched.
MAX_AGENT_ITER = 8

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
            "- UTS ≈ YS × 1.3-1.5 (wrought), × 1.1-1.3 (cast). EM ≈ Voigt-Reuss-Hill average; W(411), Mo(329) boost it.\n"
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
        max_iter=MAX_AGENT_ITER,
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
        max_iter=MAX_AGENT_ITER,
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
        max_iter=MAX_AGENT_ITER,
        memory=memory,
        llm=llm
    )



# ---------------------------------------------------------
# Agent Factories
# ---------------------------------------------------------

class _CacheBreakpointSafeLLM(LLM):
    """CrewAI LLM that strips the ``cache_breakpoint`` marker before dispatch.

    crewai >= 1.15 tags stable prompt prefixes with ``cache_breakpoint: True``
    so provider adapters can translate it into their own caching directive.
    The adapters strip it in ``base_llm``, but the litellm-backed ``LLM`` class
    used for Groq and other pass-through providers never does, so the flag
    reaches the API as an unknown message property. Groq rejects it outright:

        GroqException - 'messages.0' : for 'role:system' the following must be
        satisfied[('messages.0' : property 'cache_breakpoint' is unsupported)]

    Stripping here is safe for every provider: the marker is metadata for
    adapters, never content, and dropping it only forgoes an optional caching
    optimisation. Remove this shim once crewai strips it on the litellm path.
    """

    def _format_messages_for_provider(self, messages):
        formatted = super()._format_messages_for_provider(messages)
        return [
            {k: v for k, v in m.items() if k != "cache_breakpoint"}
            if isinstance(m, dict) else m
            for m in formatted
        ]


#: Minimum plausible length and required prefix for each provider's API key.
#: A value that fails this is treated as absent rather than passed to the API,
#: so provider selection can never be hijacked by a placeholder such as "sk-".
_KEY_SHAPES = {
    # DeepInfra keys carry no distinctive prefix, so length is the only check.
    "DEEPINFRA_API_KEY": ("", 24),
    "TOGETHER_API_KEY": ("tgp_", 30),
    "GROQ_API_KEY": ("gsk_", 20),
    "OPENAI_API_KEY": ("sk-", 20),
}


def valid_api_key(env_var: str) -> str:
    """Return the key if it looks real, otherwise an empty string.

    Placeholder values are common in checked-in .env templates ("sk-",
    "your-key-here", ""). Left unchecked they are truthy, so a bare "sk-" will
    silently win provider selection over an unset-but-intended provider and
    every call then fails 401. Failing fast here is much cheaper than
    discovering it part-way through a campaign.
    """
    raw = (os.getenv(env_var) or "").strip().strip("\"'")
    if not raw:
        return ""

    prefix, min_len = _KEY_SHAPES.get(env_var, ("", 20))
    placeholder = raw.lower() in {"none", "null", "changeme", "todo"} or "your" in raw.lower()

    if placeholder or len(raw) < min_len or (prefix and not raw.startswith(prefix)):
        logger.warning(
            "%s is set but does not look like a real key (len=%d, expected prefix %r, "
            "minimum length %d) — ignoring it for provider selection.",
            env_var, len(raw), prefix, min_len,
        )
        return ""
    return raw


def _resolve_llm(llm=None, temperature=0.1):
    """Resolve the agent LLM. Priority: DeepInfra > Together > Groq > OpenAI > Ollama.

    The published configuration is Llama 3.3 70B. Groq decommissioned that
    model on 2026-08-16. DeepInfra and Together both serve the same weights as
    meta-llama/Llama-3.3-70B-Instruct-Turbo; DeepInfra is preferred on cost.
    Groq remains in the chain for other models and for accounts holding access.

    Only keys passing valid_api_key() are considered, so a malformed value is
    skipped rather than selected and then rejected by the provider.

    ALLOYGRAPH_LLM_MODEL overrides the model for whichever provider is chosen.
    Give it the provider-native id (e.g. meta-llama/Llama-3.3-70B-Instruct);
    the litellm provider prefix is added automatically. Any override must be
    recorded with the results -- it is not the published configuration.
    """
    if llm is not None:
        return llm

    # ALLOYGRAPH_LLM_TEMPERATURE overrides the caller's default. The evaluation
    # harness sets it when --seed is given: previously the harness only built an
    # explicit LLM when --llm was also passed, so a seeded run silently used the
    # 0.1 default while printing "sampling temperature forced to 0.0".
    _t_override = (os.getenv("ALLOYGRAPH_LLM_TEMPERATURE") or "").strip()
    if _t_override:
        try:
            temperature = float(_t_override)
        except ValueError:
            logger.warning("ALLOYGRAPH_LLM_TEMPERATURE=%r is not a number; ignoring.", _t_override)

    deepinfra_key = valid_api_key("DEEPINFRA_API_KEY")
    together_key = valid_api_key("TOGETHER_API_KEY")
    groq_key = valid_api_key("GROQ_API_KEY")
    openai_key = valid_api_key("OPENAI_API_KEY")

    override = (os.getenv("ALLOYGRAPH_LLM_MODEL") or "").strip()

    if deepinfra_key:
        model = override or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        logger.info("LLM provider: DeepInfra — %s (T=%.1f)", model, temperature)
        return _CacheBreakpointSafeLLM(
            model=f"deepinfra/{model}",
            api_key=deepinfra_key,
            temperature=temperature,
            num_retries=3,
        )
    elif together_key:
        model = override or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        logger.info("LLM provider: Together AI — %s (T=%.1f)", model, temperature)
        return _CacheBreakpointSafeLLM(
            model=f"together_ai/{model}",
            api_key=together_key,
            temperature=temperature,
            num_retries=3,
        )
    elif groq_key:
        model = override or "llama-3.3-70b-versatile"
        logger.info("LLM provider: Groq — %s (T=%.1f)", model, temperature)
        return _CacheBreakpointSafeLLM(
            model=f"groq/{model}",
            api_key=groq_key,
            temperature=temperature,
            num_retries=3,
        )
    elif openai_key:
        model = override or "gpt-4o-mini"
        logger.info("LLM provider: OpenAI — %s (T=%.1f)", model, temperature)
        return _CacheBreakpointSafeLLM(
            model=model,
            api_key=openai_key,
            temperature=temperature,
        )
    else:
        logger.warning(
            "LLM provider: local Ollama — llama3.1:8b (T=%.1f). No usable cloud API "
            "key was found; results will NOT be comparable to a hosted run.", temperature
        )
        return _CacheBreakpointSafeLLM(
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
        # memory=False: the docstring on get_evaluation_agents promises "No
        # memory - ensures deterministic, reproducible results", but the Designer
        # was carrying a persistent read-write LanceDB store that survives across
        # runs, so identical inputs could yield different compositions.
        "designer": create_designer_agent(design_llm, memory=False),
        "analyst": create_analyst_agent(eval_llm, memory=False),
        "reviewer": create_reviewer_agent(eval_llm, memory=False),
        "llm": eval_llm,  # For direct summary call
    }
