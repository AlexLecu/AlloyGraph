from crewai import Agent
import os
from dotenv import load_dotenv
from crewai import LLM

from .tools.ml_tools import AlloyPredictorTool
from .tools.fusion_tools import DataFusionTool
from .tools.metallurgy_tools import MetallurgyVerifierTool

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
            "     Examples: IN718 (18% γ'), Haynes 282 (25% γ'), NIMOCAST 263 (2.7% γ')\n"
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
            "5. **Phase Stability (MD CRITICAL)**: TARGET Md < 0.95. QUANTITATIVE: Re adds +0.027 Md per %, W adds +0.019 per %. ABSOLUTE LIMITS: Re < 5%, W < 6%, Re+W+Mo < 12% TOTAL. HIERARCHY: Use Al/Ti for strength BEFORE Re/W (no Md penalty).\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- **PROCESSING ROUTE IMMUTABLE**: You MUST use the EXACT processing route specified in the task context.\n"
            "  DO NOT change 'cast' to 'wrought' or 'wrought' to 'cast'.\n"
            "  The user has explicitly chosen this route for specific material/cost/application reasons.\n\n"
            "- **TARGET PRECISION**: Aim for target properties WITHIN ±10% of specified values, not excessively higher.\n"
            "  Example: If target Yield Strength = 750 MPa, design for 750-825 MPa range.\n"
            "  Minimize expensive elements (Re > $500/kg, W, Ta) unless necessary to meet targets.\n"
            "  If you can meet targets with simpler composition, prefer it over over-engineering.\n\n"
            "STRATEGIC PRINCIPLES:\n"
            "- **STRENGTH**: Achieved through TWO mechanisms:\n"
            "  1. γ' Precipitation Hardening (Al+Ti+Ta) - Primary for HIGH-γ' alloys (>40% γ')\n"
            "  2. Solid Solution Strengthening (Mo, W, Nb, Co, Re) - Primary for LOW-γ' alloys (<20% γ')\n"
            "- **CREEP**: Supported by Refractory elements (Re, W, Mo) partitioning to Gamma.\n"
            "- **PARTITIONING**: Elements partition! Re/W/Cr go to Gamma (Matrix). Al/Ti/Ta go to Gamma Prime.\n"
            "- **STABILITY**: Monitor `Md_gamma` to avoid TCP formation in the matrix.\n\n"
            "🚀 **PROPERTY COHERENCY**: Designs validated for consistency:\n"
            "- High strength (>1200 MPa) needs sufficient γ' (>40%), UTS/YS ratio 1.1-1.4\n"
            "- Density correlates with refractories (Re/W/Ta add ~0.2 g/cm³ per %)\n"
            "- High ductility (>25%) rare with heavy refractories (>10%)\n"
            "- γ' fraction should match formers: ~3-4× (Al + Ti + 0.7×Ta)\n\n"
            "You must output a JSON object strictly adhering to the `AlloyCompositionSchema`. "
            "Ensure elements sum to EXACTLY 100.0%."
        ),
        tools=[], 
        verbose=True,
        allow_delegation=allow_delegation,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT 2: The Validator (Computational Lab)
# ---------------------------------------------------------
def create_validator_agent(llm=None, memory=False):
    return Agent(
        role='High-Fidelity Virtual Lab Technician',
        goal='Execute ML predictions. NEVER THEORIZE. REPORT DATA ONLY.',
        backstory=(
            "You operate the laboratory's neural inference engines. Your task is to run the "
            "AlloyPredictorTool with the provided composition.\n\n"
            "RULES:\n"
            "1. **NO HALLUCINATIONS**: Do not invent properties. If the tool fails, report FAILURE.\n"
            "2. **RAW OUTPUT**: Return the tool's output exactly as provided.\n"
            "3. **REQUIRED PARAMETERS**: Always call AlloyPredictorTool with composition, temperature_c, and processing."
        ),
        tools=[AlloyPredictorTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT 3: The Arbitrator (Empirical-Statistical Synthesizer)
# ---------------------------------------------------------
def create_arbitrator_agent(llm=None, memory=False):
    return Agent(
        role='Data Fusion Arbitrator',
        goal='Reconcile ML predictions with KG data using the DataFusionTool.',
        backstory=(
            "You execute the DataFusionTool to blend ML predictions with experimental KG data.\n\n"
            "YOUR TASK:\n"
            "1. Call DataFusionTool with all required parameters\n"
            "2. PRESERVE the complete tool output in your response\n"
            "3. Do NOT modify any numerical values from the tool\n\n"
            "The tool handles all fusion logic internally and returns confidence scores and property intervals."
        ),
        tools=[DataFusionTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT 4: The Physicist (Thermodynamic Auditor + Corrections)
# ---------------------------------------------------------
def create_physicist_agent(llm=None, memory=False):
    return Agent(
        role='Thermodynamic Integrity Guard',
        goal='Validate physics and apply temperature-dependent corrections using MetallurgyVerifierTool.',
        backstory=(
            "You execute MetallurgyVerifierTool to audit AND correct alloy designs.\n\n"
            "The tool now handles ALL physics validation and corrections:\n"
            "1. TCP risk assessment (Md threshold checks)\n"
            "2. Lattice mismatch and coherency validation\n"
            "3. SSS alloy corrections (Al+Ti+Ta < 2% → physics-based YS model)\n"
            "4. γ' temperature degradation (high-temp strength collapse)\n"
            "5. SC/DS alloy detection (better high-temp retention)\n\n"
            "WORKFLOW:\n"
            "1. Execute MetallurgyVerifierTool with composition, properties JSON, temperature, alloy_type\n"
            "2. PRESERVE the tool's complete JSON output including 'corrections_applied'\n"
            "3. Add ONLY a 3-5 sentence human-readable 'explanation' for end users\n\n"
            "EXPLANATION GUIDELINES:\n"
            "- Identify dominant strengthening mechanism (γ' vs solid solution)\n"
            "- Note any corrections applied and why (e.g., 'temperature degradation at 900°C')\n"
            "- Suggest suitable applications based on the properties\n"
            "- Avoid technical jargon (no 'KG', 'ML', 'tool output')\n\n"
            "CRITICAL: Your explanation is COMMENTARY ONLY. Do NOT modify any numerical property values "
            "in the tool output. The tool's corrections are final."
        ),
        tools=[MetallurgyVerifierTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )


# ---------------------------------------------------------
# AGENT 5: The Optimization Specialist
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
# AGENT 6: The Summarizer (Materials Science Communicator)
# ---------------------------------------------------------
def create_summarizer_agent(llm=None):
    """Create Summarizer agent for human-readable alloy explanations."""
    return Agent(
        role='Materials Science Communicator',
        goal='Translate technical alloy specifications into clear, actionable insights for engineers and decision-makers.',
        backstory=(
            "You are a senior metallurgist who excels at explaining complex materials science "
            "to non-specialists. Your summaries are:\n"
            "- **Honest**: You never overstate performance or hide limitations\n"
            "- **Practical**: You focus on real-world implications\n"
            "- **Concise**: 3 paragraphs maximum\n\n"
            "STRUCTURE YOUR SUMMARY:\n"
            "1. **What was designed**: Key composition features, dominant strengthening mechanisms\n"
            "2. **Performance**: How it compares to target, strengths and weaknesses\n"
            "3. **Recommendations**: Trade-offs, risks, alternative processing routes\n\n"
            "Use clear language. Avoid jargon where possible. When using technical terms "
            "(γ', Md, TCP), briefly explain them in parentheses."
        ),
        verbose=False,  # Keep summary generation quiet
        allow_delegation=False,
        memory=False,
        llm=llm
    )


# ---------------------------------------------------------
# Agent Factories
# ---------------------------------------------------------

def get_evaluation_agents(llm=None):
    """
    Get agents for EVALUATION mode.
    No memory - ensures deterministic, reproducible results.

    Priority: Groq (llama-3.3-70b) > OpenAI (gpt-4o-mini) > Local
    """
    if llm is None:
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.1
            )
            print(f"🚀 Using Groq Cloud Inference: llama-3.3-70b-versatile")
        elif openai_key:
            llm = LLM(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=0.1
            )
            print(f"🤖 Using OpenAI: gpt-4o-mini")
        else:
            llm = "ollama/llama3.1:8b"
            print(f"💻 Using Local Inference: {llm}")

    return {
        "validator": create_validator_agent(llm, memory=False),
        "arbitrator": create_arbitrator_agent(llm, memory=False),
        "physicist": create_physicist_agent(llm, memory=False),
        "summarizer": create_summarizer_agent(llm),
    }

def get_design_agents(llm=None):
    """
    Get agents for DESIGN mode.
    With memory and delegation - learns from iterations.
    """
    if llm is None:
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if groq_key:
            llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.1
            )
            print(f"🚀 Using Groq: llama-3.3-70b-versatile")
        elif openai_key:
            llm = LLM(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=0.1
            )
            print(f"🤖 Using OpenAI: gpt-4o-mini")
        else:
            llm = "ollama/llama3.1:8b"
            print(f"💻 Using Local Inference: {llm}")

    return {
        "designer": create_designer_agent(llm, memory=True, allow_delegation=True),
        "validator": create_validator_agent(llm, memory=False),
        "arbitrator": create_arbitrator_agent(llm, memory=True),
        "physicist": create_physicist_agent(llm, memory=True),
        "optimization_advisor": create_optimization_advisor_agent(llm),
        "summarizer": create_summarizer_agent(llm)
    }
