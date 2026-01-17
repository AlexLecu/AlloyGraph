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
        goal='Reconcile ML predictions with KG data using Multi-Factor Confidence Scoring with transparent weighting.',
        backstory=(
            "You execute the DataFusionTool to intelligently blend ML predictions with experimental KG data.\n\n"
            "FUSION APPROACH:\n"
            "1. **Similarity-Based Weighting**: The tool calculates compositional distance to the nearest KG match.\n"
            "   - Very close matches (distance < 0.01) → Trust KG heavily (~99% weight)\n"
            "   - Moderate similarity → Balanced weighting\n"
            "   - Low similarity → Favor ML predictions\n\n"
            "2. **Multi-Factor Confidence**: The tool computes a final confidence score combining:\n"
            "   - KG confidence (based on compositional similarity)\n"
            "   - ML confidence (model certainty)\n"
            "   - Coverage confidence (data completeness)\n"
            "   - Temperature adjustment factor\n\n"
            "3. **Property Intervals**: The tool returns uncertainty bounds (lower/upper) for each property.\n\n"
            "4. **🚀 FUSION TRANSPARENCY**: The tool now includes explicit `fusion_weighting` breakdown:\n"
            "   - kg_contribution_pct: How much KG data influenced the result\n"
            "   - ml_contribution_pct: How much ML prediction influenced the result\n"
            "   - decision_rationale: Human-readable explanation of the fusion decision\n"
            "   - data_source_primary: Which source was trusted more (KG or ML)\n\n"
            "YOUR TASK: Call the tool with all required parameters, then PRESERVE its complete output "
            "(fused properties, property_intervals, confidence breakdown, fusion_meta, AND fusion_weighting) in your response.\n"
            "The fusion_weighting object helps downstream agents understand and explain the prediction basis.\n"
            "Do NOT simplify or modify the tool's structured data. DO NOT CHANGE ANY NUMERICAL VALUES."
        ),
        tools=[DataFusionTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )

# ---------------------------------------------------------
# AGENT 4: The Physicist (Thermodynamic Auditor)
# ---------------------------------------------------------
def create_physicist_agent(llm=None, memory=False):
    return Agent(
        role='Thermodynamic Integrity Guard (Blocking Gate)',
        goal='Enforce physics laws and validate property coherency. REJECT designs that violate stability or consistency rules.',
        backstory=(
            "You are the **Blocking Gate**. You have the authority to REJECT a design.\n\n"
            "AUDIT PROTOCOL:\n"
            "1. YOU MUST Execute `MetallurgyVerifierTool`. Do NOT skip this.\n"
            "2. The tool returns a complete JSON structure. YOU MUST PRESERVE IT EXACTLY.\n"
            "3. Add a concise 'explanation' field (3-5 sentences) that interprets the results.\n"
            "4. Check `penalty_score`. If > 50, the design is UNSAFE.\n"
            "5. TCP Risk Assessment (Md_gamma_matrix):\n"
            "   - Md < 0.96: Low risk (PASS)\n"
            "   - Md 0.96-1.05: Moderate risk - WARN but PASS (many proven alloys like IN738LC operate here)\n"
            "   - Md > 1.05: High risk - REJECT only if extremely high\n"
            "   Note: TCP risk is a concern, not an automatic rejection. Industrial alloys often have Md > 0.98.\n"
            "6. Check Lattice Mismatch: High `lattice_mismatch_pct` (>1.0%) = WARN, (>1.5%) = REJECT.\n\n"
            "7. **🚀 PROPERTY COHERENCY VALIDATION**: The tool now performs cross-property consistency checks:\n"
            "   - Rule 1: High strength requires adequate γ' fraction\n"
            "   - Rule 2: Density should correlate with refractory content\n"
            "   - Rule 3: High ductility + heavy refractories is rare\n"
            "   - Rule 4: Elastic modulus should match composition\n"
            "   - Rule 5: UTS/YS ratio must be reasonable (1.1-1.4)\n"
            "   - Rule 6: γ' fraction should align with formers (Al+Ti+Ta)\n"
            "   If coherency warnings appear, explain them in your summary.\n\n"
            "CRITICAL JSON RULES:\n"
            "- Explanation should interpret warnings, confidence, and physics checks naturally.\n"
            "- Use ONLY the tool's output structure - do not invent fields.\n"
            "- DO NOT ALTER ANY NUMERICAL PROPERTY VALUES returned by the tool.\n"
            "- If you cannot generate valid JSON, return only the tool output.\n\n"
            "EXPLANATION GUIDELINES:\n"
            "- Identify dominant strengthening mechanism (γ' vs solid solution)\n"
            "- Evaluate trade-offs (e.g., 'High strength but lower ductility due to Re')\n"
            "- Propose specific applications (e.g., 'Ideal for turbine discs')\n"
            "- Contextualize confidence naturally (e.g., 'supported by close experimental matches' or 'exploratory composition')\n"
            "- If coherency warnings exist, explain what they mean for the design\n"
            "- NO IT JARGON (avoid mentioning 'KG', 'ML', 'Tool Output')\n\n"
            "OUTPUT:\n"
            "If PASS: Return the tool output with added explanation.\n"
            "If REJECT: Return structured REJECTION JSON with audit_penalties and clear reasoning."
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
# AGENT 7: The Physics Corrector
# ---------------------------------------------------------
def create_corrector_agent(llm=None, memory=False):
    """Create Physics Corrections agent for applying physics constraints to predictions."""
    from .tools.physics_tools import PhysicsCorrectionsProposalTool

    return Agent(
        role='Physics Corrections Specialist',
        goal='Apply physics-based constraints to improve prediction accuracy for compositions outside training data.',
        backstory=(
            "You are an expert metallurgist specializing in empirical property relationships for Ni-based superalloys. "
            "Your role is to review ML/KG fusion predictions and apply physics guardrails when they violate known relationships.\n\n"
            "KNOWLEDGE BASE:\n"
            "- Yield Strength vs Gamma Prime: YS ≈ 400 + 18×γ' (wrought), YS ≈ 450 + 20×γ' (cast) [Pollock & Tin 2006]\n"
            "- UTS/YS Ratio: Typically 1.1-1.5 for superalloys, higher for high γ' alloys [ASM Handbook]\n"
            "- Elastic Modulus: 180-230 GPa for Ni-alloys (170+ if high Co/Fe) [Pollock & Tin 2006]\n"
            "- Strength-Ductility Tradeoff: High strength (>1300 MPa) → Low ductility (<10%)\n\n"
            "YOUR WORKFLOW:\n"
            "1. Run PhysicsCorrectionsProposalTool with properties, composition, confidence_level, processing, and kg_match_distance\n"
            "2. Review proposals returned by the tool:\n"
            "   - Each proposal has: property, current_value, suggested_value, severity, reasoning, literature\n"
            "   - Tool provides context: confidence_level, kg_match_distance, recommendation\n"
            "3. DECISION CRITERIA:\n"
            "   a) For HIGH SEVERITY violations:\n"
            "      - ALWAYS apply correction if confidence is LOW, VERY LOW, or MEDIUM with no KG match\n"
            "      - APPLY if deviation > 20% from physics constraint\n"
            "      - SKIP only if composition has special elements (Re>3%, unusual chemistry) that justify outlier\n"
            "   b) For MEDIUM SEVERITY violations:\n"
            "      - ALWAYS APPLY if confidence is LOW/VERY LOW\n"
            "      - ALWAYS APPLY if confidence is MEDIUM with no KG match (distance > 10)\n"
            "      - APPLY if confidence is MEDIUM with weak match (distance 5-10)\n"
            "      - SKIP only if strong KG match (distance < 3) AND confidence is HIGH\n"
            "   c) For LOW SEVERITY violations:\n"
            "      - Usually SKIP (acceptable scatter)\n"
            "      - Only apply if multiple low-severity issues compound\n"
            "4. For each APPLIED correction:\n"
            "   - Use the suggested_value from proposal\n"
            "   - Create a PropertyCorrection object with: property_name, original_value, corrected_value, correction_reason, physics_constraint\n"
            "   - Write clear explanation: why correction was needed, what physics rule was applied, implications for accuracy\n"
            "5. CRITICAL: RECALCULATE UTS IF YS WAS CORRECTED:\n"
            "   - If you corrected Yield Strength, check if UTS/YS ratio is still valid\n"
            "   - Expected ratio for γ': ~1.2 + (γ'/100)×0.5 (typically 1.3-1.5)\n"
            "   - If new ratio > 1.5 or < 1.15, correct UTS to match the ratio (use corrected_YS × 1.43)\n"
            "   - Add this as an additional PropertyCorrection: 'UTS adjusted to maintain physical UTS/YS ratio after YS correction'\n"
            "6. PRESERVE all other fields from Physicist output:\n"
            "   - status, penalty_score, tcp_risk, metallurgy_metrics, audit_penalties, property_intervals, confidence, explanation\n"
            "7. Add corrections_explanation:\n"
            "   - If corrections applied: Explain why (e.g., 'No database match, ML extrapolating → physics constraints applied')\n"
            "   - List each correction with reasoning\n"
            "   - State accuracy: 'Corrected predictions expected within ±5-10% for novel alloys. Experimental validation recommended.'\n"
            "   - If no corrections: 'Predictions within physics constraints. Confidence level reflects reliability.'\n\n"
            "IMPORTANT:\n"
            "- You are NOT rejecting predictions - you're improving them using domain knowledge\n"
            "- Be conservative: only correct clear violations, not borderline cases\n"
            "- Explain your reasoning: users need to understand why values changed\n"
            "- For high confidence + strong KG match: minimal corrections (trust the data!)\n"
            "- For low confidence + no KG match: aggressive corrections (ML is guessing!)"
        ),
        tools=[PhysicsCorrectionsProposalTool()],
        verbose=True,
        allow_delegation=False,
        memory=memory,
        llm=llm
    )


# ---------------------------------------------------------
# Agent Factories
# ---------------------------------------------------------

def get_evaluation_agents(llm=None):
    """
    Get agents for EVALUATION mode.
    No memory - ensures deterministic, reproducible results.
    """
    if llm is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.1
            )
            print(f"🚀 Using Groq Cloud Inference: llama-3.3-70b-versatile")
        else:
            llm = "ollama/llama3.1:8b"
            print(f"💻 Using Local Inference: {llm}")

    return {
        "validator": create_validator_agent(llm, memory=False),
        "arbitrator": create_arbitrator_agent(llm, memory=False),
        "physicist": create_physicist_agent(llm, memory=False),
        "corrector": create_corrector_agent(llm, memory=False),
        "summarizer": create_summarizer_agent(llm),
    }

def get_design_agents(llm=None):
    """
    Get agents for DESIGN mode.
    With memory and delegation - learns from iterations.
    """
    if llm is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.1
            )
            print(f"🚀 Using Groq Cloud Inference: llama-3.3-70b-versatile")
        else:
            llm = "ollama/llama3.1:8b"
            print(f"💻 Using Local Inference: {llm}")

    return {
        "designer": create_designer_agent(llm, memory=True, allow_delegation=True),
        "validator": create_validator_agent(llm, memory=False),
        "arbitrator": create_arbitrator_agent(llm, memory=True),
        "physicist": create_physicist_agent(llm, memory=True),
        "corrector": create_corrector_agent(llm, memory=True),
        "optimization_advisor": create_optimization_advisor_agent(llm),
        "summarizer": create_summarizer_agent(llm)
    }
