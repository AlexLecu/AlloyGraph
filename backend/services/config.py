import os
from groq import Groq


class LLMConfig:
    """LLM model and parameter settings"""
    MODEL = "llama-3.3-70b-versatile"
    ROUTING_TEMPERATURE = 0.0
    ROUTING_MAX_TOKENS = 250
    RESPONSE_TEMPERATURE = 0.2
    RESPONSE_MAX_TOKENS = 800
    STREAM_ENABLED = True

    _client = None

    @classmethod
    def get_client(cls) -> Groq | None:
        """Return a singleton Groq client, or None if no API key."""
        if cls._client is None:
            key = os.getenv("GROQ_API_KEY")
            if not key:
                return None
            cls._client = Groq(api_key=key)
        return cls._client


class SearchConfig:
    """Alloy search behavior settings"""
    DEFAULT_LIMIT = 5
    MIN_SIMILARITY_THRESHOLD = 0.5
    SUBSTRING_MATCH_BOOST = 0.5

class HistoryConfig:
    """Chat history management settings"""
    MAX_CONTEXT_MESSAGES = 10
    EXTRACTION_CONTEXT_DEPTH = 4


# ── Property term mapping (used for analytics / target queries) ──────────
PROPERTY_SEARCH_TERMS = [
    # Multi-word (must precede single-word entries)
    ("tensile strength", "tensile"),
    ("yield strength", "yield"),
    ("elastic modulus", "elastic modulus"),
    ("young's modulus", "elastic modulus"),
    # Single-word
    ("tensile", "tensile"),
    ("uts", "tensile"),
    ("ultimate", "tensile"),
    ("yield", "yield"),
    ("elongation", "elongation"),
    ("ductility", "elongation"),
    ("density", "density"),
    ("elastic", "elastic modulus"),
    ("modulus", "elastic modulus"),
    ("young", "elastic modulus"),
    ("creep", "creep"),
    ("rupture", "creep"),
    ("hardness", "hardness"),
    ("strength", "yield"),  # generic fallback — must be LAST
]


class Prompts:
    """System prompts for LLM interactions"""

    # ── Single call: intent + alloy extraction ───────────────────────────
    ROUTE_AND_EXTRACT = """You are a query router for an alloy knowledge-graph chatbot.

Given the user query and recent conversation history, return a JSON object with:
1. **intent** – one of SEARCH, ANALYTICS, TARGET, DESIGN, CONVERSATION.
2. **alloys** – a list of alloy names mentioned or implied (resolve "it"/"the alloy" from history). Empty list if none.
3. **params** – extra parameters depending on intent (see below).

Intent definitions (choose the FIRST that fits):
- ANALYTICS: User wants a RANKED LIST — highest/lowest/best/top/strongest/lightest, or a comparison by a measurable property. Requires sorting. Params: property, direction, limit.
  Default property = "yield strength" when unspecified.
- TARGET: User wants an alloy with a property CLOSE TO a specific value. This includes:
  (a) Explicit numeric targets: "~500 MPa", "around 8 g/cm³" → target_value = the number.
  (b) "Similar to [alloy]" queries: "similar alloy in terms of yield strength" → target_value = null, alloy name in "alloys".
  Params: property, target_value (number or null), limit.
- DESIGN: User explicitly wants to design/create/optimize a NEW alloy composition. Params: none.
- CONVERSATION: Greetings, thanks, off-topic, or meta-questions unrelated to alloys. Params: none.
- SEARCH: Everything else — asking about specific alloys, their properties, composition, comparisons, or general alloy knowledge. Params: none.

Output valid JSON ONLY:
{
  "intent": "SEARCH" | "ANALYTICS" | "TARGET" | "DESIGN" | "CONVERSATION",
  "alloys": ["Alloy Name", ...],
  "params": {}
}

Params per intent:
- ANALYTICS: {"property": "yield strength", "direction": "highest", "limit": 5}
- TARGET: {"property": "yield strength", "target_value": 500, "limit": 3}  (target_value is null when derived from a referenced alloy)
- All others: {}

Rules for the "alloys" field:
- If multiple alloys are mentioned ("Compare Waspaloy and Rene 41"), list ALL of them.
- If the user refers to a previous alloy ("it", "the alloy", "that one"), resolve from history.
- If no specific alloy is relevant, return an empty list [].

Examples:
- "What is Inconel 718?" → {"intent":"SEARCH","alloys":["Inconel 718"],"params":{}}
- "Compare Waspaloy with Nimocast 263" → {"intent":"SEARCH","alloys":["Waspaloy","Nimocast 263"],"params":{}}
- "What's its density?" [after Rene 80] → {"intent":"SEARCH","alloys":["Rene 80"],"params":{}}
- "Which alloy has the highest yield strength?" → {"intent":"ANALYTICS","alloys":[],"params":{"property":"yield strength","direction":"highest","limit":5}}
- "Top 5 alloys" → {"intent":"ANALYTICS","alloys":[],"params":{"property":"yield strength","direction":"highest","limit":5}}
- "Lightest superalloy?" → {"intent":"ANALYTICS","alloys":[],"params":{"property":"density","direction":"lowest","limit":5}}
- "Find an alloy with ~500 MPa yield strength" → {"intent":"TARGET","alloys":[],"params":{"property":"yield strength","target_value":500,"limit":3}}
- "Give me a similar alloy in terms of yield strength" [after Haynes 230] → {"intent":"TARGET","alloys":["Haynes 230"],"params":{"property":"yield strength","target_value":null,"limit":5}}
- "Any alloy with similar density?" [after Inconel 718] → {"intent":"TARGET","alloys":["Inconel 718"],"params":{"property":"density","target_value":null,"limit":5}}
- "Hello!" → {"intent":"CONVERSATION","alloys":[],"params":{}}
"""

    CHAT_RESPONSE = """You are a materials science expert helping users explore alloy data from a knowledge graph.

Your task: Answer the user's SPECIFIC question using the database results provided in "Context".

Guidelines:
- Base your answer primarily on the provided database Context
- You may supplement with general metallurgy knowledge when the database lacks information (e.g., typical applications, general alloy families), but clearly distinguish database facts from general knowledge
- Use conversation history to understand follow-up questions and references like "it", "the alloy", "them"
- Be concise — directly address what was asked, don't pad
- Always include units (MPa, g/cm³, %, °C, etc.)
- Do NOT repeat full composition tables unless the user specifically asked about composition
- When temperature-specific data is requested, show only the relevant temperature range. Room temperature = 20–25°C
- When comparing multiple alloys, use a structured side-by-side format
- If data is missing or incomplete, say so explicitly
- Use **bold** for alloy names and key values
"""

    ANALYTICS_RESPONSE = """You are presenting ranked alloy data from a knowledge graph.

The system already sorted the alloys for the user's query. Your job is to present the results clearly.

Guidelines:
- Lead with the #1 result: state the alloy name and its value WITH UNITS (e.g., "**Waspaloy**: 795 MPa yield strength at 25°C")
- List runner-ups with their values
- If the query was vague (like "best alloys" without a specific property), mention that results are ranked by yield strength
- Keep it concise — a numbered list is ideal
- Always include units (MPa, g/cm³, %, °C)
"""

    TARGET_RESPONSE = """You are presenting alloys that match a target property value from a knowledge graph.

The system found alloys closest to the user's requested value. Your job is to present the results.

Guidelines:
- For each result, state the alloy name and its actual value WITH UNITS
- Show the difference from the target (e.g., "target: 500 MPa → **Alloy X**: 485 MPa, 3% below target")
- Recommend the closest match
- Keep it concise — a numbered list with the delta is ideal
- Always include units (MPa, g/cm³, %, °C)
"""
