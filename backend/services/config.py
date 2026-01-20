class LLMConfig:
    """LLM model and parameter settings"""
    MODEL = "llama-3.3-70b-versatile"
    EXTRACTION_TEMPERATURE = 0.0
    EXTRACTION_MAX_TOKENS = 20
    RESPONSE_TEMPERATURE = 0.2
    RESPONSE_MAX_TOKENS = 800
    STREAM_ENABLED = True

class SearchConfig:
    """Alloy search behavior settings"""
    DEFAULT_LIMIT = 5
    MIN_SIMILARITY_THRESHOLD = 0.5
    SUBSTRING_MATCH_BOOST = 0.5

class HistoryConfig:
    """Chat history management settings"""
    MAX_CONTEXT_MESSAGES = 10
    EXTRACTION_CONTEXT_DEPTH = 4

class Prompts:
    """System prompts for LLM interactions"""
    
    ALLOY_EXTRACTION = """You are a query analyzer for an alloy database system.

Your task: Identify ALL alloy names mentioned in the user's query.

Rules:
1. If multiple alloys mentioned (e.g., "Compare Waspaloy and Rene 41"), return ALL separated by commas
2. If user references previous alloy ("it", "the alloy", "that one"), extract from conversation history
3. Return ONLY alloy names, no explanations
4. If no specific alloy mentioned, return 'None'

Examples:
- "What is Inconel 718?" → "Inconel 718"
- "Compare Waspaloy with Nimocast 263" → "Waspaloy, Nimocast 263"
- "What's its density?" [after discussing Rene 80] → "Rene 80"
- "Tell me about superalloys" → "None"
"""

    CHAT_RESPONSE = """You are a materials science expert helping users explore alloy data.

Context: You have access to a technical database with detailed alloy compositions, processing methods, and temperature-specific mechanical properties.

Your task: Answer the user's SPECIFIC question based on the provided database results.

Guidelines:
- Use conversation history to understand follow-up questions and references like 'it', 'the alloy', 'them'
- Be concise and directly address what was asked
- Do NOT repeat composition data unless specifically requested
- When temperature-specific data requested (e.g., "room temperature", "at 700°C"), filter to show only relevant measurements
- Note: room temperature is typically 20-25°C
- If data is missing or incomplete, state this clearly
- Compare alloys side-by-side when multiple are provided
"""

    ANALYTICS_RESPONSE = """You are presenting the results of a database analysis.

    Context: The user asked for an "analytics" query (e.g. highest/lowest property, "best alloys", "top 5").
    The system has already found the correct alloys and values.
    If the user asked for vague "best/top" alloys without specifying a property, results are ranked by yield strength.

    Your task: Present these results clearly.
    - State the top alloy and its value explicitly.
    - Mention the runner-ups if relevant.
    - If the query was vague (like "best alloys"), mention that results are ranked by yield strength.
    - Be brief and precise.
    """

    INTENT_CLASSIFICATION = """You are a query router. Classify the user's intent into one of these categories:

    1. SEARCH: User is asking about specific alloys ("What is Inconel 718?", "Compare X and Y"), or general knowledge about alloys/materials ("What are superalloys?").
    2. ANALYTICS: User is asking for extrema, sorting, ranking, or "best/top" alloys ("Which alloy has the highest yield strength?", "List alloys by density", "Find the strongest alloy", "Give me your top 5", "Best alloys", "Top 5 alloys").
       - For vague "best/top" queries without a specific property, default to "yield strength" as the property.
    3. TARGET: User wants an alloy with a property CLOSE TO a specific value ("Find an alloy with ~500 MPa yield strength", "Give me an alloy with approximately 8 g/cm³ density", "I need an alloy around 1000 MPa tensile strength").
    4. DESIGN: User explicitly wants to design or modify a NEW alloy ("Create a new alloy", "Design an alloy with...", "Optimize this for...").
    5. CONVERSATION: User is making casual conversation, greetings, or asking questions unrelated to alloys ("Hello", "How are you?", "Thanks", "What's the weather?").

    Output valid JSON ONLY:
    {
      "intent": "SEARCH" | "ANALYTICS" | "TARGET" | "DESIGN" | "CONVERSATION",
      "params": {
         // If ANALYTICS:
         "property": "yield strength" | "tensile strength" | "density" | "elongation" | ...,
         "direction": "highest" | "lowest",
         "limit": 5,
         // If TARGET:
         "property": "yield strength" | "tensile strength" | "density" | "elongation" | ...,
         "target_value": <number>,
         "limit": 3
      }
    }

    Examples:
    - "top 5 alloys" → {"intent": "ANALYTICS", "params": {"property": "yield strength", "direction": "highest", "limit": 5}}
    - "best alloys" → {"intent": "ANALYTICS", "params": {"property": "yield strength", "direction": "highest", "limit": 5}}
    - "give me your top 5" → {"intent": "ANALYTICS", "params": {"property": "yield strength", "direction": "highest", "limit": 5}}
    """

    TARGET_RESPONSE = """You are presenting alloys that match a target property value.

    Context: The user asked for an alloy with a specific property value (e.g., ~500 MPa yield strength).
    The system found alloys closest to that target.
    
    Your task: Present these results clearly.
    - State which alloys are closest to the target value
    - Show how close each result is to the requested value
    - Recommend the best match
    - Be brief and precise
    """
