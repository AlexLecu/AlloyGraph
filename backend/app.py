from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import traceback


from alloy_crew.alloy_evaluator import AlloyEvaluationCrew
from alloy_crew.alloy_designer import IterativeDesignCrew
from services.config import LLMConfig
from services.chat_service import stream_chat_response

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "ok", "message": "Backend is running"}, 200


@app.route('/api/validate', methods=['POST'])
def validate_alloy():
    """Run the Validator Agent on a composition."""
    data = request.json
    composition = data.get('composition')
    temp = data.get('temp', 20)
    processing = data.get('processing', 'cast')
    llm = data.get('llm', f"groq/{LLMConfig.MODEL}")

    if not composition:
        return jsonify({"error": "No composition provided"}), 400

    logger.info(f"🔹 Validating: {composition} @ {temp}°C ({processing})")

    try:
        crew = AlloyEvaluationCrew(llm_config=llm)
        result = crew.run(composition=composition, processing=processing, temperature=temp)

        # Sanitize response - include ALL fields to match design mode
        sanitized_result = {
            "composition": result.get("composition", composition),
            "properties": result.get("properties", {}),
            "property_intervals": result.get("property_intervals", {}),
            "tcp_risk": result.get("tcp_risk", "Unknown"),
            "confidence": result.get("confidence", {}),
            "status": result.get("status", "UNKNOWN"),
            "explanation": result.get("explanation", ""),
            "audit_penalties": result.get("audit_penalties", []),
            "metallurgy_metrics": result.get("metallurgy_metrics", {}),
            "penalty_score": result.get("penalty_score", 0.0),
            "corrections_applied": result.get("corrections_applied", []),
            "corrections_explanation": result.get("corrections_explanation", ""),
        }

        # Include error field if present
        if "error" in result:
            sanitized_result["error"] = result["error"]

        return jsonify({"result": sanitized_result})
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/design', methods=['POST'])
def design():
    """Design alloy based on target properties"""
    data = request.json
    
    # Extract target_props as a dict
    target_props = data.get('target_props', {})
    
    # Fallback to individual params for backward compatibility
    if not target_props:
        target_props = {'Yield Strength': data.get('yield_strength', 1000)}
        if data.get('tensile_strength', 0) > 0:
            target_props['Tensile Strength'] = data.get('tensile_strength')
        if data.get('elongation', 0) > 0:
            target_props['Elongation'] = data.get('elongation')
        if data.get('elastic_modulus', 0) > 0:
            target_props['Elastic Modulus'] = data.get('elastic_modulus')
        if data.get('density', 99) < 99:
            target_props['Density'] = data.get('density')
        if data.get('gamma_prime', 0) > 0:
            target_props['Gamma Prime'] = data.get('gamma_prime')
    
    processing = data.get('processing', 'cast')
    temperature = data.get('temp', 900)
    max_iter = data.get('max_iter', 3)

    logger.info(f"🎨 DESIGN REQUEST: Targets={target_props}, Processing={processing}, Temp={temperature}°C")

    try:
        crew = IterativeDesignCrew(target_props)
        result = crew.loop(max_iterations=max_iter, processing=processing, temperature=temperature)

        # Validate composition if present
        composition_status = "UNKNOWN"
        if result.get("composition"):
            try:
                validation = AlloyEvaluationCrew.validate_composition(result.get("composition"))
                if validation.get("warnings"):
                    composition_status = "WARNING"
                else:
                    composition_status = "VALID"
            except Exception as e:
                composition_status = "INVALID"
                result["composition_validation_error"] = str(e)

        # Sanitize response - include ALL fields to match evaluate mode
        sanitized_result = {
            "composition": result.get("composition", {}),
            "properties": result.get("properties", {}),
            "property_intervals": result.get("property_intervals", {}),
            "tcp_risk": result.get("tcp_risk", "Unknown"),
            "confidence": result.get("confidence", {}),
            "design_status": result.get("design_status", "success"),
            "composition_status": composition_status,
            "status": result.get("status", "UNKNOWN"),  # ✅ Add PASS/REJECT/FAIL status
            "issues": result.get("issues", []),
            "recommendations": result.get("recommendations", []),
            "explanation": result.get("explanation", ""),
            "audit_penalties": result.get("audit_penalties", []),
            "metallurgy_metrics": result.get("metallurgy_metrics", {}),
            "penalty_score": result.get("penalty_score", 0.0),
            "corrections_applied": result.get("corrections_applied", []),
            "corrections_explanation": result.get("corrections_explanation", ""),
        }

        # Include error field if present (for backwards compatibility)
        if "error" in result:
            sanitized_result["error"] = result["error"]

        # Include composition validation error if present
        if "composition_validation_error" in result:
            sanitized_result["composition_validation_error"] = result["composition_validation_error"]

        # Optionally include reasoning for debugging (but keep it short)
        if "reasoning" in result and len(str(result["reasoning"])) < 500:
            sanitized_result["reasoning"] = result["reasoning"]

        return jsonify({"result": sanitized_result})
    except Exception as e:
        logger.error(f"Design failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat_kg_stream():
    """Stream chat response with alloy data first."""
    data = request.json
    prompt = data.get('prompt')
    session_id = data.get('sessionId', 'default')
    history = data.get('history', [])

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    logger.info(f"🔹 Chat [{session_id}]: {prompt}")

    response = Response(
        stream_with_context(stream_chat_response(prompt, session_id, history)),
        mimetype='application/x-ndjson'
    )
    response.headers['X-Accel-Buffering'] = 'no'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
