from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import traceback


from alloy_crew.alloy_evaluator import AlloyEvaluationCrew
from alloy_crew.alloy_designer import IterativeDesignCrew
from services.chat_service import stream_chat_response
from services import job_store

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create the schema and set WAL once per worker start, so the first request does
# not race 3 other workers all initialising the same file.
job_store.init_db()


@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "ok", "message": "Backend is running"}, 200


def _run_validation(composition, processing, temp, llm):
    """The actual evaluation. Runs on a background thread, not in the request.

    Raises on failure; job_store records the exception as the job's error.
    """
    logger.info(f"🔹 Validating: {composition} @ {temp}°C ({processing})")

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
        "analyst_reasoning": result.get("analyst_reasoning", ""),
        "reviewer_assessment": result.get("reviewer_assessment", ""),
        "investigation_findings": result.get("investigation_findings", ""),
        "source_reliability": result.get("source_reliability", ""),
    }

    # Include error field if present. This is an in-band failure report from the
    # pipeline, not a raised exception, so the job still counts as done and the
    # frontend surfaces result.error exactly as it did before.
    if "error" in result:
        sanitized_result["error"] = result["error"]

    return sanitized_result


@app.route('/api/validate', methods=['POST'])
def validate_alloy():
    """Start an evaluation job and return its id immediately.

    This used to run the crew inline and reply with the result, which held the
    connection open for 100-200 s. The CloudUT reverse proxy in front of the
    public deployment cuts off around 60 s and returned 504, and that hop is not
    ours to configure. So the contract is now submit-then-poll: this returns 202
    in well under a second, and GET /api/validate/status/<job_id> reports the
    outcome. The only HTTP caller is the frontend; the evaluation scripts import
    AlloyEvaluationCrew directly and are unaffected.
    """
    data = request.json or {}
    composition = data.get('composition')
    temp = data.get('temp', 20)
    processing = data.get('processing', 'cast')
    llm = data.get('llm')

    if not composition:
        return jsonify({"error": "No composition provided"}), 400

    # Opportunistic housekeeping: no scheduler exists in a sync worker, and
    # submission is exactly when the table grows.
    try:
        job_store.cleanup()
    except Exception as e:  # noqa: BLE001 - never fail a submission over this
        logger.warning("Job store cleanup failed (continuing): %s", e)

    job_id = job_store.create_job(
        "validate",
        {"composition": composition, "temp": temp, "processing": processing},
    )
    job_store.run_in_background(
        job_id, _run_validation, composition, processing, temp, llm
    )
    logger.info("Job %s queued: validate %s @ %s°C (%s)",
                job_id, composition, temp, processing)

    return jsonify({"job_id": job_id, "status": job_store.STATUS_PENDING}), 202


@app.route('/api/validate/status/<job_id>', methods=['GET'])
def validate_status(job_id):
    """Report a job's state, with the result when done or the error when failed.

    404 for an unknown id. That covers a typo, an id from before a container
    restart (the store is deliberately non-durable), and one whose finished row
    has passed its TTL.
    """
    try:
        job = job_store.get_job(job_id)
    except Exception as e:  # noqa: BLE001
        logger.error("Job store read failed for %s: %s", job_id, e)
        return jsonify({"error": "Could not read job state."}), 500

    if job is None:
        return jsonify({
            "error": "Unknown job id. It may have expired or the server "
                     "restarted; please run the analysis again.",
            "status": "not_found",
        }), 404

    payload = {"job_id": job["id"], "status": job["status"]}

    if job["status"] == job_store.STATUS_DONE:
        payload["result"] = job["result"]
    elif job["status"] == job_store.STATUS_FAILED:
        payload["error"] = job["error"] or "The analysis failed."

    return jsonify(payload), 200

@app.route('/api/design', methods=['POST'])
def design():
    """Design alloy based on target properties"""
    data = request.json
    
    # Extract target_props as a dict
    target_props = data.get('target_props', {})
    target_props = {k: v for k, v in target_props.items() if v and float(v) > 0}

    # Fallback to individual params for backward compatibility
    if not target_props:
        target_props = {'Yield Strength': data.get('yield_strength', 1000)}
        if data.get('tensile_strength') is not None:
            target_props['Tensile Strength'] = data['tensile_strength']
        if data.get('elongation') is not None:
            target_props['Elongation'] = data['elongation']
        if data.get('elastic_modulus') is not None:
            target_props['Elastic Modulus'] = data['elastic_modulus']
        if data.get('density') and float(data['density']) > 0:
            target_props['Density'] = data['density']
        if data.get('gamma_prime') and float(data['gamma_prime']) > 0:
            target_props['Gamma Prime'] = data['gamma_prime']
    
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
            "status": result.get("status", "UNKNOWN"),
            "issues": result.get("issues", []),
            "recommendations": result.get("recommendations", []),
            "explanation": result.get("explanation", ""),
            "audit_penalties": result.get("audit_penalties", []),
            "metallurgy_metrics": result.get("metallurgy_metrics", {}),
            "penalty_score": result.get("penalty_score", 0.0),
            "corrections_applied": result.get("corrections_applied", []),
            "corrections_explanation": result.get("corrections_explanation", ""),
            "analyst_reasoning": result.get("analyst_reasoning", ""),
            "reviewer_assessment": result.get("reviewer_assessment", ""),
            "investigation_findings": result.get("investigation_findings", ""),
            "source_reliability": result.get("source_reliability", ""),
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
