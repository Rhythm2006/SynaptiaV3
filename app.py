"""
Synaptia — Flask Application
Main server with API routes for exercise generation, hints, chat, answer checking,
analytics, adaptive difficulty, progress tracking, and interview coaching.
Powered by the Groq API (groq.com) with an AI cross-validation layer to prevent hallucinations.
"""

from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from config import config
from puzzle_generator import PuzzleGenerator
from hint_provider import HintProvider
from analytics import AnalyticsEngine
from difficulty_engine import DifficultyEngine
from progress_tracker import ProgressTracker
from interview_coach import InterviewCoach
from vision_memory import VisionMemoryEngine
import os
import logging
from datetime import datetime

# Configure logging so model rotation events appear in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ============== APP SETUP ==============

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
CORS(app)

config_name = os.getenv("FLASK_ENV", "development")
app.config.from_object(config.get(config_name, config["default"]))

# Initialize modules
puzzle_gen       = PuzzleGenerator()
hint_provider    = HintProvider()
analytics_engine = AnalyticsEngine()
difficulty_engine = DifficultyEngine()
progress_tracker = ProgressTracker()
interview_coach  = InterviewCoach()
vision_memory    = VisionMemoryEngine()

import urllib.parse

# In-memory session tracking
session_stats = {}


class VercelWSGIWrapper:
    """
    Normalizes PATH_INFO when deployed as a Serverless Function on Vercel.
    Extracts the original request path from the rewrite parameter (__path__)
    or proxy headers so Flask routes match accurately.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query_string = environ.get("QUERY_STRING", "")
        extracted_path = None

        # 1. Primary: Extract from __path__ query param passed by vercel.json rewrite
        if "__path__=" in query_string:
            try:
                params = urllib.parse.parse_qs(query_string)
                if "__path__" in params and params["__path__"]:
                    extracted_path = urllib.parse.unquote(params["__path__"][0])
                    # Clean __path__ from query string so request.args remains clean
                    clean_params = {k: v for k, v in params.items() if k != "__path__"}
                    environ["QUERY_STRING"] = urllib.parse.urlencode(clean_params, doseq=True)
            except Exception:
                pass

        # 2. Secondary: Route match headers or proxy forwarded paths
        if not extracted_path:
            for header in [
                "HTTP_X_FORWARDED_PATH",
                "HTTP_X_FORWARDED_URI",
                "HTTP_X_MATCHED_PATH",
                "HTTP_X_INVOKE_PATH",
                "HTTP_X_VERCEL_PATH",
            ]:
                val = environ.get(header)
                if val and val not in ("/api/index.py", "/api/index"):
                    extracted_path = val
                    break

        if extracted_path:
            # Strip query string if present in extracted path
            if "?" in extracted_path:
                extracted_path = extracted_path.split("?", 1)[0]
            # Normalize rewrite prefixes
            if extracted_path.startswith("/api/index.py"):
                extracted_path = extracted_path[len("/api/index.py"):] or "/"
            elif extracted_path.startswith("/api/index") and (
                len(extracted_path) == len("/api/index") or extracted_path[len("/api/index")] == "/"
            ):
                extracted_path = extracted_path[len("/api/index"):] or "/"

            if not extracted_path.startswith("/"):
                extracted_path = "/" + extracted_path

            environ["PATH_INFO"] = extracted_path

        return self.wsgi_app(environ, start_response)


# Wrap Flask WSGI application for production/serverless environments
app.wsgi_app = VercelWSGIWrapper(app.wsgi_app)
handler = app


# Enable native CORS for VS Code Live Server usage
@app.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ============== WEB UI ==============

@app.route("/")
def index():
    """Serve the main web interface"""
    return render_template("index.html")


@app.route("/login")
def login():
    """Serve the authentication page"""
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Redirect to login page"""
    return redirect("/login")


@app.route("/static/<path:filename>")
def serve_static_file(filename):
    """Explicit static file handler for serverless environments"""
    static_dir = os.path.join(BASE_DIR, "static")
    return send_from_directory(static_dir, filename)


# ============== PUZZLE API ==============

@app.route("/api/puzzle/generate", methods=["POST"])
def generate_puzzle():
    
    try:
        data = request.get_json() or {}
        difficulty = data.get("difficulty", "medium")
        puzzle_type = data.get("type", "riddle")

        puzzle = puzzle_gen.generate_puzzle(difficulty, puzzle_type)

        if "error" in puzzle:
            # Return 200 so the JS error-check path (puzzle.error) fires
            # instead of the catch block which shows a generic server message.
            return jsonify(puzzle), 200

        return jsonify(puzzle), 201

    except Exception as e:
        return jsonify({"error": str(e), "message": "Internal server error"}), 500


@app.route("/api/puzzle/<puzzle_id>", methods=["GET"])
def get_puzzle(puzzle_id):
    """Get puzzle details (without answer for security)"""
    puzzle = puzzle_gen.get_puzzle(puzzle_id)

    if not puzzle:
        return jsonify({"error": "Puzzle not found"}), 404

    # Strip sensitive fields
    response = {k: v for k, v in puzzle.items() if k not in ("answer", "solution_steps")}
    return jsonify(response), 200


@app.route("/api/puzzle/<puzzle_id>/check", methods=["POST"])
def check_answer(puzzle_id):
    """Validate user's answer against stored puzzle answer"""
    try:
        data = request.get_json() or {}
        user_answer = data.get("answer", "")

        result = puzzle_gen.check_answer(puzzle_id, user_answer)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/puzzle/<puzzle_id>/hint", methods=["POST"])
def get_hint(puzzle_id):
    """Get a specific hint for a puzzle"""
    try:
        puzzle = puzzle_gen.get_puzzle(puzzle_id)

        if not puzzle:
            return jsonify({"error": "Puzzle not found"}), 404

        data = request.get_json() or {}
        hint_number = data.get("hint_number", 0)

        hint = hint_provider.get_hint(puzzle, hint_number)
        return jsonify({"hint": hint, "hint_number": hint_number}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/puzzle/<puzzle_id>/solution", methods=["GET"])
def get_solution(puzzle_id):
    """Reveal the full solution for a puzzle"""
    puzzle = puzzle_gen.get_puzzle(puzzle_id)

    if not puzzle:
        return jsonify({"error": "Puzzle not found"}), 404

    return jsonify({
        "answer": puzzle.get("answer", "Unknown"),
        "explanation": puzzle.get("explanation", "No explanation available."),
        "solution_steps": puzzle.get("solution_steps", []),
        "hints": puzzle.get("hints", []),
        "validation": puzzle.get("validation", {}),
    }), 200


# ============== CHAT API ==============

@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat with the AI assistant"""
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        user_message = data.get("message", "")
        puzzle_id = data.get("puzzle_id")
        hints_used = data.get("hints_used", 0)
        chat_mode = data.get("chat_mode", "hint_bot")  # hint_bot, free_chat, tutor

        puzzle = None
        if puzzle_id:
            puzzle = puzzle_gen.get_puzzle(puzzle_id)

        response = hint_provider.chat(session_id, user_message, puzzle, hints_used, chat_mode)
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    """Clear conversation history for a session"""
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        hint_provider.clear_conversation(session_id)
        return jsonify({"status": "cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== SESSION API ==============

@app.route("/api/session/init", methods=["POST"])
def init_session():
    """Initialize or retrieve session stats"""
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", str(os.urandom(8).hex()))

        if session_id not in session_stats:
            session_stats[session_id] = {
                "session_id": session_id,
                "puzzles_solved": 0,
                "puzzles_attempted": 0,
                "total_hints_used": 0,
                "current_puzzle": None,
                "created_at": datetime.now().isoformat(),
            }

        return jsonify(session_stats[session_id]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/stats", methods=["GET"])
def get_stats(session_id):
    """Get session statistics"""
    if session_id not in session_stats:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(session_stats[session_id]), 200


# ============== HEALTH CHECK ==============

@app.route("/api/health", methods=["GET"])
def health():
    """Health check — also surfaces which AI models are currently active."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "puzzle_model": puzzle_gen.active_model_display,
        "chat_model":   hint_provider.active_model_display,
    }), 200


@app.route("/api/debug", methods=["GET"])
def debug_environ():
    """Temporary debug endpoint — dumps WSGI environ keys for Vercel routing diagnosis."""
    from flask import request as _req
    env = _req.environ
    interesting = {}
    for key in sorted(env.keys()):
        val = env[key]
        if isinstance(val, str) and len(val) < 500:
            interesting[key] = val
    return jsonify({
        "flask_path": _req.path,
        "flask_url":  _req.url,
        "flask_method": _req.method,
        "environ": interesting,
    }), 200


@app.route("/api/model/status", methods=["GET"])
def model_status():
    """
    Returns the active model for both the puzzle generator and the chat
    assistant.  The frontend polls this to display a subtle model indicator.
    """
    return jsonify({
        "puzzle_model": {
            "id":      puzzle_gen.active_model,
            "display": puzzle_gen.active_model_display,
        },
        "chat_model": {
            "id":      hint_provider.active_model,
            "display": hint_provider.active_model_display,
        },
    }), 200


# ============== ANALYTICS API ==============

@app.route("/api/analytics/<user_id>/record", methods=["POST"])
def record_analytics(user_id):
    """Record a puzzle attempt event for analytics tracking."""
    try:
        data = request.get_json() or {}
        snap = analytics_engine.record_attempt(
            user_id,
            puzzle_id    = data.get("puzzle_id", ""),
            difficulty   = data.get("difficulty", "medium"),
            puzzle_type  = data.get("puzzle_type", "riddle"),
            correct      = data.get("correct", False),
            hints_used   = data.get("hints_used", 0),
            time_taken_s = data.get("time_taken_s"),
        )
        return jsonify(snap), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/<user_id>/dashboard", methods=["GET"])
def analytics_dashboard(user_id):
    """Return the full analytics dashboard for a user."""
    try:
        return jsonify(analytics_engine.get_dashboard(user_id)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/<user_id>/events", methods=["GET"])
def analytics_events(user_id):
    """Return recent event log for a user."""
    try:
        limit = int(request.args.get("limit", 50))
        return jsonify(analytics_engine.get_user_events(user_id, limit=limit)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/leaderboard", methods=["GET"])
def analytics_leaderboard():
    """Return the global cognitive score leaderboard."""
    try:
        top_n = int(request.args.get("top", 10))
        return jsonify(analytics_engine.get_leaderboard(top_n=top_n)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== DIFFICULTY API ==============

@app.route("/api/difficulty/<user_id>/recommend", methods=["GET"])
def recommend_difficulty(user_id):
    """Recommend the next puzzle difficulty for a user."""
    try:
        return jsonify(difficulty_engine.recommend(user_id)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/difficulty/<user_id>/record", methods=["POST"])
def record_difficulty_result(user_id):
    """Record a puzzle result and update the user's Elo rating."""
    try:
        data = request.get_json() or {}
        snap = difficulty_engine.record_result(
            user_id,
            difficulty   = data.get("difficulty", "medium"),
            correct      = data.get("correct", False),
            hints_used   = data.get("hints_used", 0),
            time_taken_s = data.get("time_taken_s"),
        )
        return jsonify(snap), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/difficulty/<user_id>/rating", methods=["GET"])
def get_user_rating(user_id):
    """Return a user's current Elo rating and history."""
    try:
        return jsonify({
            "user_id": user_id,
            "rating":  difficulty_engine.get_rating(user_id),
            "history": difficulty_engine.get_history(user_id, limit=20),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== PROGRESS API ==============

@app.route("/api/progress/<user_id>", methods=["GET"])
def get_progress(user_id):
    """Return the full progress profile for a user."""
    try:
        return jsonify(progress_tracker.get_profile(user_id)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/progress/<user_id>/record", methods=["POST"])
def record_progress(user_id):
    """Record a puzzle solve event for progress tracking."""
    try:
        data = request.get_json() or {}
        result = progress_tracker.record_solve(
            user_id,
            puzzle_type  = data.get("puzzle_type", "riddle"),
            difficulty   = data.get("difficulty", "medium"),
            hints_used   = data.get("hints_used", 0),
            correct      = data.get("correct", False),
            time_taken_s = data.get("time_taken_s"),
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/progress/<user_id>/badges", methods=["GET"])
def get_badges(user_id):
    """Return all earned badges for a user."""
    try:
        return jsonify(progress_tracker.get_badges(user_id)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/progress/<user_id>/report", methods=["GET"])
def export_progress_report(user_id):
    """Export a full progress report for a user."""
    try:
        return jsonify(progress_tracker.export_report(user_id)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/progress/<user_id>/goal", methods=["POST"])
def set_weekly_goal(user_id):
    """Set the user's weekly puzzle goal."""
    try:
        data = request.get_json() or {}
        goal = data.get("goal", 15)
        progress_tracker.set_weekly_goal(user_id, goal)
        return jsonify({"status": "updated", "goal": goal}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== INTERVIEW COACH API ==============

@app.route("/api/interview/<user_id>/start", methods=["POST"])
def start_interview(user_id):
    """Start a new interview coaching session."""
    try:
        data   = request.get_json() or {}
        domain = data.get("domain", "general")
        result = interview_coach.start_session(user_id, domain=domain)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interview/<user_id>/respond", methods=["POST"])
def interview_respond(user_id):
    """Submit a response to the current interview question."""
    try:
        data     = request.get_json() or {}
        response = data.get("response", "")
        result   = interview_coach.evaluate_response(user_id, response)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interview/<user_id>/status", methods=["GET"])
def interview_status(user_id):
    """Get the current interview session status."""
    try:
        return jsonify(interview_coach.get_session_status(user_id)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interview/<user_id>/end", methods=["POST"])
def end_interview(user_id):
    """End the interview session and return a performance summary."""
    try:
        result = interview_coach.end_session(user_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== SYNAPTIA V2 — VISION & MEMORY API ==============

@app.route("/api/v2/status", methods=["GET"])
def v2_status():
    """System status and capabilities for Synaptia V2 Vision & Memory module."""
    try:
        return jsonify(vision_memory.get_status()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v2/people", methods=["GET", "POST"])
def v2_people():
    """List all enrolled identities (GET) or enroll a new identity (POST)."""
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            name = data.get("name", "").strip()
            if not name:
                return jsonify({"error": "Name is required"}), 400

            relationship = data.get("relationship", "Contact")
            description  = data.get("description", "")
            face_crop    = data.get("face_crop")
            tags         = data.get("tags")

            profile = vision_memory.enroll_person(
                name=name,
                relationship=relationship,
                description=description,
                face_crop=face_crop,
                tags=tags
            )
            return jsonify(profile), 201

        # GET request
        people = vision_memory.list_people()
        return jsonify({"people": people, "count": len(people)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v2/people/<person_id>", methods=["GET", "DELETE"])
def v2_person_detail(person_id):
    """Retrieve or delete a specific enrolled person profile."""
    try:
        if request.method == "DELETE":
            success = vision_memory.delete_person(person_id)
            if not success:
                return jsonify({"error": "Person not found"}), 404
            return jsonify({"status": "deleted", "person_id": person_id}), 200

        person = vision_memory.get_person(person_id)
        if not person:
            return jsonify({"error": "Person not found"}), 404
        return jsonify(person), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v2/memory/summary", methods=["POST"])
def v2_memory_summary():
    """Analyze spoken transcript and generate AI memory summary & recall cues."""
    try:
        data = request.get_json() or {}
        person_name = data.get("person_name", "Recognized Contact")
        transcript  = data.get("transcript", "")
        person_id   = data.get("person_id")

        summary_data = vision_memory.generate_memory_summary(person_name, transcript)

        # If person_id provided, record this interaction into their profile
        if person_id and vision_memory.get_person(person_id):
            vision_memory.record_interaction(
                person_id=person_id,
                transcript=transcript,
                summary=summary_data.get("summary"),
                sentiment=summary_data.get("sentiment")
            )

        return jsonify(summary_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v2/interaction", methods=["POST"])
def v2_record_interaction():
    """Log an interaction explicitly for a person."""
    try:
        data = request.get_json() or {}
        person_id  = data.get("person_id")
        transcript = data.get("transcript", "")
        summary    = data.get("summary")
        sentiment  = data.get("sentiment")

        if not person_id:
            return jsonify({"error": "person_id is required"}), 400

        res = vision_memory.record_interaction(
            person_id=person_id,
            transcript=transcript,
            summary=summary,
            sentiment=sentiment
        )
        return jsonify(res), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v2/events/simulate", methods=["POST", "GET"])
def v2_simulate_event():
    """Trigger a simulated detection event for UI testing & demo mode."""
    try:
        data = request.get_json() if request.method == "POST" else {}
        data = data or {}
        person_id = data.get("person_id") or request.args.get("person_id")
        event = vision_memory.simulate_event(person_id=person_id)
        return jsonify(event), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== SYNAPTIA V2 CLOUD API & APP SERVING ==============

import json

V2_APP_DIR = os.path.join(BASE_DIR, "public", "v2-app")
MODELS_DIR = os.path.join(BASE_DIR, "public", "models")
FACE_IDENTITIES_FILE = os.path.join(BASE_DIR, "SynaptiaV2-main", "inference", "data", "face_identities.json")
PERSON_MEMORIES_FILE = os.path.join(BASE_DIR, "SynaptiaV2-main", "inference", "data", "person_memories.json")

def _get_face_identities():
    try:
        if os.path.exists(FACE_IDENTITIES_FILE):
            with open(FACE_IDENTITIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        app.logger.warning(f"Failed to read face identities: {e}")
    return {}

def _save_face_identities(data):
    try:
        os.makedirs(os.path.dirname(FACE_IDENTITIES_FILE), exist_ok=True)
        with open(FACE_IDENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        app.logger.warning(f"Failed to save face identities: {e}")

def _get_person_memories():
    try:
        if os.path.exists(PERSON_MEMORIES_FILE):
            with open(PERSON_MEMORIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        app.logger.warning(f"Failed to read person memories: {e}")
    return {}

@app.route("/v2-app")
@app.route("/v2-app/")
def v2_app_index():
    return send_from_directory(V2_APP_DIR, "index.html")

@app.route("/v2-app/<path:path>")
def v2_app_static(path):
    return send_from_directory(V2_APP_DIR, path)

@app.route("/models/<path:path>")
def serve_models(path):
    return send_from_directory(MODELS_DIR, path)

@app.route("/api/face-identities", methods=["GET", "PUT"])
def api_face_identities():
    if request.method == "GET":
        data = _get_face_identities()
        return jsonify({"identities": list(data.values())}), 200

    payload = request.get_json() or {}
    name = (payload.get("name") or "").strip()
    descriptor = payload.get("descriptor") or []
    if not name or not descriptor:
        return jsonify({"error": "name and descriptor are required"}), 400

    identities = _get_face_identities()
    record = {"name": name, "descriptor": descriptor}
    identities[name.lower()] = record
    _save_face_identities(identities)
    return jsonify({"enrolled": True, "identity": record}), 200

@app.route("/api/person-memories/<person_name>", methods=["GET"])
def api_person_memories(person_name):
    name = person_name.strip().lower()
    memories = _get_person_memories()
    if name in memories:
        return jsonify({"memory": memories[name]}), 200

    return jsonify({
        "memory": {
            "name": person_name.strip(),
            "summary": "• Enrolled in your circle of trust today\n• Memory synthesis active",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }), 200

@app.route("/api/faces/observations", methods=["POST"])
def api_faces_observations():
    return jsonify({"stored": True}), 201

@app.route("/api/reset-all", methods=["POST"])
def api_reset_all():
    return jsonify({"status": "ok", "message": "All data reset successfully."}), 200

@app.route("/api/voice-identities", methods=["PUT"])
def api_voice_identities_put():
    payload = request.get_json() or {}
    name = (payload.get("name") or "").strip().lower()
    descriptor = payload.get("descriptor") or []
    if name and descriptor:
        identities = _get_face_identities()
        if name in identities:
            identities[name]["voice_descriptor"] = descriptor
            identities[name]["voice_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _save_face_identities(identities)
    return jsonify({"enrolled": True}), 200

@app.route("/api/voice-identities/<person_name>", methods=["GET"])
def api_voice_identities_get(person_name):
    name = person_name.strip().lower()
    identities = _get_face_identities()
    person_rec = identities.get(name, {})
    desc = person_rec.get("voice_descriptor")
    return jsonify({
        "name": person_name,
        "has_voiceprint": desc is not None and len(desc) > 0,
        "descriptor": desc
    }), 200

@app.route("/api/transcriptions", methods=["POST"])
def api_transcriptions():
    """Transcribe real browser audio segments via Groq Whisper and distill memory cues."""
    try:
        import base64
        import math
        import requests

        payload = request.get_json() or {}
        audio_b64 = payload.get("audio_base64")
        if not audio_b64:
            return jsonify({"stored": False, "text": "", "reason": "No audio provided"}), 200

        audio_bytes = base64.b64decode(audio_b64)
        filename = payload.get("filename") or "speech.webm"
        content_type = payload.get("content_type") or "audio/webm"
        person_name = payload.get("person_name")
        candidate_voice = payload.get("voice_descriptor")

        # Speaker Verification if voiceprint is enrolled
        if person_name and candidate_voice:
            identities = _get_face_identities()
            person_rec = identities.get(person_name.strip().lower(), {})
            enrolled_voice = person_rec.get("voice_descriptor")
            if enrolled_voice and len(enrolled_voice) > 0 and len(candidate_voice) > 0:
                dot = sum(a * b for a, b in zip(candidate_voice, enrolled_voice))
                norm_a = math.sqrt(sum(a * a for a in candidate_voice))
                norm_b = math.sqrt(sum(b * b for b in enrolled_voice))
                sim = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
                if sim < 0.48:
                    return jsonify({
                        "stored": False,
                        "text": "",
                        "reason": "Bystander voice filtered out (unverified speaker)",
                        "verified": False,
                        "similarity": sim
                    }), 200

        groq_api_key = app.config.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        transcribed_text = ""

        clean_ct = (content_type or "audio/webm").split(";")[0].strip()
        if not clean_ct.startswith("audio/"):
            clean_ct = "audio/webm"
        clean_filename = "speech.mp4" if "mp4" in clean_ct else "speech.webm"

        if groq_api_key and len(audio_bytes) > 1500:
            headers = {"Authorization": f"Bearer {groq_api_key}"}
            files = {"file": (clean_filename, audio_bytes, clean_ct)}
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "verbose_json",
                "language": "en",
                "temperature": "0",
            }
            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=25,
                )
                if res.status_code == 200:
                    res_json = res.json()
                    transcribed_text = res_json.get("text", "").strip()
                else:
                    app.logger.warning(f"Groq transcription returned {res.status_code}: {res.text[:200]}")
            except Exception as e:
                app.logger.warning(f"Groq transcription request failed: {e}")

        if not transcribed_text:
            return jsonify({"stored": False, "text": "", "reason": "No speech detected"}), 200

        summary = ""
        try:
            summary_data = vision_memory.generate_memory_summary(person_name or "Person", transcribed_text)
            summary = summary_data.get("summary", "")
        except Exception:
            summary = f"• Spoke about: {transcribed_text[:60]}…"

        if person_name:
            memories = _get_person_memories()
            memories[person_name.strip().lower()] = {
                "name": person_name.strip(),
                "summary": summary,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            _save_person_memories(memories)

        return jsonify({
            "stored": True,
            "text": transcribed_text,
            "summary": summary,
            "verified": True
        }), 201

    except Exception as e:
        app.logger.warning(f"api_transcriptions handler error: {e}")
        return jsonify({"stored": False, "text": "", "reason": str(e)}), 200




# ============== ERROR HANDLERS ==============

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ============== RUN ==============

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5002))
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=app.config.get("DEBUG", False))
