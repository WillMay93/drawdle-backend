# app.py — secure version
import os
import json
import re
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Secure API key loading ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not set — add it to your .env or Render environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Targets storage placeholder ---
TARGETS = {
    "cat": {"prompt": "cat", "public_name": "animal", "category": "animal", "colour": "black"},
}

# --- Helper: pick target of the day ---
def get_daily_target_id(date_override: str = None):
    dt = datetime.fromisoformat(date_override) if date_override else datetime.now(timezone.utc)
    easy_targets = [k for k, v in TARGETS.items() if v.get("colour") in ["red", "yellow", "blue", "green", "black"]]
    if not easy_targets:
        return "cat"
    days_since_epoch = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).days
    return easy_targets[days_since_epoch % len(easy_targets)]


@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


def parse_json_from_text(text):
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@app.route("/target", methods=["GET"])
def get_target():
    """Expose today's target for the frontend."""
    target_id = get_daily_target_id()
    target = TARGETS.get(target_id)
    if not target:
        return jsonify({"error": "No target found"}), 404
    return jsonify({
        "target_id": target_id,
        "public_name": target["public_name"],
        "colour": target.get("colour"),
        "category": target.get("category")
    })


@app.route("/submit", methods=["POST", "OPTIONS"])
def submit():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.get_json(force=True)
        image_base64 = data.get("image_base64")
        attempt = int(data.get("attempt", 1))

        target_id = get_daily_target_id()
        target_info = TARGETS.get(target_id)
        if not image_base64 or not target_info:
            return jsonify({"success": False, "message": "Missing image or invalid target"}), 400

        target_prompt = target_info["prompt"]
        expected_category = target_info.get("category", "")
        expected_colour = target_info.get("colour", "")

        img_data = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

        system_instruction = (
            "You are a strict AI judge for a drawing game. "
            "Compare the drawing to the target concept. Output only JSON with fields: "
            "score (0-100), guess (string), correct (bool), color_match (bool), "
            "shape_match (bool), style_score (0-25), and category (string). "
            f"The target is '{target_prompt}' with expected color '{expected_colour}' "
            f"and category '{expected_category}'."
        )

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_instruction}]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": f"Target (secret): {target_prompt}"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_data}"}
                ]}
            ],
            temperature=0.0
        )

        parsed = parse_json_from_text(getattr(response, "output_text", "")) or {}

        return jsonify({
            "success": bool(parsed.get("correct")),
            "score": int(parsed.get("score", 0)),
            "guess": parsed.get("guess", ""),
            "category": parsed.get("category", ""),
            "color_match": bool(parsed.get("color_match")),
            "shape_match": bool(parsed.get("shape_match")),
            "style_score": int(parsed.get("style_score", 0)),
            "expected_category": expected_category,
            "expected_colour": expected_colour,
            "target_id": target_id
        })

    except Exception as e:
        print("Error in /submit:", e)
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
