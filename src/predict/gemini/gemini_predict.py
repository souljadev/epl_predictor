import json
import os
from typing import Dict

from dotenv import load_dotenv
from google import genai

# ----------------------------------
# ENV + CLIENT SETUP
# ----------------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"


# ----------------------------------
# PROMPT BUILDER
# ----------------------------------
def build_prompt(home_team: str, away_team: str) -> str:
    return f"""
Predict the football match below.

Return ONLY valid JSON in exactly this format:
{{
  "predicted_score": "X-Y",
  "predicted_winner": "Team Name or Draw"
}}

Match:
{home_team} vs {away_team}
""".strip()


# ----------------------------------
# GEMINI CALL
# ----------------------------------
def call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned empty response")

    return response.text.strip()


# ----------------------------------
# PARSER
# ----------------------------------
def parse_prediction(raw_text: str) -> Dict[str, str]:
    cleaned = raw_text.strip()

    # Remove Markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")

        # Remove optional language tag like "json"
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini did not return valid JSON after cleaning:\n{raw_text}"
        ) from e

    required = {"predicted_score", "predicted_winner"}
    if not required.issubset(parsed):
        raise RuntimeError(f"Missing required keys: {parsed}")

    return {
        "predicted_score": parsed["predicted_score"].strip(),
        "predicted_winner": parsed["predicted_winner"].strip(),
    }


# ----------------------------------
# PUBLIC API
# ----------------------------------
def get_gemini_prediction(home_team: str, away_team: str) -> Dict[str, str]:
    prompt = build_prompt(home_team, away_team)
    raw = call_gemini(prompt)
    parsed = parse_prediction(raw)

    return {
        "home_team": home_team,
        "away_team": away_team,
        **parsed,
    }


# ----------------------------------
# CLI TEST
# ----------------------------------
if __name__ == "__main__":
    prediction = get_gemini_prediction("Arsenal", "Brentford")
    print(json.dumps(prediction, indent=2))
