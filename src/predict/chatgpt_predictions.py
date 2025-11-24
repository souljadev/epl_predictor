import sys
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from db import insert_predictions, get_conn   # noqa: E402
from openai import OpenAI


def ask_chatgpt(fixtures_df):
    client = OpenAI()

    prompt = "Predict the final scores for these fixtures:\n\n"
    for _, row in fixtures_df.iterrows():
        prompt += f"{row['HomeTeam']} vs {row['AwayTeam']}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def parse_chatgpt_output(text):
    """
    Expected ChatGPT format:
    Arsenal 2-1 Chelsea
    Man City 3-0 Wolves
    """
    lines = text.strip().split("\n")
    out = []

    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue

        home = parts[0]
        score = parts[1]
        away = parts[2]

        out.append((home, away, score))

    return out


def generate_chatgpt_predictions():
    with get_conn() as conn:
        fixtures = pd.read_sql("SELECT * FROM fixtures", conn)

    raw = ask_chatgpt(fixtures)
    rows = parse_chatgpt_output(raw)

    for home, away, score in rows:
        row_dict = {
            "date": "chat",         # optional placeholder
            "home_team": home,
            "away_team": away,
            "model_version": "chatgpt",
            "dixon_coles_probs": None,
            "elo_probs": None,
            "ensemble_probs": None,
            "home_win_prob": None,
            "draw_prob": None,
            "away_win_prob": None,
            "exp_goals_home": None,
            "exp_goals_away": None,
            "exp_total_goals": None,
            "score_pred": None,
            "chatgpt_pred": score,
        }

        insert_predictions(row_dict)

    print("✓ ChatGPT predictions saved to DB")


if __name__ == "__main__":
    generate_chatgpt_predictions()
