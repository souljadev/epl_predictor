# src/agent/chatgpt_engine.py

import os
import sys
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

try:
    # OpenAI Python SDK v1.x
    from openai import OpenAI
except ImportError:
    OpenAI = None  # handled below


# ------------------------------------------------------------------------------------
# Path / import setup so we can import db from src/
# ------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local/src
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import (  # type: ignore
    get_upcoming_fixtures,
    upsert_chatgpt_predictions,
)


# ------------------------------------------------------------------------------------
# Poisson helpers (reused for ensemble logic if desired)
# ------------------------------------------------------------------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ------------------------------------------------------------------------------------
# Prompt + parsing helpers
# ------------------------------------------------------------------------------------
def _build_prompt_for_fixtures(fixtures_df: pd.DataFrame) -> str:
    """
    Build a compact prompt listing fixtures and expected CSV response format.
    We do NOT include fixture_id in the prompt; we join back on (Date, HomeTeam, AwayTeam).
    """
    lines = ["You are a football prediction assistant.",
             "For each EPL match, predict the final score and a confidence level.",
             "",
             "Return ONLY a CSV with columns:",
             "Date,HomeTeam,AwayTeam,PredictedScore,Confidence",
             "",
             "Where:",
             "- PredictedScore is like '2-1' or '1-1'",
             "- Confidence is an integer percent like '70%'",
             "",
             "Here are the matches:"]

    for _, row in fixtures_df.iterrows():
        lines.append(
            f"{row['Date']} - {row['HomeTeam']} vs {row['AwayTeam']}"
        )

    lines.append("")
    lines.append("Now output the CSV ONLY (no explanation).")

    return "\n".join(lines)


def _parse_chatgpt_csv_response(text: str) -> pd.DataFrame:
    """
    Parse the CSV that ChatGPT returns with:
    Date,HomeTeam,AwayTeam,PredictedScore,Confidence

    Confidence may be '70%' or '0.7'. We convert to float 0–1.
    """
    # Try to locate the CSV portion (in case the model adds extra text)
    # Heuristic: take lines from first line containing 'Date,HomeTeam' onward.
    lines = text.strip().splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if "Date" in line and "HomeTeam" in line and "PredictedScore" in line:
            start_idx = i
            break
    csv_text = "\n".join(lines[start_idx:])

    df = pd.read_csv(pd.compat.StringIO(csv_text)) if hasattr(pd, "compat") else pd.read_csv(
        # fallback for older pandas without pd.compat.StringIO
        __import__("io").StringIO(csv_text)
    )

    # Normalize columns
    expected_cols = ["Date", "HomeTeam", "AwayTeam", "PredictedScore", "Confidence"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ChatGPT CSV missing columns: {missing}. Got columns: {df.columns.tolist()}")

    # Clean confidence → float 0–1
    def _conf_to_float(x):
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):
                try:
                    return float(x[:-1]) / 100.0
                except ValueError:
                    return None
            try:
                v = float(x)
                return v if v <= 1.0 else v / 100.0
            except ValueError:
                return None
        try:
            v = float(x)
            return v if v <= 1.0 else v / 100.0
        except Exception:
            return None

    df["Confidence"] = df["Confidence"].apply(_conf_to_float)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["PredictedScore"] = df["PredictedScore"].astype(str).str.strip()

    return df


# ------------------------------------------------------------------------------------
# Core call: single ChatGPT model for a block of fixtures
# ------------------------------------------------------------------------------------
def generate_chatgpt_predictions_for_fixtures(
    fixtures_df: pd.DataFrame,
    model_name: str,
    temperature: float = 0.2,
) -> pd.DataFrame:
    """
    Call ChatGPT for a batch of fixtures and return a DataFrame:

    Columns:
      Date, HomeTeam, AwayTeam, PredictedScore, Confidence
    """
    if fixtures_df.empty:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "PredictedScore", "Confidence"])

    if OpenAI is None:
        raise ImportError(
            "OpenAI SDK not installed. Install with `pip install openai` "
            "and ensure OPENAI_API_KEY is set."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    prompt = _build_prompt_for_fixtures(fixtures_df)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert football prediction engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    content = resp.choices[0].message.content
    df_preds = _parse_chatgpt_csv_response(content)

    # We want to make sure we only keep rows that match fixtures_df
    # Join back on Date, HomeTeam, AwayTeam
    fixtures_keyed = fixtures_df.copy()
    fixtures_keyed["Date"] = pd.to_datetime(fixtures_keyed["Date"]).dt.date

    merged = pd.merge(
        fixtures_keyed,
        df_preds,
        how="inner",
        left_on=["Date", "HomeTeam", "AwayTeam"],
        right_on=["Date", "HomeTeam", "AwayTeam"],
        suffixes=("", "_pred"),
    )

    # Result has original fixture columns + PredictedScore, Confidence
    return merged[["fixture_id", "Date", "HomeTeam", "AwayTeam", "PredictedScore", "Confidence"]]


# ------------------------------------------------------------------------------------
# Ensemble builder: combine gpt-4o-mini + gpt-5.1
# ------------------------------------------------------------------------------------
def build_ensemble_predictions(
    df_mini: pd.DataFrame,
    df_51: pd.DataFrame,
    w_51: float = 0.7,
    w_mini: float = 0.3,
) -> pd.DataFrame:
    """
    Build ensemble predictions from two DataFrames with columns:
      fixture_id, Date, HomeTeam, AwayTeam, PredictedScore, Confidence
    Matching is done on fixture_id.

    Ensemble rules:
      - If both agree on score → use that score, high blended confidence
      - Else if both agree on winner (team) → ensemble winner that team, pick score from higher-confidence model
      - Else (winner disagree) → pick winner & score from weighted higher-confidence model (favor gpt-5.1)
      - Confidence = weighted blend of both confidences
    """
    if df_mini.empty and df_51.empty:
        return pd.DataFrame(columns=["fixture_id", "Date", "HomeTeam", "AwayTeam", "PredictedScore", "Confidence"])

    # Ensure numeric confidence
    for df in (df_mini, df_51):
        if "Confidence" in df.columns:
            df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")

    left_cols = ["fixture_id", "Date", "HomeTeam", "AwayTeam", "PredictedScore", "Confidence"]
    right_cols = ["fixture_id", "Date", "HomeTeam", "AwayTeam", "PredictedScore", "Confidence"]

    mini = df_mini[left_cols].rename(
        columns={
            "PredictedScore": "Score_mini",
            "Confidence": "Conf_mini",
        }
    )
    m51 = df_51[right_cols].rename(
        columns={
            "PredictedScore": "Score_51",
            "Confidence": "Conf_51",
        }
    )

    merged = pd.merge(
        mini,
        m51,
        on=["fixture_id", "Date", "HomeTeam", "AwayTeam"],
        how="outer",
    )

    def winner_from_score(score: str, home: str, away: str) -> str:
        if not isinstance(score, str) or "-" not in score:
            return ""
        try:
            h_str, a_str = score.split("-")
            h, a = int(h_str), int(a_str)
        except Exception:
            return ""

        if h > a:
            return home
        if a > h:
            return away
        return "Draw"

    def resolve_row(row):
        score_mini = row.get("Score_mini", "")
        score_51 = row.get("Score_51", "")
        conf_mini = row.get("Conf_mini", 0.0) or 0.0
        conf_51 = row.get("Conf_51", 0.0) or 0.0

        home = row["HomeTeam"]
        away = row["AwayTeam"]

        win_mini = winner_from_score(score_mini, home, away)
        win_51 = winner_from_score(score_51, home, away)

        # If both scores exactly same → use that
        if score_mini and score_51 and score_mini == score_51:
            score_ens = score_mini
        else:
            # Scores differ or one missing
            if win_mini and win_51 and win_mini == win_51:
                # Same winner, diff score → pick score from higher-confidence model
                if conf_51 >= conf_mini:
                    score_ens = score_51 or score_mini
                else:
                    score_ens = score_mini or score_51
            else:
                # Winners disagree or missing → weighted preference to higher confidence *and* w_51 vs w_mini
                # Compute effective weights
                eff_51 = conf_51 * w_51
                eff_mini = conf_mini * w_mini
                if eff_51 >= eff_mini:
                    score_ens = score_51 or score_mini
                else:
                    score_ens = score_mini or score_51

        # Confidence: blended if both present, else whichever exists
        if conf_51 and conf_mini:
            conf_ens = (w_51 * conf_51 + w_mini * conf_mini) / (w_51 + w_mini)
        else:
            conf_ens = conf_51 or conf_mini or 0.5

        return pd.Series({"PredictedScore": score_ens, "Confidence": conf_ens})

    ens = merged.apply(resolve_row, axis=1)
    out = pd.concat([merged[["fixture_id", "Date", "HomeTeam", "AwayTeam"]], ens], axis=1)

    return out


# ------------------------------------------------------------------------------------
# Top-level pipeline: run both models + ensemble and write to DB
# ------------------------------------------------------------------------------------
def run_full_chatgpt_pipeline(
    run_id: str,
    days_ahead: int = 7,
    model_name_mini: str = "gpt-4o-mini",
    model_name_51: str = "gpt-5.1",
):
    """
    High-level function to be called from the agent.

    Steps:
      1. Get upcoming fixtures from DB (0..days_ahead)
      2. Call ChatGPT twice (mini + 5.1)
      3. Build ensemble predictions
      4. Write all three variants into predictions_chatgpt table via db.upsert_chatgpt_predictions
         with model_name in {'gpt-4o-mini', 'gpt-5.1', 'gpt-ensemble'}
    """
    fixtures = get_upcoming_fixtures(days_ahead=days_ahead)
    if fixtures.empty:
        print("[ChatGPT] No upcoming fixtures found; skipping ChatGPT predictions.")
        return

    # Ensure these columns exist
    for col in ["fixture_id", "Date", "HomeTeam", "AwayTeam"]:
        if col not in fixtures.columns:
            raise ValueError(f"get_upcoming_fixtures is missing column: {col}")

    print(f"[ChatGPT] Generating predictions for {len(fixtures)} fixtures...")

    df_mini = generate_chatgpt_predictions_for_fixtures(
        fixtures_df=fixtures,
        model_name=model_name_mini,
        temperature=0.2,
    )
    print(f"[ChatGPT] gpt-4o-mini predictions generated for {len(df_mini)} fixtures.")

    df_51 = generate_chatgpt_predictions_for_fixtures(
        fixtures_df=fixtures,
        model_name=model_name_51,
        temperature=0.2,
    )
    print(f"[ChatGPT] gpt-5.1 predictions generated for {len(df_51)} fixtures.")

    df_ens = build_ensemble_predictions(df_mini, df_51)
    print(f"[ChatGPT] Ensemble predictions generated for {len(df_ens)} fixtures.")

    # Write to DB
    run_ts = datetime.utcnow().isoformat()

    if not df_mini.empty:
        upsert_chatgpt_predictions(run_id=run_id, model_name="gpt-4o-mini", run_ts=run_ts, df=df_mini)

    if not df_51.empty:
        upsert_chatgpt_predictions(run_id=run_id, model_name="gpt-5.1", run_ts=run_ts, df=df_51)

    if not df_ens.empty:
        upsert_chatgpt_predictions(run_id=run_id, model_name="gpt-ensemble", run_ts=run_ts, df=df_ens)

    print("[ChatGPT] All ChatGPT predictions (mini, 5.1, ensemble) saved to DB.")
