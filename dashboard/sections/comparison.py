import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss
from datetime import date

SEASON_START = pd.Timestamp("2024-08-01")
SEASON_END   = pd.Timestamp("2025-06-15")

# Official EPL team list used for filtering predictions
EPL_TEAMS_2024 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea",
    "Crystal Palace","Everton","Fulham","Ipswich","Leicester",
    "Liverpool","Man City","Man United","Newcastle",
    "Nott'm Forest","Southampton","Tottenham","West Ham","Wolves"
}

# Team name normalization across predictions + FBref results
TEAM_FIX = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester Utd": "Man United",

    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
}

# ------------------------------------------------------------
# DB LOAD
# ------------------------------------------------------------
def load_results(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT date, home_team, away_team, FTHG, FTAG, Result
        FROM results
    """, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def load_predictions(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            date,
            home_team,
            away_team,
            model_version,
            home_win_prob,
            draw_prob,
            away_win_prob,
            exp_goals_home,
            exp_goals_away,
            exp_total_goals,
            score_pred,
            chatgpt_pred,
            created_at
        FROM predictions
    """, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])



# ------------------------------------------------------------
# WINNER HELPERS
# ------------------------------------------------------------
def winner_from_score(score):
    try:
        h, a = map(int, score.split("-"))
        if h > a: return "H"
        if h < a: return "A"
        return "D"
    except:
        return None


# ------------------------------------------------------------
# RENDER COMPARISON VIEW
# ------------------------------------------------------------
def render(db_path: Path):
    st.subheader("Model vs ChatGPT vs Actual — Past 30 Days")

    today = pd.Timestamp.today().normalize()
    window_start = today - pd.Timedelta(days=30)

    # --------------------------
    # LOAD DATA
    # --------------------------
    preds = load_predictions(db_path)
    results = load_results(db_path)

    # --------------------------
    # NORMALIZE TEAM NAMES
    # --------------------------
    preds["home_team"] = preds["home_team"].replace(TEAM_FIX)
    preds["away_team"] = preds["away_team"].replace(TEAM_FIX)
    results["home_team"] = results["home_team"].replace(TEAM_FIX)
    results["away_team"] = results["away_team"].replace(TEAM_FIX)

    # --------------------------
    # FILTER TO EPL MATCHES ONLY
    # --------------------------
    preds = preds[
        preds["home_team"].isin(EPL_TEAMS_2024) &
        preds["away_team"].isin(EPL_TEAMS_2024)
    ]

    # --------------------------
    # DATE FILTERING
    # --------------------------
    preds = preds[preds["date"].between(window_start, today)]
    results = results[results["date"].between(window_start, today)]

    if preds.empty:
        st.info("No EPL predictions found in the past 30 days.")
        return

    # --------------------------
    # SPLIT MODEL VS CHATGPT
    # --------------------------
    model_df = preds[preds["model_version"].str.startswith("dc_elo")].copy()
    gpt_df   = preds[preds["model_version"] == "chatgpt"].copy()

    model_df = model_df.rename(columns={"score_pred": "model_score"})
    gpt_df = gpt_df.rename(columns={"chatgpt_pred": "chatgpt_score"})
    gpt_df = gpt_df[["date","home_team","away_team","chatgpt_score"]]

    # --------------------------
    # MERGE → ONE ROW PER MATCH
    # --------------------------
    merged = (
        model_df.merge(
            gpt_df,
            on=["date","home_team","away_team"],
            how="left"
        )
        .merge(
            results[["date","home_team","away_team","FTHG","FTAG","Result"]],
            on=["date","home_team","away_team"],
            how="left"
        )
    )

    # --------------------------
    # WINNER CALCULATIONS
    # --------------------------
    def winner_from_probs(row):
        arr = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        return {0:"H",1:"D",2:"A"}[int(np.argmax(arr))]

    merged["model_winner"] = merged.apply(winner_from_probs, axis=1)
    merged["gpt_winner"] = merged["chatgpt_score"].apply(winner_from_score)
    merged["actual_winner"] = merged["Result"]

    merged["model_correct"] = merged["model_winner"] == merged["actual_winner"]
    merged["gpt_correct"] = merged["gpt_winner"] == merged["actual_winner"]

    # --------------------------
    # DISPLAY TABLE
    # --------------------------
    merged["H Prob"] = (merged["home_win_prob"] * 100).round(1)
    merged["D Prob"] = (merged["draw_prob"] * 100).round(1)
    merged["A Prob"] = (merged["away_win_prob"] * 100).round(1)

    display_df = merged[
        [
            "date",
            "home_team",
            "away_team",
            "FTHG",
            "FTAG",
            "actual_winner",
            "H Prob","D Prob","A Prob",
            "model_score",
            "model_winner",
            "model_correct",
            "chatgpt_score",
            "gpt_winner",
            "gpt_correct",
        ]
    ]

    def highlight(val):
        if val is True:
            return "background-color: #c6efce;"  # green
        if val is False:
            return "background-color: #ffc7ce;"  # red
        return ""

    st.dataframe(
        display_df.style.applymap(highlight, subset=["model_correct","gpt_correct"]),
        use_container_width=True
    )

    # --------------------------
    # MATCH EXPANDERS
    # --------------------------
    st.markdown("#### Per-match Details")

    for _, row in merged.iterrows():
        with st.expander(
            f"{row['date'].date()} — {row['home_team']} vs {row['away_team']} "
            f"(Actual: {row['FTHG']}-{row['FTAG']})"
        ):
            st.write(f"### Actual Winner: `{row['actual_winner']}`")

            st.write(
                f"**Model Winner:** `{row['model_winner']}` — "
                f"{'✔️ Correct' if row['model_correct'] else '❌ Wrong'}"
            )

            st.write(
                f"**ChatGPT Winner:** `{row['gpt_winner']}` — "
                f"{'✔️ Correct' if row['gpt_correct'] else '❌ Wrong'}"
            )

            st.write("---")

            st.write("### Probabilities")
            st.write(
                f"Home `{row['home_win_prob']:.3f}`, "
                f"Draw `{row['draw_prob']:.3f}`, "
                f"Away `{row['away_win_prob']:.3f}`"
            )

            st.write("### Expected Goals")
            st.write(
                f"Home `{row['exp_goals_home']:.2f}`, "
                f"Away `{row['exp_goals_away']:.2f}`, "
                f"Total `{row['exp_total_goals']:.2f}`"
            )

            st.write("### Score Predictions")
            st.write(f"Model predicted: `{row['model_score']}`")
            st.write(f"ChatGPT predicted: `{row['chatgpt_score']}`")
