import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss
from datetime import date


SEASON_START = pd.Timestamp("2024-08-01")
SEASON_END   = pd.Timestamp("2025-06-15")

EPL_TEAMS_2024 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea",
    "Crystal Palace","Everton","Fulham","Ipswich","Leicester",
    "Liverpool","Man City","Man United","Newcastle",
    "Nott'm Forest","Southampton","Tottenham","West Ham","Wolves"
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

def load_predictions_for_date(db_path: Path, target_ts: pd.Timestamp) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
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
        WHERE date = ?
        ORDER BY home_team, away_team, model_version
        """,
        conn,
        params=(target_ts.strftime("%Y-%m-%d"),),
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def winner_from_goals(h, a):
    if h > a:
        return "H"
    if h == a:
        return "D"
    return "A"


def winner_from_score(score):
    try:
        h, a = map(int, score.split("-"))
        return winner_from_goals(h, a)
    except:
        return None


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return 0, 0, 0

    df = df.copy()
    df["actual"] = df.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)

    # Use model probabilities only (dc_elo)
    df = df[df["model_version"].str.startswith("dc_elo")]

    if df.empty:
        return 0, 0, 0

    df["actual_idx"] = df["actual"].map({"H":0, "D":1, "A":2})
    df["pred_class"] = df.apply(
        lambda r: np.argmax([r["home_win_prob"], r["draw_prob"], r["away_win_prob"]]),
        axis=1,
    )

    accuracy = (df["pred_class"] == df["actual_idx"]).mean()

    y_true = df["actual_idx"].values
    probs  = df[["home_win_prob","draw_prob","away_win_prob"]].values

    try:
        brier = brier_score_loss(y_true, probs, labels=[0,1,2])
    except:
        brier = np.nan

    try:
        ll = log_loss(y_true, probs, labels=[0,1,2])
    except:
        ll = np.nan

    return accuracy, brier, ll


# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
def render(db_path: Path):
    st.subheader("Match Predictions")

    today = date.today()
    selected_date = st.date_input(
        "Select match date",
        value=today,
    )

    target_ts = pd.to_datetime(selected_date)

    df = load_predictions_for_date(db_path, target_ts)

    if df.empty:
        st.info("No predictions found for this date. Generate predictions first.")
        return

    st.markdown(f"### Predictions for {target_ts.date()}")

    # ------------------------------------------------------------
    # Fix duplicates: pivot ChatGPT joins horizontally
    # ------------------------------------------------------------

    model_df = df[df["model_version"].str.startswith("dc_elo")].copy()
    gpt_df   = df[df["model_version"] == "chatgpt"].copy()

    # Rename columns to avoid confusion
    model_df = model_df.rename(columns={
        "score_pred": "model_score"
    })
    gpt_df = gpt_df.rename(columns={
        "chatgpt_pred": "chatgpt_score"
    })

    # Only keep ChatGPT's score for merge
    gpt_df = gpt_df[["date","home_team","away_team","chatgpt_score"]]

    # Merge → ONE ROW PER MATCH
    merged = model_df.merge(
        gpt_df,
        on=["date","home_team","away_team"],
        how="left"
    )

    # ------------------------------------------------------------
    # Display table
    # ------------------------------------------------------------
    # Add probabilities to BOTH dataframes
    merged["H Prob"] = (merged["home_win_prob"] * 100).round(1)
    merged["D Prob"] = (merged["draw_prob"] * 100).round(1)
    merged["A Prob"] = (merged["away_win_prob"] * 100).round(1)

    display_df = merged.copy()


    display_df = display_df[
        [
            "date",
            "home_team",
            "away_team",
            "model_version",
            "H Prob", "D Prob", "A Prob",
            "exp_goals_home",
            "exp_goals_away",
            "exp_total_goals",
            "model_score",
            "chatgpt_score",
        ]
    ]

    st.dataframe(display_df, use_container_width=True)

    # ------------------------------------------------------------
    # Expanders (1 per match)
    # ------------------------------------------------------------
    st.markdown("#### Per-match details")

    for _, row in merged.iterrows():
        with st.expander(
            f"{row['home_team']} vs {row['away_team']} — "
            f"{row['H Prob']:.1f}% / "
            f"{row['D Prob']:.1f}% / "
            f"{row['A Prob']:.1f}%"
        ):
            st.write(f"**Model version:** `{row['model_version']}`")
            st.write(
                f"**Probabilities** – Home: `{row['home_win_prob']:.3f}`, "
                f"Draw: `{row['draw_prob']:.3f}`, "
                f"Away: `{row['away_win_prob']:.3f}`"
            )
            st.write(
                f"**Expected goals** – Home: `{row['exp_goals_home']:.2f}`, "
                f"Away: `{row['exp_goals_away']:.2f}`, "
                f"Total: `{row['exp_total_goals']:.2f}`"
            )
            st.write(f"**Model score:** `{row['model_score']}`")
            if row["chatgpt_score"]:
                st.write(f"**ChatGPT score:** `{row['chatgpt_score']}`")
            else:
                st.write("_No ChatGPT prediction for this match._")
