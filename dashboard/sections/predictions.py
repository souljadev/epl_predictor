import sqlite3
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path

# ------------------------------------------------------------
# DB LOADER
# ------------------------------------------------------------
def load_predictions_for_date(db_path: Path, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load ALL predictions for a specific date:
    - model (dc_elo_ensemble_live_...)
    - chatgpt (model_version = "chatgpt")

    Returns raw table; pivoting is done in render().
    """
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
        params=(target_date.strftime("%Y-%m-%d"),),
    )

    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ------------------------------------------------------------
# RENDER PREDICTIONS PAGE
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
    # Split MODEL vs CHATGPT predictions
    # ------------------------------------------------------------
    model_df = df[df["model_version"].str.startswith("dc_elo")].copy()
    gpt_df   = df[df["model_version"] == "chatgpt"].copy()

    # Rename for clarity
    model_df = model_df.rename(columns={
        "score_pred": "model_score"
    })
    gpt_df = gpt_df.rename(columns={
        "chatgpt_pred": "chatgpt_score"
    })

    # Only keep ChatGPT's score for the merge
    gpt_df = gpt_df[["date","home_team","away_team","chatgpt_score"]]

    # ------------------------------------------------------------
    # Merge model + ChatGPT horizontally → ONE ROW PER MATCH
    # ------------------------------------------------------------
    merged = model_df.merge(
        gpt_df,
        on=["date","home_team","away_team"],
        how="left"
    )

    # ------------------------------------------------------------
    # Build display table
    # ------------------------------------------------------------
    display_df = merged.copy()

    display_df["date"] = display_df["date"].dt.date

    display_df["H Prob"] = display_df["home_win_prob"].apply(lambda x: f"{x*100:.1f}%")
    display_df["D Prob"] = display_df["draw_prob"].apply(lambda x: f"{x*100:.1f}%")
    display_df["A Prob"] = display_df["away_win_prob"].apply(lambda x: f"{x*100:.1f}%")


    display_df = display_df[
        [
            "date",
            "home_team",
            "away_team",
            "H Prob", "D Prob", "A Prob",
            "exp_goals_home",
            "exp_goals_away",
            "exp_total_goals",
            "model_score",
            "chatgpt_score",
            "model_version",
        ]
    ]

    st.dataframe(display_df, use_container_width=True)

    # ------------------------------------------------------------
    # Expanders per match
    # ------------------------------------------------------------
    st.markdown("### Match Details")

    for _, row in merged.iterrows():

        label = (
            f"{row['home_team']} vs {row['away_team']} — "
            f"{row['home_win_prob']*100:.1f}% / "
            f"{row['draw_prob']*100:.1f}% / "
            f"{row['away_win_prob']*100:.1f}%"
        )

        with st.expander(label):

            st.write(f"**Model Version:** `{row['model_version']}`")

            st.write(
                f"**Probabilities** — "
                f"Home: `{row['home_win_prob']:.3f}`, "
                f"Draw: `{row['draw_prob']:.3f}`, "
                f"Away: `{row['away_win_prob']:.3f}`"
            )

            st.write(
                f"**Expected Goals** — "
                f"Home: `{row['exp_goals_home']:.2f}`, "
                f"Away: `{row['exp_goals_away']:.2f}`, "
                f"Total: `{row['exp_total_goals']:.2f}`"
            )

            st.write(f"**Model predicted score:** `{row['model_score']}`")

            if row["chatgpt_score"]:
                st.write(f"**ChatGPT predicted score:** `{row['chatgpt_score']}`")
            else:
                st.write("_No ChatGPT prediction available for this match._")
