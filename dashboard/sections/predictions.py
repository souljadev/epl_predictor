import sqlite3
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st


def load_predictions_for_date(db_path: Path, target_date: pd.Timestamp) -> pd.DataFrame:
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
            chatgpt_pred
        FROM predictions
        WHERE date = ?
        ORDER BY date, home_team, away_team
        """,
        conn,
        params=(target_date.strftime("%Y-%m-%d"),),
    )
    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


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

    # Nicely formatted table
    display_df = df.copy()
    display_df["H Prob"] = (display_df["home_win_prob"] * 100).round(1)
    display_df["D Prob"] = (display_df["draw_prob"] * 100).round(1)
    display_df["A Prob"] = (display_df["away_win_prob"] * 100).round(1)

    display_df = display_df[
        [
            "date",
            "home_team",
            "away_team",
            "model_version",
            "H Prob",
            "D Prob",
            "A Prob",
            "exp_goals_home",
            "exp_goals_away",
            "exp_total_goals",
            "score_pred",
            "chatgpt_pred",
        ]
    ]

    st.dataframe(display_df, use_container_width=True)

    st.markdown("#### Per-match details")
    for _, row in df.iterrows():
        with st.expander(
            f"{row['home_team']} vs {row['away_team']} — "
            f"{(row['home_win_prob']*100):.1f}% / "
            f"{(row['draw_prob']*100):.1f}% / "
            f"{(row['away_win_prob']*100):.1f}%"
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
            st.write(f"**Most likely scoreline (sampled):** `{row['score_pred']}`")
            if row["chatgpt_pred"]:
                st.write("**ChatGPT prediction:**")
                st.write(row["chatgpt_pred"])
            else:
                st.write("_No ChatGPT overlay stored for this match._")
