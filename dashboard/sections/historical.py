import sqlite3
from pathlib import Path
from datetime import timedelta, date
import numpy as pd
import pandas as pd
import streamlit as st


def load_historical(db_path: Path, days_back: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            r.date,
            r.home_team,
            r.away_team,
            r.FTHG,
            r.FTAG,
            r.Result,
            p.model_version,
            p.home_win_prob,
            p.draw_prob,
            p.away_win_prob
        FROM results r
        LEFT JOIN predictions p
          ON r.date = p.date
         AND r.home_team = p.home_team
         AND r.away_team = p.away_team
        ORDER BY r.date DESC
        """,
        conn,
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp(date.today() - timedelta(days=days_back))
    df = df[df["date"] >= cutoff]

    return df


def winner_from_goals(hg, ag):
    if pd.isna(hg) or pd.isna(ag):
        return None
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def render(db_path: Path):
    st.subheader("Historical Performance")

    days_back = st.slider("Lookback window (days)", min_value=7, max_value=365, value=60)

    df = load_historical(db_path, days_back=days_back)
    if df.empty:
        st.info("No historical data found in that window.")
        return

    df["actual"] = df.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)

    def predicted_label(row):
        if pd.isna(row["home_win_prob"]):
            return None
        probs = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        idx = int(pd.Series(probs).idxmax())
        return {0: "H", 1: "D", 2: "A"}[idx]

    df["predicted"] = df.apply(predicted_label, axis=1)
    df["correct"] = df["actual"] == df["predicted"]

    display_df = df[
        [
            "date",
            "home_team",
            "away_team",
            "FTHG",
            "FTAG",
            "Result",
            "model_version",
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
            "actual",
            "predicted",
            "correct",
        ]
    ]

    st.dataframe(display_df, use_container_width=True)

    accuracy = df["correct"].mean(skipna=True)
    st.markdown(f"**Hit rate over window:** `{accuracy:.3f}`")

    # Simple aggregate: correct vs incorrect count
    summary = df["correct"].value_counts(dropna=True).rename(index={True: "Correct", False: "Incorrect"})
    st.bar_chart(summary)
