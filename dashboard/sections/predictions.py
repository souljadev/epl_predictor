import sqlite3
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path

# ------------------------------------------------------------
# WINNER + SCORE HELPERS
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
    except Exception:
        return None


def winner_to_team(label, home, away):
    if label == "H":
        return home
    if label == "A":
        return away
    if label == "D":
        return "Draw"
    return None


def parse_score(score):
    """
    '2-1' -> (2,1)
    returns (None, None) if bad.
    """
    try:
        h, a = map(int, str(score).split("-"))
        return h, a
    except Exception:
        return None, None


# ------------------------------------------------------------
# DB LOADER — ALL ROWS FOR DATE
# ------------------------------------------------------------
def load_predictions_for_date(db_path: Path, target_date: pd.Timestamp) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT
            date,
            home_team,
            away_team,
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
        ORDER BY home_team, away_team, created_at
        """,
        conn,
        params=(target_date.strftime("%Y-%m-%d"),),
    )

    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


# ------------------------------------------------------------
# RENDER PREDICTIONS PAGE
# ------------------------------------------------------------
def render(db_path: Path):
    st.subheader("Upcoming Predictions (Next 2 Matchdays)")

    # --------------------------------------------------------
    # Find next 2 matchdays with predictions available
    # --------------------------------------------------------
    conn = sqlite3.connect(db_path)
    df_dates = pd.read_sql_query(
        """
        SELECT DISTINCT date 
        FROM predictions
        WHERE date >= DATE('now')
        ORDER BY date ASC
        """,
        conn,
    )
    conn.close()

    if df_dates.empty:
        st.info("No upcoming predictions found in the database.")
        return

    # Get next 2 unique match dates
    upcoming_dates = (
        df_dates["date"]
        .drop_duplicates()
        .sort_values()
        .head(2)
        .tolist()
    )

    st.markdown(f"### Showing matchdays: {', '.join(upcoming_dates)}")

    # --------------------------------------------------------
    # Iterate through the next two matchdays
    # --------------------------------------------------------
    for d in upcoming_dates:
        date_ts = pd.to_datetime(d)

        st.markdown(f"## Matchday — {date_ts.date()}")

        df = load_predictions_for_date(db_path, date_ts)

        if df.empty:
            st.info(f"No predictions for {date_ts.date()}.")
            continue

        # ----------------------------------------------------
        # SAME logic as before — SPLIT / DEDUPE / MERGE / DISPLAY
        # (your existing block goes here unchanged)
        # ----------------------------------------------------

        # --- 1. Split model vs ChatGPT ---
        model_df = df[df["score_pred"].notna()].copy()
        gpt_df   = df[df["chatgpt_pred"].notna()].copy()

        # --- 2. Smart dedupe model versions ---
        def pick_best_model_row(group: pd.DataFrame) -> pd.Series:
            best_idx = None
            best_err = None
            for idx, row in group.iterrows():
                h, a = parse_score(row["score_pred"])
                if h is None or a is None:
                    continue
                lamH = float(row["exp_goals_home"])
                lamA = float(row["exp_goals_away"])
                err = (h - lamH)**2 + (a - lamA)**2
                if best_err is None or err < best_err:
                    best_err = err
                    best_idx = idx
            return group.loc[best_idx] if best_idx is not None else group.iloc[0]

        if not model_df.empty:
            model_df = (
                model_df.groupby(["date", "home_team", "away_team"], as_index=False)
                .apply(pick_best_model_row)
                .reset_index(drop=True)
            )

        # --- 3. ChatGPT: take last per match ---
        if not gpt_df.empty:
            gpt_df = (
                gpt_df.sort_values("created_at")
                .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
            )

        model_df = model_df.rename(columns={"score_pred": "model_score"})
        gpt_df["chatgpt_score"] = gpt_df["chatgpt_pred"]

        gpt_df = gpt_df[["date","home_team","away_team","chatgpt_score"]]

        # --- 4. Merge ---
        merged = model_df.merge(
            gpt_df,
            on=["date","home_team","away_team"],
            how="left",
        )

        if merged.empty:
            st.info(f"No merged model/ChatGPT predictions for {date_ts.date()}.")
            continue

        # --- 5. Build display frame ---
        display_df = merged.copy()

        display_df["date"] = display_df["date"].dt.date
        display_df["exp_goals_home"] = display_df["exp_goals_home"].round(0).astype(int)
        display_df["exp_goals_away"] = display_df["exp_goals_away"].round(0).astype(int)
        display_df["exp_total_goals"] = display_df["exp_total_goals"].round(0).astype(int)

        display_df["H Prob"] = display_df["home_win_prob"].apply(lambda x: f"{x*100:.1f}%")
        display_df["D Prob"] = display_df["draw_prob"].apply(lambda x: f"{x*100:.1f}%")
        display_df["A Prob"] = display_df["away_win_prob"].apply(lambda x: f"{x*100:.1f}%")

        display_df["model_winner"] = display_df["model_score"].apply(winner_from_score)
        display_df["chatgpt_winner"] = display_df["chatgpt_score"].apply(winner_from_score)

        display_df["model_winner_team"] = display_df.apply(
            lambda r: winner_to_team(r["model_winner"], r["home_team"], r["away_team"]),
            axis=1,
        )
        display_df["chatgpt_winner_team"] = display_df.apply(
            lambda r: winner_to_team(r["chatgpt_winner"], r["home_team"], r["away_team"]),
            axis=1,
        )

        # --- 6. Highlighting ---
        def highlight_row(row):
            styles = [""] * len(row)
            idx_model_score = row.index.get_loc("model_score")
            idx_chat_score = row.index.get_loc("chatgpt_score")
            idx_model_win = row.index.get_loc("model_winner_team")
            idx_chat_win = row.index.get_loc("chatgpt_winner_team")

            if (
                pd.notna(row["model_score"])
                and pd.notna(row["chatgpt_score"])
                and row["model_score"] == row["chatgpt_score"]
            ):
                styles[idx_model_score] = "background-color: yellow;"
                styles[idx_chat_score] = "background-color: yellow;"

            if (
                pd.notna(row["model_winner_team"])
                and pd.notna(row["chatgpt_winner_team"])
                and row["model_winner_team"] == row["chatgpt_winner_team"]
            ):
                styles[idx_model_win] = "background-color: lightgreen;"
                styles[idx_chat_win] = "background-color: lightgreen;"

            return styles

        display_df = display_df[
            [
                "date",
                "home_team",
                "away_team",
                "H Prob","D Prob","A Prob",
                "exp_goals_home","exp_goals_away","exp_total_goals",
                "model_score","model_winner_team",
                "chatgpt_score","chatgpt_winner_team",
            ]
        ]

        st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)

        
