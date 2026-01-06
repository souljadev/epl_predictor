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
    try:
        h, a = map(int, str(score).split("-"))
        return h, a
    except Exception:
        return None, None


# ------------------------------------------------------------
# LOAD MODEL + CHATGPT PREDICTIONS
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
        WHERE DATE(date) = ?
        ORDER BY home_team, away_team, created_at
        """,
        conn,
        params=(target_date.strftime("%Y-%m-%d"),),
    )

    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


# ------------------------------------------------------------
# LOAD GEMINI PREDICTIONS
# ------------------------------------------------------------
def load_gemini_predictions_for_date(
    db_path: Path,
    target_date: pd.Timestamp
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT
            match_date AS date,
            home_team,
            away_team,
            predicted_score AS gemini_score,
            created_at
        FROM gemini_predictions
        WHERE DATE(match_date) = ?
        ORDER BY created_at
        """,
        conn,
        params=(target_date.strftime("%Y-%m-%d"),),
    )

    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


# ------------------------------------------------------------
# RENDER DASHBOARD
# ------------------------------------------------------------
def render(db_path: Path):
    st.subheader("Upcoming Predictions (Next 2 Matchdays)")

    # --------------------------------------------------------
    # GET UPCOMING MATCHDAYS FROM FIXTURES
    # --------------------------------------------------------
    conn = sqlite3.connect(db_path)
    df_dates = pd.read_sql_query(
        """
        SELECT DISTINCT Date AS date
        FROM fixtures
        WHERE Date >= DATE('now')
        ORDER BY Date ASC
        """,
        conn,
    )
    conn.close()

    if df_dates.empty:
        st.info("No upcoming fixtures found.")
        return

    upcoming_dates = (
        df_dates["date"]
        .drop_duplicates()
        .sort_values()
        .head(2)
        .tolist()
    )

    st.markdown(f"### Showing matchdays: {', '.join(upcoming_dates)}")

    # --------------------------------------------------------
    # ITERATE MATCHDAYS
    # --------------------------------------------------------
    for d in upcoming_dates:
        date_ts = pd.to_datetime(d)
        st.markdown(f"## Matchday — {date_ts.date()}")

        df = load_predictions_for_date(db_path, date_ts)
        gemini_df = load_gemini_predictions_for_date(db_path, date_ts)

        if df.empty:
            st.info(f"No predictions for {date_ts.date()}.")
            continue

        # -------------------------
        # SPLIT SOURCES
        # -------------------------
        model_df = df[df["score_pred"].notna()].copy()
        gpt_df = df[df["chatgpt_pred"].notna()].copy()

        # -------------------------
        # MODEL: BEST VERSION
        # -------------------------
        def pick_best_model_row(group):
            best_idx = None
            best_err = None
            for idx, row in group.iterrows():
                h, a = parse_score(row["score_pred"])
                if h is None:
                    continue
                err = (h - row["exp_goals_home"]) ** 2 + (a - row["exp_goals_away"]) ** 2
                if best_err is None or err < best_err:
                    best_err = err
                    best_idx = idx
            return group.loc[best_idx] if best_idx is not None else group.iloc[0]

        if not model_df.empty:
            model_df = (
                model_df
                .groupby(["date", "home_team", "away_team"], as_index=False)
                .apply(pick_best_model_row, include_groups=False)
                .reset_index(drop=True)
            )

        # -------------------------
        # CHATGPT: MOST RECENT
        # -------------------------
        if not gpt_df.empty:
            gpt_df = (
                gpt_df.sort_values("created_at")
                .drop_duplicates(
                    subset=["date", "home_team", "away_team"],
                    keep="last"
                )
            )

        # -------------------------
        # GEMINI: MOST RECENT
        # -------------------------
        if not gemini_df.empty:
            gemini_df = (
                gemini_df.sort_values("created_at")
                .drop_duplicates(
                    subset=["date", "home_team", "away_team"],
                    keep="last"
                )
            )

        # -------------------------
        # PREP FOR MERGE
        # -------------------------
        model_df = model_df.rename(columns={"score_pred": "model_score"})

        gpt_df["chatgpt_score"] = gpt_df["chatgpt_pred"]
        gpt_df = gpt_df[["date", "home_team", "away_team", "chatgpt_score"]]

        gemini_df = gemini_df[["date", "home_team", "away_team", "gemini_score"]]

        # -------------------------
        # MERGE ALL
        # -------------------------
        merged = (
            model_df
            .merge(gpt_df, on=["date", "home_team", "away_team"], how="left")
            .merge(gemini_df, on=["date", "home_team", "away_team"], how="left")
        )

        if merged.empty:
            st.info("No merged predictions.")
            continue

        # -------------------------
        # DISPLAY PREP
        # -------------------------
        display_df = merged.copy()
        display_df["date"] = display_df["date"].dt.date

        # display_df["exp_goals_home"] = display_df["exp_goals_home"].round(0).astype(int)
        # display_df["exp_goals_away"] = display_df["exp_goals_away"].round(0).astype(int)
        # display_df["exp_total_goals"] = display_df["exp_total_goals"].round(0).astype(int)

        display_df["H Prob"] = display_df["home_win_prob"].apply(lambda x: f"{x*100:.1f}%")
        display_df["D Prob"] = display_df["draw_prob"].apply(lambda x: f"{x*100:.1f}%")
        display_df["A Prob"] = display_df["away_win_prob"].apply(lambda x: f"{x*100:.1f}%")

        # -------------------------
        # WINNERS
        # -------------------------
        display_df["model_winner_team"] = display_df.apply(
            lambda r: winner_to_team(
                winner_from_score(r["model_score"]),
                r["home_team"],
                r["away_team"]
            ),
            axis=1,
        )

        display_df["chatgpt_winner_team"] = display_df.apply(
            lambda r: winner_to_team(
                winner_from_score(r["chatgpt_score"]),
                r["home_team"],
                r["away_team"]
            ),
            axis=1,
        )

        display_df["gemini_winner_team"] = display_df.apply(
            lambda r: winner_to_team(
                winner_from_score(r["gemini_score"]),
                r["home_team"],
                r["away_team"]
            ),
            axis=1,
        )

        # -------------------------
        # HIGHLIGHTING
        # -------------------------
        def highlight_row(row):
            styles = [""] * len(row)

            cols = row.index.tolist()
            i = lambda c: cols.index(c)

            if row["model_score"] == row["chatgpt_score"]:
                styles[i("model_score")] = styles[i("chatgpt_score")] = "background-color:#ffe599;"

            if row["model_score"] == row["gemini_score"]:
                styles[i("model_score")] = styles[i("gemini_score")] = "background-color:#ffe599;"

            if row["model_winner_team"] == row["chatgpt_winner_team"]:
                styles[i("model_winner_team")] = styles[i("chatgpt_winner_team")] = "background-color:#d9ead3;"

            if row["model_winner_team"] == row["gemini_winner_team"]:
                styles[i("model_winner_team")] = styles[i("gemini_winner_team")] = "background-color:#d9ead3;"

            return styles

        display_df = display_df[
            [
                "date",
                "home_team",
                "away_team",
                "H Prob", "D Prob", "A Prob",
                # "exp_goals_home", "exp_goals_away", "exp_total_goals",
                "model_score", "model_winner_team",
                "chatgpt_score", "chatgpt_winner_team",
                "gemini_score", "gemini_winner_team",
            ]
        ]

        st.dataframe(
            display_df.style.apply(highlight_row, axis=1),
            use_container_width=True,
        )
