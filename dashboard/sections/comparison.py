import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

SEASON_START = pd.Timestamp("2024-08-01")
SEASON_END   = pd.Timestamp("2025-06-15")

EPL_TEAMS_2024 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea",
    "Crystal Palace","Everton","Fulham","Ipswich","Leicester",
    "Liverpool","Man City","Man United","Newcastle",
    "Nott'm Forest","Southampton","Tottenham","West Ham","Wolves"
}

TEAM_FIX = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester Utd": "Man United",
    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
}

# ------------------------------------------------------------
# DB LOADERS
# ------------------------------------------------------------
def load_results(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, home_team, away_team, FTHG, FTAG
        FROM results
        """,
        conn,
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def load_predictions(db_path: Path) -> pd.DataFrame:
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
            score_pred,
            chatgpt_pred
        FROM predictions
        """,
        conn,
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def load_gemini_predictions(db_path: Path) -> pd.DataFrame:
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
        """,
        conn,
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def winner_from_score(score):
    try:
        h, a = map(int, score.split("-"))
        if h > a:
            return "H"
        if h < a:
            return "A"
        return "D"
    except Exception:
        return None


def accuracy_pct(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return 100 * valid.mean()



# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
def render(db_path: Path):
    st.subheader("Model vs ChatGPT vs Gemini vs Actual — Past 30 Days")

    today = pd.Timestamp.today().normalize()
    window_start = today - pd.Timedelta(days=30)

    preds = load_predictions(db_path)
    results = load_results(db_path)
    gemini = load_gemini_predictions(db_path)

    # Normalize team names
    for df in (preds, results, gemini):
        df["home_team"] = df["home_team"].replace(TEAM_FIX)
        df["away_team"] = df["away_team"].replace(TEAM_FIX)

    # EPL only
    preds = preds[
        preds["home_team"].isin(EPL_TEAMS_2024)
        & preds["away_team"].isin(EPL_TEAMS_2024)
    ]

    # Date filter
    preds   = preds[preds["date"].between(window_start, today)]
    results = results[results["date"].between(window_start, today)]
    gemini  = gemini[gemini["date"].between(window_start, today)]

    if preds.empty:
        st.info("No EPL matches found in the past 30 days.")
        return

    # --------------------------------------------------------
    # SPLIT MODEL VS CHATGPT
    # --------------------------------------------------------
    model_df = preds[preds["model_version"].str.startswith("dc_elo")].copy()
    gpt_df   = preds[preds["model_version"] == "chatgpt"].copy()

    model_df = model_df.rename(columns={"score_pred": "model_score"})
    gpt_df   = gpt_df.rename(columns={"chatgpt_pred": "chatgpt_score"})

    gpt_df = (
        gpt_df[["date","home_team","away_team","chatgpt_score"]]
        .sort_values("date")
        .groupby(["date","home_team","away_team"], as_index=False)
        .last()
    )

    # --------------------------------------------------------
    # GEMINI (LATEST PER MATCH)
    # --------------------------------------------------------
    gemini_df = (
        gemini.sort_values("created_at")
        .groupby(["date","home_team","away_team"], as_index=False)
        .last()
    )

    # --------------------------------------------------------
    # DEDUPE MODEL VERSIONS
    # --------------------------------------------------------
    def pick_best_model_row(group: pd.DataFrame) -> pd.Series:
        probs = group[["home_win_prob","draw_prob","away_win_prob"]].values
        idx = np.argmax(np.max(probs, axis=1))
        return group.iloc[idx]

    model_df = (
        model_df
        .groupby(["date","home_team","away_team"], as_index=False)
        .apply(pick_best_model_row, include_groups=False)
        .reset_index(drop=True)
    )

    results = results.drop_duplicates(
        subset=["date","home_team","away_team"],
        keep="last"
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------
    merged = (
        model_df
        .merge(gpt_df, on=["date","home_team","away_team"], how="left")
        .merge(
            gemini_df[["date","home_team","away_team","gemini_score"]],
            on=["date","home_team","away_team"],
            how="left",
        )
        .merge(
            results[["date","home_team","away_team","FTHG","FTAG"]],
            on=["date","home_team","away_team"],
            how="left",
        )
    )

    merged = merged.dropna(subset=["FTHG", "FTAG"])
    # --------------------------------------------------------
    # PREDICTION MADE FLAGS (ISOLATED, REUSABLE)
    # --------------------------------------------------------
    model_pred_made  = merged["model_score"].notna()
    gpt_pred_made    = merged["chatgpt_score"].notna()
    gemini_pred_made = merged["gemini_score"].notna()


    # --------------------------------------------------------
    # WINNERS + SCORES
    # --------------------------------------------------------
    def winner_from_probs(row):
        arr = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        return {0:"H",1:"D",2:"A"}[int(np.argmax(arr))]

    merged["model_winner"]  = merged.apply(winner_from_probs, axis=1)
    merged["gpt_winner"]    = merged["chatgpt_score"].apply(winner_from_score)
    merged["gemini_winner"] = merged["gemini_score"].apply(winner_from_score)

    merged["actual_score"] = (
        merged["FTHG"].astype(int).astype(str)
        + "-"
        + merged["FTAG"].astype(int).astype(str)
    )

    merged["actual_winner"] = np.where(
        merged["FTHG"] > merged["FTAG"], merged["home_team"],
        np.where(merged["FTHG"] < merged["FTAG"], merged["away_team"], "Draw")
    )

    merged["model_winner_team"] = np.where(
        merged["model_winner"] == "H", merged["home_team"],
        np.where(merged["model_winner"] == "A", merged["away_team"], "Draw")
    )

    merged["gpt_winner_team"] = np.where(
        merged["gpt_winner"] == "H", merged["home_team"],
        np.where(merged["gpt_winner"] == "A", merged["away_team"], "Draw")
    )

    merged["gemini_winner_team"] = np.where(
        merged["gemini_winner"] == "H", merged["home_team"],
        np.where(
            merged["gemini_winner"] == "A", merged["away_team"],
            np.where(merged["gemini_winner"] == "D", "Draw", np.nan)
        )
    )

    # --------------------------------------------------------
    # CORRECTNESS
    # --------------------------------------------------------
    merged["model_correct"]  = merged["model_winner_team"]  == merged["actual_winner"]
    merged["gpt_correct"]    = merged["gpt_winner_team"]    == merged["actual_winner"]
    merged["gemini_correct"] = merged["gemini_winner_team"] == merged["actual_winner"]

    merged["model_score_correct"]  = merged["model_score"]   == merged["actual_score"]
    merged["gpt_score_correct"]    = merged["chatgpt_score"] == merged["actual_score"]
    merged["gemini_score_correct"] = merged["gemini_score"]  == merged["actual_score"]

    # --------------------------------------------------------
    # DRAW ACCURACY MASKS (ISOLATED)
    # --------------------------------------------------------
    actual_draw = merged["actual_winner"] == "Draw"

    model_pred_draw  = merged["model_winner_team"]  == "Draw"
    gpt_pred_draw    = merged["gpt_winner_team"]    == "Draw"
    gemini_pred_draw = merged["gemini_winner_team"] == "Draw"

    def draw_accuracy(pred_draw, pred_made):
        mask = actual_draw & pred_made
        if mask.sum() == 0:
            return np.nan
        return 100 * (pred_draw & actual_draw & pred_made).sum() / mask.sum()

    # --------------------------------------------------------
    # ADDITIVE: ACCURACY SUMMARY (NEW)
    # --------------------------------------------------------
    accuracy_df = pd.DataFrame({
        "Source": ["Model", "ChatGPT", "Gemini"],
        "Winner Accuracy (%)": [
            accuracy_pct(merged["model_correct"]),
            accuracy_pct(merged["gpt_correct"]),
            accuracy_pct(merged["gemini_correct"]),
        ],
        "Exact Score Accuracy (%)": [
            accuracy_pct(merged["model_score_correct"]),
            accuracy_pct(merged["gpt_score_correct"]),
            accuracy_pct(merged["gemini_score_correct"]),
        ],
        "Draw Accuracy (%)": [
            draw_accuracy(model_pred_draw,  model_pred_made),
            draw_accuracy(gpt_pred_draw,    gpt_pred_made),
            draw_accuracy(gemini_pred_draw, gemini_pred_made),
        ],
        "Predictions Made": [
            model_pred_made.sum(),
            gpt_pred_made.sum(),
            gemini_pred_made.sum(),
        ],
    })

    st.markdown("### Accuracy Summary (Past 30 Days)")
    st.dataframe(accuracy_df.round(2), use_container_width=True)

    # --------------------------------------------------------
    # FINAL DISPLAY (UNCHANGED)
    # --------------------------------------------------------
    merged["date"] = merged["date"].dt.date
    merged = merged.sort_values("date", ascending=False)

    display_df = merged[
        [
            "date",
            "home_team",
            "away_team",
            "actual_score",
            "actual_winner",
            "model_score",
            "model_winner_team",
            "model_correct",
            "model_score_correct",
            "chatgpt_score",
            "gpt_winner_team",
            "gpt_correct",
            "gpt_score_correct",
            "gemini_score",
            "gemini_winner_team",
            "gemini_correct",
            "gemini_score_correct",
        ]
    ]

    def highlight(val):
        if val is True:
            return "background-color: #c6efce;"
        if val is False:
            return "background-color: #ffc7ce;"
        return ""

    st.dataframe(
        display_df.style.map(
            highlight,
            subset=[
                "model_correct",
                "gpt_correct",
                "gemini_correct",
                "model_score_correct",
                "gpt_score_correct",
                "gemini_score_correct",
            ],
        ),
        use_container_width=True,
    )
