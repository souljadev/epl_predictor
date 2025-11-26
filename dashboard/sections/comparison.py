import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss

SEASON_START = pd.Timestamp("2024-08-01")
SEASON_END = pd.Timestamp("2025-06-15")

EPL_TEAMS_2024 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea","Crystal Palace",
    "Everton","Fulham","Ipswich","Leicester","Liverpool","Man City","Man United",
    "Newcastle","Nott'm Forest","Southampton","Tottenham","West Ham","Wolves"
}


def load_results(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, home_team, away_team, FTHG, FTAG, Result
        FROM results
        """,
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


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
            away_win_prob
        FROM predictions
        """,
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def winner_from_goals(hg, ag):
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def compute_metrics(df: pd.DataFrame):
    df = df.copy()
    df["actual"] = df.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)
    df["predicted_class"] = df.apply(
        lambda r: np.argmax(
            [r["home_win_prob"], r["draw_prob"], r["away_win_prob"]]
        ),
        axis=1,
    )
    df["actual_idx"] = df["actual"].map({"H": 0, "D": 1, "A": 2})

    y_true = df["actual_idx"].values
    probs = df[["home_win_prob", "draw_prob", "away_win_prob"]].values

    accuracy = (df["predicted_class"] == df["actual_idx"]).mean()

    # Multiclass Brier & Log-loss need labels specified
    try:
        brier = brier_score_loss(
            y_true,
            probs,
            labels=[0, 1, 2],
        )
    except Exception:
        brier = np.nan

    eps = 1e-12
    probs_clip = np.clip(probs, eps, 1 - eps)
    try:
        ll = log_loss(
            y_true,
            probs_clip,
            labels=[0, 1, 2],
        )
    except Exception:
        ll = np.nan

    return accuracy, brier, ll


def render(db_path: Path):
    st.subheader("Model Comparison — Current EPL Season")

    results = load_results(db_path)
    preds = load_predictions(db_path)

    # Filter by season + team membership
    results = results[
        (results["date"] >= SEASON_START)
        & (results["date"] <= SEASON_END)
        & (results["home_team"].isin(EPL_TEAMS_2024))
        & (results["away_team"].isin(EPL_TEAMS_2024))
    ].copy()

    preds = preds[
        (preds["date"] >= SEASON_START)
        & (preds["date"] <= SEASON_END)
    ].copy()

    merged = results.merge(
        preds,
        on=["date", "home_team", "away_team"],
        how="inner",
    )

    if merged.empty:
        st.warning("No overlapping rows between predictions and results for this season.")
        return

    st.write(f"Matched rows: **{len(merged)}**")

    # Overall metrics
    acc, brier, ll = compute_metrics(merged)

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{acc:.3f}")
    col2.metric("Brier Score", f"{brier:.3f}")
    col3.metric("Log Loss", f"{ll:.3f}")

    # By model_version
    st.markdown("### Metrics by Model Version")

    rows = []
    for mv, df_mv in merged.groupby("model_version"):
        acc_mv, brier_mv, ll_mv = compute_metrics(df_mv)
        rows.append(
            {
                "model_version": mv,
                "n_matches": len(df_mv),
                "accuracy": acc_mv,
                "brier": brier_mv,
                "log_loss": ll_mv,
            }
        )

    mv_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    st.dataframe(mv_df, use_container_width=True)

    # Simple chart: accuracy by model_version
    st.markdown("#### Accuracy by Model Version")
    st.bar_chart(
        mv_df.set_index("model_version")["accuracy"],
        use_container_width=True,
    )
