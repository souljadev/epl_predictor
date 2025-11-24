import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # soccer_agent_local/
DB_PATH = ROOT / "data" / "soccer_agent.db"
EVAL_DIR = ROOT / "models" / "evaluation"
CHATGPT_VS_MODEL_CSV = EVAL_DIR / "chatgpt_vs_model.csv"
BACKTEST_METRICS_CSV = EVAL_DIR / "metrics_backtests_summary.csv"

# ---------------------------------------------------------------------
# STREAMLIT CONFIG (must be FIRST)
# ---------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="EPL Agent Dashboard",
    page_icon="⚽",
)

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
@contextmanager
def get_conn():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_latest_run_id(table: str = "predictions_model") -> str | None:
    """Return latest run_id based on run_ts from a predictions table."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT run_id
            FROM {table}
            WHERE run_id IS NOT NULL
            ORDER BY run_ts DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------
# Load predictions (model + ChatGPT ensemble) for upcoming fixtures
# ---------------------------------------------------------------------
def load_predictions_for_latest_run(days_ahead: int = 7) -> pd.DataFrame:
    """
    Load upcoming fixtures (today..today+days_ahead) with:
      - model probabilities + xG
      - ChatGPT ensemble score + confidence (if present)

    Reads directly from SQLite: fixtures, predictions_model, predictions_chatgpt.
    ChatGPT is filtered to model_name='gpt-ensemble'.
    """
    run_id = get_latest_run_id("predictions_model")
    if not run_id:
        return pd.DataFrame()

    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    with get_conn() as conn:
        query = """
        WITH pm_latest AS (
            SELECT *
            FROM predictions_model pm
            WHERE run_id = ?
              AND pm.run_ts = (
                  SELECT MAX(run_ts)
                  FROM predictions_model
                  WHERE fixture_id = pm.fixture_id
                    AND run_id = ?
              )
        ),
        pc_latest AS (
            SELECT *
            FROM predictions_chatgpt pc
            WHERE run_id = ?
              AND model_name = 'gpt-ensemble'
              AND pc.run_ts = (
                  SELECT MAX(run_ts)
                  FROM predictions_chatgpt
                  WHERE fixture_id = pc.fixture_id
                    AND run_id = ?
                    AND model_name = 'gpt-ensemble'
              )
        )
        SELECT
            f.date            AS date,
            f.home_team       AS HomeTeam,
            f.away_team       AS AwayTeam,
            pm_latest.pH      AS pH,
            pm_latest.pD      AS pD,
            pm_latest.pA      AS pA,
            pm_latest.exp_home_goals  AS ExpHomeGoals,
            pm_latest.exp_away_goals  AS ExpAwayGoals,
            pm_latest.exp_total_goals AS ExpTotalGoals,
            pc_latest.predicted_score AS ChatGPTScore,
            pc_latest.confidence      AS ChatGPTConfidence
        FROM fixtures f
        JOIN pm_latest
          ON pm_latest.fixture_id = f.fixture_id
        LEFT JOIN pc_latest
          ON pc_latest.fixture_id = f.fixture_id
        WHERE f.date >= ? AND f.date <= ?
        ORDER BY f.date, f.home_team, f.away_team
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(
                run_id,
                run_id,
                run_id,
                run_id,
                today.isoformat(),
                cutoff.isoformat(),
            ),
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------
# SAFE loader for chatgpt_vs_model.csv (DB-driven comparison artifact)
# ---------------------------------------------------------------------
def load_chatgpt_vs_model() -> pd.DataFrame:
    if not CHATGPT_VS_MODEL_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(CHATGPT_VS_MODEL_CSV)

    # Normalize date column
    possible_cols = ["Date", "date", "match_date"]
    for col in possible_cols:
        if col in df.columns:
            df["Date"] = pd.to_datetime(df[col], errors="coerce")
            break
    else:
        df["Date"] = pd.NaT

    return df


# ---------------------------------------------------------------------
# Backtest metrics loader
# ---------------------------------------------------------------------
def load_backtest_metrics() -> pd.DataFrame:
    if not BACKTEST_METRICS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(BACKTEST_METRICS_CSV)


# ---------------------------------------------------------------------
# UI: Predictions (Upcoming Fixtures)
# ---------------------------------------------------------------------
def section_predictions():
    st.header("🔮 Upcoming Fixtures – Model & ChatGPT (Ensemble) Predictions")

    df = load_predictions_for_latest_run(days_ahead=7)
    if df.empty:
        st.info(
            "No upcoming fixtures with predictions found.\n\n"
            "Make sure:\n"
            " - The agent has run\n"
            " - Model predictions exist in DB\n"
            " - ChatGPT predictions (ensemble) have been generated"
        )
        return

    df_display = df.copy()

    # ----------------------------------------------------------
    # Determine Model Winner (team name)
    # ----------------------------------------------------------
    def model_winner(row):
        probs = [row["pH"], row["pD"], row["pA"]]
        if any(pd.isna(probs)):
            return ""
        idx = int(pd.Series(probs).idxmax())
        if idx == 0:
            return row["HomeTeam"]
        elif idx == 1:
            return "Draw"
        else:
            return row["AwayTeam"]

    df_display["Model Winner"] = df_display.apply(model_winner, axis=1)

    # ----------------------------------------------------------
    # Model Score (rounded xG → scoreline)
    # ----------------------------------------------------------
    def model_score(row):
        if pd.isna(row["ExpHomeGoals"]) or pd.isna(row["ExpAwayGoals"]):
            return ""
        h = int(round(row["ExpHomeGoals"]))
        a = int(round(row["ExpAwayGoals"]))
        return f"{h}-{a}"

    df_display["Model Score"] = df_display.apply(model_score, axis=1)

    # ----------------------------------------------------------
    # Determine ChatGPT Winner from score string (team name)
    # ----------------------------------------------------------
    def chatgpt_winner(score, home, away):
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

    df_display["ChatGPT Winner"] = df_display.apply(
        lambda r: chatgpt_winner(r.get("ChatGPTScore", ""), r["HomeTeam"], r["AwayTeam"]),
        axis=1,
    )

    # ----------------------------------------------------------
    # Winner Match Indicator
    # ----------------------------------------------------------
    def match_indicator(row):
        mw = row["Model Winner"]
        cw = row["ChatGPT Winner"]

        if cw == "":
            return "⚪ No Data"

        if mw == cw:
            return "🟩 Match"

        if mw == "Draw" or cw == "Draw":
            return "🟨 Partial"

        return "🟥 No Match"

    df_display["Winner Match?"] = df_display.apply(match_indicator, axis=1)

    # ----------------------------------------------------------
    # Agreement Stats (Upcoming Fixtures)
    # ----------------------------------------------------------
    counts = df_display["Winner Match?"].value_counts(dropna=False)
    total_with_data = len(df_display[df_display["Winner Match?"] != "⚪ No Data"]) or 1

    full_match = counts.get("🟩 Match", 0)
    partial = counts.get("🟨 Partial", 0)
    no_match = counts.get("🟥 No Match", 0)
    no_data = counts.get("⚪ No Data", 0)

    col_stats = st.columns(4)
    col_stats[0].metric(
        "Full Match (same winner)",
        f"{full_match} / {total_with_data}",
        f"{full_match / total_with_data:.0%}",
    )
    col_stats[1].metric(
        "Partial (Draw vs Win)",
        f"{partial} / {total_with_data}",
        f"{partial / total_with_data:.0%}",
    )
    col_stats[2].metric(
        "No Match",
        f"{no_match} / {total_with_data}",
        f"{no_match / total_with_data:.0%}",
    )
    col_stats[3].metric(
        "No ChatGPT Data",
        f"{no_data}",
    )

    # ----------------------------------------------------------
    # Prepare table for display
    # ----------------------------------------------------------
    df_display.rename(
        columns={
            "date": "Date",
            "HomeTeam": "Home",
            "AwayTeam": "Away",
            "pH": "Prob Home Win",
            "pD": "Prob Draw",
            "pA": "Prob Away Win",
            "ChatGPTScore": "ChatGPT Score",
            "ChatGPTConfidence": "ChatGPT Conf.",
        },
        inplace=True,
    )

    df_display["Date"] = df_display["Date"].astype(str)

    # Convert probabilities to integer %
    for col in ["Prob Home Win", "Prob Draw", "Prob Away Win"]:
        df_display[col] = (df_display[col] * 100).round().astype(int).astype(str) + "%"

    # Format ChatGPT confidence: assume 0–1 float in DB
    if "ChatGPT Conf." in df_display.columns:
        df_display["ChatGPT Conf."] = df_display["ChatGPT Conf."].apply(
            lambda x: f"{int(round(float(x) * 100))}%" if pd.notnull(x) else ""
        )

    # Only keep the most readable columns for the table
    cols_order = [
        "Date",
        "Home",
        "Away",
        "Prob Home Win",
        "Prob Draw",
        "Prob Away Win",
        "Model Winner",
        "Model Score",
        "ChatGPT Score",
        "ChatGPT Winner",
        "ChatGPT Conf.",
        "Winner Match?",
    ]
    cols_order = [c for c in cols_order if c in df_display.columns]
    df_display = df_display[cols_order]

    # ----------------------------------------------------------
    # Styling function for color-coded Winner Match column
    # ----------------------------------------------------------
    def style_match(val):
        if isinstance(val, str):
            if "🟩" in val:
                return "color: green; font-weight: bold;"
            if "🟥" in val:
                return "color: red; font-weight: bold;"
            if "🟨" in val:
                return "color: goldenrod; font-weight: bold;"
            if "⚪" in val:
                return "color: grey;"
        return ""

    styled_df = df_display.style.applymap(
        style_match,
        subset=["Winner Match?"],
    )

    st.subheader("Predictions for Next 7 Days")
    st.dataframe(styled_df, use_container_width=True)


# ---------------------------------------------------------------------
# UI: Model vs ChatGPT comparison (Recent Results – current season only)
# ---------------------------------------------------------------------
def section_comparison():
    st.header("⚔️ Model vs ChatGPT – Recent Results (Current Season)")

    df = load_chatgpt_vs_model()
    if df.empty:
        st.info("No comparison data available yet. Run compare_models.py first.")
        return

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        st.info("No valid dated rows in comparison data.")
        return

    # Determine season_start_year (Jul–Jun) and filter to MOST RECENT season
    season_year = df["Date"].dt.year.where(df["Date"].dt.month >= 7, df["Date"].dt.year - 1)
    current_season = season_year.max()
    df = df[season_year == current_season].copy()

    if df.empty:
        st.info("No rows for the current season in comparison data.")
        return

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    st.caption(f"Showing matches from current season starting {current_season}-{current_season+1} "
               f" (range: {min_date} → {max_date})")

    # ----------------------------------------------------------
    # ChatGPT variant selector
    # ----------------------------------------------------------
    variant_label = st.selectbox(
        "ChatGPT variant for comparison",
        [
            "gpt-ensemble (default)",
            "gpt-4o-mini",
            "gpt-5.1",
        ],
        index=0,
    )

    if variant_label.startswith("gpt-ensemble"):
        chat_winner_col = "chat_ens_winner"
        chat_score_col = "chat_ens_score"
        chat_correct_win_col = "correct_winner_ens"
        chat_correct_score_col = "correct_score_ens"
        variant_name = "Ensemble"
    elif variant_label.startswith("gpt-4o-mini"):
        chat_winner_col = "chat_mini_winner"
        chat_score_col = "chat_mini_score"
        chat_correct_win_col = "correct_winner_mini"
        chat_correct_score_col = "correct_score_mini"
        variant_name = "GPT-4o-mini"
    else:
        chat_winner_col = "chat_51_winner"
        chat_score_col = "chat_51_score"
        chat_correct_win_col = "correct_winner_51"
        chat_correct_score_col = "correct_score_51"
        variant_name = "GPT-5.1"

    # ----------------------------------------------------------
    # Summary metrics (overall accuracy, current season)
    # ----------------------------------------------------------
    model_acc = df["correct_winner_model"].mean() if "correct_winner_model" in df else None
    chat_acc = df[chat_correct_win_col].mean() if chat_correct_win_col in df else None

    model_score_acc = df["correct_score_model"].mean() if "correct_score_model" in df else None
    chat_score_acc = df[chat_correct_score_col].mean() if chat_correct_score_col in df else None

    avg_xg_err = df["model_xg_error"].mean() if "model_xg_error" in df else None

    cols = st.columns(5)
    cols[0].metric(
        "Model – Winner Accuracy",
        f"{model_acc:.1%}" if model_acc is not None else "n/a",
    )
    cols[1].metric(
        f"{variant_name} – Winner Accuracy",
        f"{chat_acc:.1%}" if chat_acc is not None else "n/a",
    )
    cols[2].metric(
        "Model – Correct Score",
        f"{model_score_acc:.1%}" if model_score_acc is not None else "n/a",
    )
    cols[3].metric(
        f"{variant_name} – Correct Score",
        f"{chat_score_acc:.1%}" if chat_score_acc is not None else "n/a",
    )
    cols[4].metric(
        "Model xG Error (avg)",
        f"{avg_xg_err:.2f}" if avg_xg_err is not None else "n/a",
    )

    # ----------------------------------------------------------
    # Agreement Stats over recent results (using winners)
    # ----------------------------------------------------------
    st.subheader("Agreement Stats (Winner)")

    if {"model_winner_pred", chat_winner_col}.issubset(df.columns):
        has_both = df["model_winner_pred"].notna() & df[chat_winner_col].notna()
        total_both = has_both.sum()

        if total_both > 0:
            agree = has_both & (df["model_winner_pred"] == df[chat_winner_col])
            disagree = has_both & ~agree

            agree_rate = agree.sum() / total_both

            acc_when_agree = (
                df.loc[agree, "correct_winner_model"].mean()
                if "correct_winner_model" in df and agree.any()
                else None
            )

            acc_model_disagree = (
                df.loc[disagree, "correct_winner_model"].mean()
                if "correct_winner_model" in df and disagree.any()
                else None
            )
            acc_chat_disagree = (
                df.loc[disagree, chat_correct_win_col].mean()
                if chat_correct_win_col in df and disagree.any()
                else None
            )

            a_cols = st.columns(4)
            a_cols[0].metric(
                "Agreement Rate (winner)",
                f"{agree_rate:.1%}",
                f"{agree.sum()} / {total_both}",
            )
            a_cols[1].metric(
                "Accuracy when both agree (model)",
                f"{acc_when_agree:.1%}" if acc_when_agree is not None else "n/a",
            )
            a_cols[2].metric(
                "Model accuracy when disagree",
                f"{acc_model_disagree:.1%}"
                if acc_model_disagree is not None
                else "n/a",
            )
            a_cols[3].metric(
                f"{variant_name} accuracy when disagree",
                f"{acc_chat_disagree:.1%}"
                if acc_chat_disagree is not None
                else "n/a",
            )
        else:
            st.info("No rows with both model and ChatGPT winner predictions.")
    else:
        st.info(
            f"Columns 'model_winner_pred' and '{chat_winner_col}' not found "
            "in comparison file. Run compare_models.py with latest version."
        )

    # ----------------------------------------------------------
    # Match-level table
    # ----------------------------------------------------------
    st.subheader("Match-level Comparison (Current Season)")

    # Derived actual result as score string
    df = df.copy()
    df["Actual Score"] = df.apply(
        lambda r: f"{int(r['FTHG'])}-{int(r['FTAG'])}" if pd.notnull(r["FTHG"]) and pd.notnull(r["FTAG"]) else "",
        axis=1,
    )

    keep_cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "Actual Score",
        "model_score_pred",
        chat_score_col,
        "model_winner_pred",
        chat_winner_col,
        "correct_winner_model",
        chat_correct_win_col,
        "correct_score_model",
        chat_correct_score_col,
        "model_xg_error",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_display = df[keep_cols].copy()
    if "Date" in df_display.columns:
        df_display["Date"] = df_display["Date"].dt.date.astype(str)

    # Rename for readability
    rename_map = {
        "HomeTeam": "Home",
        "AwayTeam": "Away",
        "model_score_pred": "Model Score",
        chat_score_col: f"{variant_name} Score",
        "model_winner_pred": "Model Winner",
        chat_winner_col: f"{variant_name} Winner",
        "correct_winner_model": "Model Winner Correct?",
        chat_correct_win_col: f"{variant_name} Winner Correct?",
        "correct_score_model": "Model Score Correct?",
        chat_correct_score_col: f"{variant_name} Score Correct?",
        "model_xg_error": "Model xG Error",
    }
    df_display.rename(columns=rename_map, inplace=True)

    st.dataframe(df_display, use_container_width=True)


# ---------------------------------------------------------------------
# UI: Backtest accuracy
# ---------------------------------------------------------------------
def section_backtest_accuracy():
    st.header("📊 Backtest Accuracy – Expanding vs Rolling")

    df = load_backtest_metrics()
    if df.empty:
        st.info("Backtest metrics not found.")
        return

    st.subheader("Summary Table")
    st.dataframe(df, use_container_width=True)

    if "label" in df.columns:
        pivot = df.set_index("label")
        metrics = pivot[["accuracy", "brier", "log_loss", "mae_total"]]
        st.subheader("Key Metrics")
        st.write(metrics.style.format("{:.4f}"))


# ---------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------
def main():
    st.sidebar.title("⚽ EPL Agent Dashboard")
    page = st.sidebar.radio(
        "View",
        (
            "Predictions (Upcoming Fixtures)",
            "Model vs ChatGPT (Recent Results)",
            "Backtest Accuracy",
        ),
    )

    if page.startswith("Predictions"):
        section_predictions()
    elif page.startswith("Model vs ChatGPT"):
        section_comparison()
    else:
        section_backtest_accuracy()


if __name__ == "__main__":
    main()
