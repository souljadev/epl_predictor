import pandas as pd
import streamlit as st

from utils.loaders import load_comparison_from_db


def section_historical_accuracy():
    st.header("📈 Historical Accuracy – Model & ChatGPT")

    # Load full comparison live from DB (predictions + results)
    df = load_comparison_from_db()
    if df.empty:
        st.info("No comparison data found in DB. Make sure predictions and results exist.")
        return

    df = df.copy()
    df = df.dropna(subset=["Date"])
    if df.empty:
        st.info("No valid dated rows in comparison data.")
        return

    # Only keep rows that have model score OR ChatGPT score
    score_series = df.get("score_pred", pd.Series([None] * len(df)))
    chat_series = df.get("chatgpt_pred", pd.Series([None] * len(df)))

    df = df[
        (score_series.notna() & (score_series != "")) |
        (chat_series.notna() & (chat_series != ""))
    ]
    if df.empty:
        st.info("No rows have model or ChatGPT scores.")
        return

    # ------------------------------------------------------------------------------------
    # Convert winner codes (H/A/D) → actual team names
    # ------------------------------------------------------------------------------------
    def winner_to_team(row, val):
        if val == "H":
            return row["home_team"]
        if val == "A":
            return row["away_team"]
        if val == "D":
            return "Draw"
        return None

    df["Model Winner Name"] = df.apply(
        lambda r: winner_to_team(r, r.get("model_winner_pred")), axis=1
    )
    df["Actual Winner Name"] = df.apply(
        lambda r: winner_to_team(r, r.get("actual_winner")), axis=1
    )
    df["ChatGPT Winner Name"] = df.apply(
        lambda r: winner_to_team(r, r.get("chatgpt_winner_pred")), axis=1
    )

    # ------------------------------------------------------------------------------------
    # Model score
    # ------------------------------------------------------------------------------------
    if "score_pred" in df.columns:
        df["Model Score"] = df["score_pred"].fillna("")
    else:
        df["Model Score"] = (
            df["exp_goals_home"].fillna(0).round().astype(int).astype(str)
            + "-"
            + df["exp_goals_away"].fillna(0).round().astype(int).astype(str)
        )

    # ------------------------------------------------------------------------------------
    # Actual score (if results exist)
    # ------------------------------------------------------------------------------------
    if {"FTHG", "FTAG"}.issubset(df.columns):
        df["Actual Score"] = df["actual_score"].fillna("")
    else:
        df["Actual Score"] = None

    # ChatGPT score
    df["ChatGPT Score"] = df.get("chatgpt_pred", "")

    # ------------------------------------------------------------------------------------
    # Accuracy metrics
    # ------------------------------------------------------------------------------------
    model_win_acc = df["correct_winner_model"].mean() if "correct_winner_model" in df else None
    model_score_acc = df["correct_score_model"].mean() if "correct_score_model" in df else None

    model_pred_miss_pct = (
        (df["model_xg_error"] * 100).mean()
        if "model_xg_error" in df else None
    )

    chat_cols_exist = (
        "correct_winner_chatgpt" in df.columns and
        "correct_score_chatgpt" in df.columns
    )

    chat_win_acc = df["correct_winner_chatgpt"].mean() if chat_cols_exist else None
    chat_score_acc = df["correct_score_chatgpt"].mean() if chat_cols_exist else None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model – Winner Accuracy",
                f"{model_win_acc:.1%}" if model_win_acc is not None else "n/a")
    col2.metric("Model – Exact Score Accuracy",
                f"{model_score_acc:.1%}" if model_score_acc is not None else "n/a")
    col3.metric("Prediction Miss (%)",
                f"{model_pred_miss_pct:.0f}%" if model_pred_miss_pct is not None else "n/a")
    col4.metric("ChatGPT – Winner Accuracy",
                f"{chat_win_acc:.1%}" if chat_win_acc is not None else "n/a")
    col5.metric("ChatGPT – Exact Score Accuracy",
                f"{chat_score_acc:.1%}" if chat_score_acc is not None else "n/a")

    # ------------------------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------------------------
    desired_cols = [
        "Date",
        "home_team",
        "away_team",
        "Actual Score",
        "Model Score",
        "ChatGPT Score",
        "Actual Winner Name",
        "Model Winner Name",
        "ChatGPT Winner Name",
        "correct_winner_model",
        "correct_score_model",
        "correct_winner_chatgpt",
        "correct_score_chatgpt",
        "model_xg_error",
    ]

    existing_cols = [c for c in desired_cols if c in df.columns]
    df_display = df[existing_cols].copy()

    df_display.rename(columns={
        "home_team": "Home",
        "away_team": "Away",
        "correct_winner_model": "Model Winner Correct?",
        "correct_score_model": "Model Score Correct?",
        "correct_winner_chatgpt": "ChatGPT Winner Correct?",
        "correct_score_chatgpt": "ChatGPT Score Correct?",
        "model_xg_error": "Prediction Miss (raw)",
    }, inplace=True)

    # ------------------------------------------------------------------------------------
    # SAFE % conversion (fixes your NaN error!)
    # ------------------------------------------------------------------------------------
    if "Prediction Miss (raw)" in df_display.columns:
        df_display["Prediction Miss (%)"] = (
            df_display["Prediction Miss (raw)"] * 100
        ).round().apply(lambda x: f"{int(x)}%" if pd.notna(x) else "")
        df_display.drop(columns=["Prediction Miss (raw)"], inplace=True)

    # Sort newest first
    df_display = df_display.sort_values("Date", ascending=False)

    st.subheader("Matches with Model and/or ChatGPT Scores")
    st.dataframe(df_display, use_container_width=True)
