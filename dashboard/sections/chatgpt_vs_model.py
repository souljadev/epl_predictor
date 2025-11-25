import pandas as pd
import streamlit as st

from utils.loaders import load_comparison_from_db


def section_chatgpt_vs_model():
    st.header("⚔️ ChatGPT vs Model – Head-to-head")

    df = load_comparison_from_db()
    if df.empty:
        st.info("No comparison data found in DB.")
        return

    df = df.copy()
    df = df.dropna(subset=["Date"])
    if df.empty:
        st.info("No valid dated rows in comparison data.")
        return

    required = {
        "correct_winner_model",
        "correct_winner_chatgpt",
        "model_winner_pred",
        "chatgpt_winner_pred",
    }
    if not required.issubset(df.columns):
        st.info("Comparison data is missing required columns for ChatGPT vs Model view.")
        return

    has_both = df["correct_winner_model"].notna() & df["correct_winner_chatgpt"].notna()
    df_h = df[has_both].copy()
    if df_h.empty:
        st.info("No rows where both model and ChatGPT predictions exist alongside results.")
        return

    agree = df_h["model_winner_pred"] == df_h["chatgpt_winner_pred"]
    disagree = ~agree
    total = len(df_h)

    agree_rate = agree.mean()
    acc_when_agree_model = df_h.loc[agree, "correct_winner_model"].mean()
    acc_when_agree_chat = df_h.loc[agree, "correct_winner_chatgpt"].mean()

    acc_model_disagree = df_h.loc[disagree, "correct_winner_model"].mean()
    acc_chat_disagree = df_h.loc[disagree, "correct_winner_chatgpt"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Agreement rate (winner)",
        f"{agree_rate:.1%}",
        f"{agree.sum()} / {total}",
    )
    col2.metric(
        "Model accuracy when agree",
        f"{acc_when_agree_model:.1%}" if pd.notna(acc_when_agree_model) else "n/a",
    )
    col3.metric(
        "ChatGPT accuracy when agree",
        f"{acc_when_agree_chat:.1%}" if pd.notna(acc_when_agree_chat) else "n/a",
    )
    col4.metric(
        "When disagree – who is better?",
        "Model" if acc_model_disagree > acc_chat_disagree else
        "ChatGPT" if acc_chat_disagree > acc_model_disagree else
        "Tie",
    )

    st.subheader("Disagreement Cases (who was right?)")

    df_dis = df_h[disagree].copy()
    if df_dis.empty:
        st.info("No disagreement rows.")
        return

    if {"FTHG", "FTAG"}.issubset(df_dis.columns):
        df_dis["Actual Score"] = df_dis["actual_score"].fillna("")

    cols_keep = [
        "Date",
        "home_team",
        "away_team",
        "Actual Score",
        "model_winner_pred",
        "chatgpt_winner_pred",
        "correct_winner_model",
        "correct_winner_chatgpt",
    ]
    cols_keep = [c for c in cols_keep if c in df_dis.columns]

    df_display = df_dis[cols_keep].copy()
    df_display.rename(
        columns={
            "home_team": "Home",
            "away_team": "Away",
            "model_winner_pred": "Model Winner (H/D/A)",
            "chatgpt_winner_pred": "ChatGPT Winner (H/D/A)",
            "correct_winner_model": "Model Correct?",
            "correct_winner_chatgpt": "ChatGPT Correct?",
        },
        inplace=True,
    )

    st.dataframe(df_display, use_container_width=True)
