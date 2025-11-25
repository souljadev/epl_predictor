import pandas as pd
import streamlit as st

from utils.loaders import load_comparison_from_db


def section_backtests():
    st.header("📊 Backtest Accuracy – By Model Version")

    df = load_comparison_from_db()
    if df.empty:
        st.info("No comparison data found in DB.")
        return

    if "model_version" not in df.columns:
        st.info("model_version column not found in comparison data.")
        return

    grouped = df.groupby("model_version")

    rows = []
    for mv, g in grouped:
        n = len(g)
        win_acc = g["correct_winner_model"].mean() if "correct_winner_model" in g else None
        score_acc = g["correct_score_model"].mean() if "correct_score_model" in g else None
        pred_miss = g["model_xg_error"].mean() if "model_xg_error" in g else None

        rows.append(
            {
                "model_version": mv,
                "matches": n,
                "winner_accuracy": win_acc,
                "score_accuracy": score_acc,
                "prediction_miss": pred_miss,
            }
        )

    if not rows:
        st.info("No grouped metrics could be computed.")
        return

    mdf = pd.DataFrame(rows).sort_values("model_version")

    # Convert prediction miss to % for readability
    if "prediction_miss" in mdf.columns:
        mdf["prediction_miss_pct"] = (mdf["prediction_miss"] * 100).round(1)

    st.subheader("Summary by model_version")
    st.dataframe(
        mdf.style.format(
            {
                "winner_accuracy": "{:.3f}",
                "score_accuracy": "{:.3f}",
                "prediction_miss": "{:.3f}",
                "prediction_miss_pct": "{:.1f}",
            }
        ),
        use_container_width=True,
    )
