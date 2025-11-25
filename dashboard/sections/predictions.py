import pandas as pd
import streamlit as st

from utils.loaders import load_upcoming_predictions


def section_upcoming_predictions():
    st.header("🔮 Upcoming Fixtures – Model & ChatGPT Predictions")

    df = load_upcoming_predictions(days_ahead=7)
    if df.empty:
        st.info(
            "No upcoming fixtures with predictions found.\n\n"
            "Make sure:\n"
            " - Fixtures exist in the DB (future dates)\n"
            " - Model predictions have been run and written to DB\n"
            " - ChatGPT predictions (chatgpt_pred) have been generated"
        )
        return

    df_display = df.copy()

    # ----------------------------------------------------------
    # Model winner from probabilities
    # ----------------------------------------------------------
    def model_winner(row):
        probs = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        if any(pd.isna(probs)):
            return ""
        idx = int(pd.Series(probs).idxmax())
        if idx == 0:
            return row["home_team"]
        elif idx == 1:
            return "Draw"
        else:
            return row["away_team"]

    df_display["Model Winner"] = df_display.apply(model_winner, axis=1)

    # ----------------------------------------------------------
    # Model score
    # ----------------------------------------------------------
    def model_score(row):
        if isinstance(row.get("score_pred"), str) and "-" in row["score_pred"]:
            return row["score_pred"]
        if pd.isna(row["exp_goals_home"]) or pd.isna(row["exp_goals_away"]):
            return ""
        h = int(round(row["exp_goals_home"]))
        a = int(round(row["exp_goals_away"]))
        return f"{h}-{a}"

    df_display["Model Score"] = df_display.apply(model_score, axis=1)

    # ----------------------------------------------------------
    # ChatGPT winner
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
        lambda r: chatgpt_winner(r.get("chatgpt_pred", ""), r["home_team"], r["away_team"]),
        axis=1,
    )

    # ----------------------------------------------------------
    # Winner Match?
    # ----------------------------------------------------------
    def match_indicator(row):
        mw = row["Model Winner"]
        cw = row["ChatGPT Winner"]

        if cw == "":
            return "⚪ No ChatGPT"
        if mw == cw:
            return "🟩 Match"
        if mw == "Draw" or cw == "Draw":
            return "🟨 Partial"
        return "🟥 No Match"

    df_display["Winner Match?"] = df_display.apply(match_indicator, axis=1)

    # ----------------------------------------------------------
    # Score Match? (Exact Score)
    # ----------------------------------------------------------
    def score_match_indicator(row):
        m = row.get("Model Score", "")
        c = row.get("chatgpt_pred", "")

        if not isinstance(c, str) or "-" not in c:
            return "⚪ No ChatGPT Score"
        if m == "":
            return "⚪ No Model Score"
        if m == c:
            return "🟩 Exact Match"
        return "🟥 No Match"

    df_display["Score Match?"] = df_display.apply(score_match_indicator, axis=1)

    # ----------------------------------------------------------
    # Agreement stats (winner)
    # ----------------------------------------------------------
    counts = df_display["Winner Match?"].value_counts(dropna=False)
    total_with_data = len(df_display[df_display["Winner Match?"] != "⚪ No ChatGPT"]) or 1

    full_match = counts.get("🟩 Match", 0)
    partial = counts.get("🟨 Partial", 0)
    no_match = counts.get("🟥 No Match", 0)
    no_data = counts.get("⚪ No ChatGPT", 0)

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
    col_stats[3].metric("No ChatGPT Data", f"{no_data}")

    # ----------------------------------------------------------
    # Rename columns for UI
    # ----------------------------------------------------------
    df_display.rename(
        columns={
            "date": "Date",
            "home_team": "Home",
            "away_team": "Away",
            "home_win_prob": "Prob Home Win",
            "draw_prob": "Prob Draw",
            "away_win_prob": "Prob Away Win",
            "chatgpt_pred": "ChatGPT Score",
        },
        inplace=True,
    )

    df_display["Date"] = df_display["Date"].astype(str)

    for col in ["Prob Home Win", "Prob Draw", "Prob Away Win"]:
        df_display[col] = (df_display[col] * 100).round().astype(int).astype(str) + "%"

    # ----------------------------------------------------------
    # Desired column order
    # ----------------------------------------------------------
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
        "Winner Match?",
        "Score Match?",
    ]

    cols_order = [c for c in cols_order if c in df_display.columns]
    df_display = df_display[cols_order]

    # ----------------------------------------------------------
    # Styling for match indicators
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

    styled_df = (
        df_display.style
        .applymap(style_match, subset=["Winner Match?"])
        .applymap(style_match, subset=["Score Match?"])
    )

    st.subheader("Predictions for Next 7 Days")
    st.dataframe(styled_df, use_container_width=True)
