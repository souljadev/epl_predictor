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
    st.subheader("Match Predictions")

    today = date.today()
    selected_date = st.date_input("Select match date", value=today)
    target_ts = pd.to_datetime(selected_date)

    df = load_predictions_for_date(db_path, target_ts)

    if df.empty:
        st.info("No predictions for this date.")
        return

    st.markdown(f"### Predictions for {target_ts.date()}")

    # --------------------------------------------------------
    # SPLIT MODEL vs CHATGPT
    # --------------------------------------------------------
    model_df = df[df["score_pred"].notna()].copy()
    gpt_df   = df[df["chatgpt_pred"].notna()].copy()

    # --------------------------------------------------------
    # DEDUPE MODEL ROWS — CHOOSE SCORE CLOSEST TO xG
    # --------------------------------------------------------
    def pick_best_model_row(group: pd.DataFrame) -> pd.Series:
        """
        For all model rows of the same match, pick the score_pred
        that is closest to (exp_goals_home, exp_goals_away).
        """
        best_idx = None
        best_err = None

        for idx, row in group.iterrows():
            h, a = parse_score(row["score_pred"])
            if h is None or a is None:
                continue
            lamH = float(row["exp_goals_home"])
            lamA = float(row["exp_goals_away"])
            err = (h - lamH) ** 2 + (a - lamA) ** 2
            if best_err is None or err < best_err:
                best_err = err
                best_idx = idx

        # fallback: if none parsable, just take first
        if best_idx is None:
            return group.iloc[0]
        return group.loc[best_idx]

    if not model_df.empty:
        model_df = (
            model_df.groupby(["date", "home_team", "away_team"], as_index=False)
            .apply(pick_best_model_row)
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # DEDUPE CHATGPT ROWS — KEEP LAST PER MATCH
    # --------------------------------------------------------
    if not gpt_df.empty:
        gpt_df = (
            gpt_df.sort_values("created_at")
            .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
            .copy()
        )

    # rename & align columns
    model_df = model_df.rename(columns={"score_pred": "model_score"})
    gpt_df["chatgpt_score"] = gpt_df["chatgpt_pred"]

    gpt_df = gpt_df[["date", "home_team", "away_team", "chatgpt_score"]]

    # --------------------------------------------------------
    # MERGE → ONE ROW PER MATCH
    # --------------------------------------------------------
    merged = model_df.merge(
        gpt_df,
        on=["date", "home_team", "away_team"],
        how="left",
    )

    if merged.empty:
        st.info("No merged model/ChatGPT predictions for this date.")
        return

    # --------------------------------------------------------
    # BUILD DISPLAY TABLE
    # --------------------------------------------------------
    display_df = merged.copy()

    # date only, no time
    display_df["date"] = display_df["date"].dt.date

    # round xG to whole numbers
    display_df["exp_goals_home"] = display_df["exp_goals_home"].round(0).astype(int)
    display_df["exp_goals_away"] = display_df["exp_goals_away"].round(0).astype(int)
    display_df["exp_total_goals"] = display_df["exp_total_goals"].round(0).astype(int)

    # probabilities as %
    display_df["H Prob"] = display_df["home_win_prob"].apply(lambda x: f"{x*100:.1f}%")
    display_df["D Prob"] = display_df["draw_prob"].apply(lambda x: f"{x*100:.1f}%")
    display_df["A Prob"] = display_df["away_win_prob"].apply(lambda x: f"{x*100:.1f}%")

    # winners
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

    # --------------------------------------------------------
    # HIGHLIGHTING RULES
    # --------------------------------------------------------
    def highlight_row(row):
        styles = [""] * len(row)

        idx_model_score = row.index.get_loc("model_score")
        idx_chat_score = row.index.get_loc("chatgpt_score")
        idx_model_win = row.index.get_loc("model_winner_team")
        idx_chat_win = row.index.get_loc("chatgpt_winner_team")

        # Yellow if matching score
        if (
            pd.notna(row["model_score"])
            and pd.notna(row["chatgpt_score"])
            and row["model_score"] == row["chatgpt_score"]
        ):
            styles[idx_model_score] = "background-color: yellow; font-weight: bold;"
            styles[idx_chat_score] = "background-color: yellow; font-weight: bold;"

        # Green if matching winner
        if (
            pd.notna(row["model_winner_team"])
            and pd.notna(row["chatgpt_winner_team"])
            and row["model_winner_team"] == row["chatgpt_winner_team"]
        ):
            styles[idx_model_win] = "background-color: lightgreen; font-weight: bold;"
            styles[idx_chat_win] = "background-color: lightgreen; font-weight: bold;"

        return styles

    # final display columns
    display_df = display_df[
        [
            "date",
            "home_team",
            "away_team",
            "H Prob",
            "D Prob",
            "A Prob",
            "exp_goals_home",
            "exp_goals_away",
            "exp_total_goals",
            "model_score",
            "model_winner_team",
            "chatgpt_score",
            "chatgpt_winner_team",
        ]
    ]

    st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)

    # --------------------------------------------------------
    # MATCH DETAIL EXPANDERS
    # --------------------------------------------------------
    st.markdown("### Match Details")

    for _, row in merged.iterrows():
        label = (
            f"{row['home_team']} vs {row['away_team']} — "
            f"{row['home_win_prob']*100:.1f}% / "
            f"{row['draw_prob']*100:.1f}% / "
            f"{row['away_win_prob']*100:.1f}%"
        )

        model_winner_code = winner_from_score(row["model_score"])
        chatgpt_winner_code = (
            winner_from_score(row["chatgpt_score"]) if row["chatgpt_score"] else None
        )

        model_winner_team = winner_to_team(
            model_winner_code, row["home_team"], row["away_team"]
        )
        chatgpt_winner_team = winner_to_team(
            chatgpt_winner_code, row["home_team"], row["away_team"]
        )

        with st.expander(label):
            st.write(
                f"**Home/Draw/Away** — "
                f"{row['home_win_prob']:.3f} / "
                f"{row['draw_prob']:.3f} / "
                f"{row['away_win_prob']:.3f}"
            )
            st.write(
                f"**Rounded xG** — "
                f"Home: `{round(row['exp_goals_home'])}`, "
                f"Away: `{round(row['exp_goals_away'])}`, "
                f"Total: `{round(row['exp_total_goals'])}`"
            )
            st.write(f"**Model score:** `{row['model_score']}`")
            st.write(f"**Model winner:** `{model_winner_team}`")

            if row["chatgpt_score"]:
                st.write(f"**ChatGPT score:** `{row['chatgpt_score']}`")
                st.write(f"**ChatGPT winner:** `{chatgpt_winner_team}`")
            else:
                st.write("_ChatGPT did not predict a score for this match._")
