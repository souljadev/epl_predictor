import streamlit as st
import pandas as pd

# Expand usable width, prevent horizontal scroll
st.markdown("""
<style>
    .block-container {
        max-width: 95% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .element-container iframe {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("models/predictions/predictions_full.csv")
    exact = pd.read_csv("models/predictions/predictions_exact_scores.csv")

    # Parse and format date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return df, exact


df, exact = load_data()


# ============================================================
# TITLE
# ============================================================
st.title("⚽ Soccer Prediction Dashboard")
st.caption("Data-driven match predictions • Exact scores • Betting markets")


# ============================================================
# OVERVIEW TABLE (NORMAL WIDTH, CLEAN OVER/UNDER)
# ============================================================
st.markdown("---")
st.header("📊 Match Overview — Predicted Outcomes for All Fixtures")

overview_rows = []

for _, row in df.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]

    # Determine predicted outcome (1X2)
    probs = {
        "Home": row["home_win_prob"],
        "Draw": row["draw_prob"],
        "Away": row["away_win_prob"],
    }
    outcome = max(probs, key=probs.get)
    outcome_prob = probs[outcome]

    if outcome == "Home":
        predicted = home
    elif outcome == "Away":
        predicted = away
    else:
        predicted = "Draw"

    # Determine Over/Under side and probability
    over_p = row["over_2_5"]
    under_p = row["under_2_5"]
    if over_p >= under_p:
        ou_side = "Over"
        ou_prob = over_p
    else:
        ou_side = "Under"
        ou_prob = under_p

    overview_rows.append(
        {
            "Date": row["Date"],
            "Match": f"{home} vs {away}",
            "Predicted Result": predicted,
            "Result Probability": f"{outcome_prob * 100:.1f}%",
            "Most Likely Score": row["most_likely_score"],
            "Score Prob %": f"{row['most_likely_score_prob'] * 100:.1f}%",
            "Over/Under": f"{ou_side} ({ou_prob * 100:.1f}%)",
        }
    )

overview_df = pd.DataFrame(overview_rows)

st.dataframe(
    overview_df,
    hide_index=True,
    column_config=None,
    use_container_width=True
)


# ============================================================
# MATCH SELECTOR
# ============================================================
st.markdown("---")
st.header("🔍 Match Details & Market Breakdown")

match_list = df.apply(
    lambda r: f"{r['Date']} - {r['HomeTeam']} vs {r['AwayTeam']}", axis=1
)

selection = st.selectbox("Choose a match:", match_list)

date_selected, teams = selection.split(" - ")
home_selected, away_selected = teams.split(" vs ")

row = df[
    (df["Date"] == date_selected)
    & (df["HomeTeam"] == home_selected)
    & (df["AwayTeam"] == away_selected)
].iloc[0]


# ============================================================
# 1X2 PROBABILITIES
# ============================================================
st.subheader(f"📌 {home_selected} vs {away_selected} — {date_selected}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Home Win %", f"{row['home_win_prob'] * 100:.1f}%")
with col2:
    st.metric("Draw %", f"{row['draw_prob'] * 100:.1f}%")
with col3:
    st.metric("Away Win %", f"{row['away_win_prob'] * 100:.1f}%")


# ============================================================
# MOST LIKELY SCORE + EXPECTED GOALS
# ============================================================
st.markdown("---")
st.subheader("🎯 Most Likely Score & Expected Goals")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Most Likely Score", row["most_likely_score"])
with colB:
    st.metric("Score Probability", f"{row['most_likely_score_prob'] * 100:.1f}%")
with colC:
    st.metric("Total xG", f"{row['exp_total_goals']:.2f}")

st.write(
    f"- Expected Home Goals: **{row['exp_home_goals']:.2f}**\n"
    f"- Expected Away Goals: **{row['exp_away_goals']:.2f}**"
)


# ============================================================
# MARKET BREAKDOWN
# ============================================================
st.markdown("---")
st.subheader("📈 Betting Market Breakdown")

col1, col2 = st.columns(2)

with col1:
    st.write("### BTTS (Both Teams To Score)")
    st.write(f"**Yes:** {row['btts_yes'] * 100:.1f}%")
    st.write(f"**No:** {row['btts_no'] * 100:.1f}%")

    st.write("### Double Chance")
    st.write(f"**1X:** {row['double_chance_1x'] * 100:.1f}%")
    st.write(f"**X2:** {row['double_chance_x2'] * 100:.1f}%")
    st.write(f"**12:** {row['double_chance_12'] * 100:.1f}%")

with col2:
    st.write("### Over / Under 2.5 Goals")
    st.write(f"**Over 2.5:** {row['over_2_5'] * 100:.1f}%")
    st.write(f"**Under 2.5:** {row['under_2_5'] * 100:.1f}%")


# ============================================================
# EXACT SCORE MATRIX
# ============================================================
st.markdown("---")
st.subheader("🔢 Exact Score Probability Matrix")

matrix = exact[
    (exact["HomeTeam"] == home_selected)
    & (exact["AwayTeam"] == away_selected)
].copy()

matrix[["HG", "AG"]] = matrix["score"].str.split("-", expand=True).astype(int)
pivot = matrix.pivot_table(index="HG", columns="AG", values="prob").fillna(0)

st.write("Probability for each exact score (in %):")
st.table((pivot * 100).round(2))


# ============================================================
# AI INTERPRETATION
# ============================================================
st.markdown("---")
st.subheader("🧠 Match Interpretation")


def interpret(row):
    home = row["HomeTeam"]
    away = row["AwayTeam"]

    pH = row["home_win_prob"]
    pD = row["draw_prob"]
    pA = row["away_win_prob"]

    winner_prob = max([pH, pD, pA])
    if winner_prob == pH:
        predicted = f"{home} win"
    elif winner_prob == pA:
        predicted = f"{away} win"
    else:
        predicted = "Draw"

    return (
        f"**Match Date:** {row['Date']}\n"
        f"**Prediction:** {predicted} ({winner_prob * 100:.1f}%)\n\n"
        f"**Most likely score:** {row['most_likely_score']} "
        f"({row['most_likely_score_prob'] * 100:.1f}%)\n"
        f"**Expected goals:** {row['exp_total_goals']:.2f} xG\n"
        f"**BTTS Yes:** {row['btts_yes'] * 100:.1f}%\n"
        f"**Over 2.5 Goals:** {row['over_2_5'] * 100:.1f}%"
    )


st.markdown(interpret(row))
