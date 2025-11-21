import streamlit as st
import pandas as pd
import plotly.express as px
# Expand width but not too wide
st.set_page_config(layout="wide")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_metrics():
    path = "models/evaluation/metrics_backtests_summary.csv"
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    if "run_id" not in df.columns:
        df["run_id"] = df.index + 1  # fallback

    return df


df = load_metrics()


# =========================================================
# TITLE
# =========================================================
st.title("📊 Model Accuracy Dashboard")
st.caption("Comparing backtest performance across model runs")


# =========================================================
# SUMMARY TABLE
# =========================================================
st.markdown("### 📋 Summary of Backtest Metrics")

show_cols = [
    "run_id", "n_matches", "accuracy", "brier", "log_loss",
    "mae_home", "mae_away", "mae_total"
]

st.dataframe(
    df[show_cols],
    hide_index=True,
    use_container_width=True
)


# =========================================================
# METRIC HIGHLIGHTS
# =========================================================
st.markdown("---")
st.header("🏆 Best Run Highlights")

col1, col2, col3, col4 = st.columns(4)

best_acc = df.loc[df["accuracy"].idxmax()]
best_log = df.loc[df["log_loss"].idxmin()]
best_brier = df.loc[df["brier"].idxmin()]
best_mae = df.loc[df["mae_total"].idxmin()]

with col1:
    st.metric(
        "Highest Accuracy",
        f"{best_acc['accuracy']*100:.2f}%",
        help=f"Run ID: {best_acc['run_id']}"
    )
with col2:
    st.metric(
        "Lowest Log-Loss",
        f"{best_log['log_loss']:.4f}",
        help=f"Run ID: {best_log['run_id']}"
    )
with col3:
    st.metric(
        "Lowest Brier Score",
        f"{best_brier['brier']:.4f}",
        help=f"Run ID: {best_brier['run_id']}"
    )
with col4:
    st.metric(
        "Lowest Total Goal MAE",
        f"{best_mae['mae_total']:.4f}",
        help=f"Run ID: {best_mae['run_id']}"
    )


# =========================================================
# CHARTS
# =========================================================
st.markdown("---")
st.header("📈 Performance Charts")

# Accuracy Trend
fig_acc = px.line(
    df,
    x="run_id",
    y="accuracy",
    markers=True,
    title="Accuracy Over Model Runs"
)
st.plotly_chart(fig_acc, use_container_width=True)

# Log Loss Trend
fig_ll = px.line(
    df,
    x="run_id",
    y="log_loss",
    markers=True,
    title="Log Loss Over Model Runs"
)
st.plotly_chart(fig_ll, use_container_width=True)

# Brier Score Trend
fig_brier = px.line(
    df,
    x="run_id",
    y="brier",
    markers=True,
    title="Brier Score Over Model Runs"
)
st.plotly_chart(fig_brier, use_container_width=True)

# MAE Total
fig_mae = px.line(
    df,
    x="run_id",
    y="mae_total",
    markers=True,
    title="Total MAE Over Model Runs"
)
st.plotly_chart(fig_mae, use_container_width=True)


# =========================================================
# DETAIL VIEW
# =========================================================
st.markdown("---")
st.header("🔍 Compare Specific Runs")

run_ids = df["run_id"].tolist()
selected_runs = st.multiselect("Select runs to compare:", run_ids)

if selected_runs:
    st.dataframe(
        df[df["run_id"].isin(selected_runs)][show_cols],
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Built automatically from your model backtest history.")
