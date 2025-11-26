import sys
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # .../soccer_agent_local
SRC = ROOT / "src"
DB_PATH = ROOT / "data" / "soccer_agent.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sections import predictions, comparison, historical  # type: ignore


def main():
    st.set_page_config(
        page_title="EPL Agent Dashboard",
        layout="wide",
    )

    st.title("⚽ EPL Agent — Model Dashboard")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Predictions", "Comparison", "Historical"],
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"DB: `{DB_PATH.name}`")

    if page == "Predictions":
        predictions.render(DB_PATH)
    elif page == "Comparison":
        comparison.render(DB_PATH)
    elif page == "Historical":
        historical.render(DB_PATH)


if __name__ == "__main__":
    main()
