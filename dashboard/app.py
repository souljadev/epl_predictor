import streamlit as st

from sections.predictions import section_upcoming_predictions
from sections.historical import section_historical_accuracy
from sections.chatgpt_vs_model import section_chatgpt_vs_model
from sections.backtests import section_backtests


st.set_page_config(
    layout="wide",
    page_title="EPL Agent Dashboard",
    page_icon="⚽",
)


def main():
    st.sidebar.title("⚽ EPL Agent Dashboard")

    page = st.sidebar.radio(
        "View",
        (
            "Upcoming Predictions",
            "Historical Accuracy",
            "ChatGPT vs Model",
            "Backtests",
        ),
    )

    if page == "Upcoming Predictions":
        section_upcoming_predictions()
    elif page == "Historical Accuracy":
        section_historical_accuracy()
    elif page == "ChatGPT vs Model":
        section_chatgpt_vs_model()
    else:
        section_backtests()


if __name__ == "__main__":
    main()
