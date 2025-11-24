import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # soccer_agent_local/
SRC = ROOT / "src"
sys.path.append(str(SRC))

from db import init_db, insert_chatgpt_predictions  # noqa: E402

FIXTURES_PATH = ROOT / "data" / "raw" / "fixtures_today.csv"
CHATGPT_CSV = ROOT / "data" / "processed" / "predictions_chatgpt.csv"


def main():
    init_db()

    if not FIXTURES_PATH.exists():
        raise FileNotFoundError(f"Fixtures file not found: {FIXTURES_PATH}")
    if not CHATGPT_CSV.exists():
        raise FileNotFoundError(f"ChatGPT CSV not found: {CHATGPT_CSV}")

    fixtures = pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
    chat_df = pd.read_csv(CHATGPT_CSV)

    merged = pd.merge(
        fixtures[["Date", "HomeTeam", "AwayTeam"]],
        chat_df,
        on=["HomeTeam", "AwayTeam"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping fixtures between fixtures_today.csv and predictions_chatgpt.csv.\n"
            "Check team names and that you predicted the right games."
        )

    run_id = datetime.utcnow().strftime("%Y%m%d")
    insert_chatgpt_predictions(merged, prompt_version="manual_v1", run_id=run_id)
    print(f"Inserted {len(merged)} ChatGPT predictions into 'predictions_chatgpt' with run_id={run_id}.")


if __name__ == "__main__":
    main()
