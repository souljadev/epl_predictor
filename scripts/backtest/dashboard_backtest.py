import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY_FILE = ROOT / "models" / "history" / "agent_history.csv"


def main():
    if not HISTORY_FILE.exists():
        print(f"No agent history file found at {HISTORY_FILE}")
        return

    df = pd.read_csv(HISTORY_FILE)

    df = df.sort_values("fitness", ascending=False)

    print("\n=== Top 10 configurations by fitness ===")
    cols_show = [
        "run_tag", "fitness",
        "metric_n_matches", "metric_accuracy", "metric_brier",
        "metric_log_loss",
        "w_dc", "w_elo", "rho_init", "elo_k_factor", "elo_home_adv"
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    print(df[cols_show].head(10).to_string(index=False))

    print("\nYou can also open this CSV in Excel / Sheets:")
    print(HISTORY_FILE)


if __name__ == "__main__":
    main()
