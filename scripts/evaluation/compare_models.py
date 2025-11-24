from pathlib import Path
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from db import init_db, get_conn, upsert_results_from_epl_combined  # noqa: E402

CONFIG_PATH = ROOT / "config.yaml"
OUT_PATH = ROOT / "models" / "history" / "comparison.csv"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


def winner_label(h, a):
    return "H" if h > a else ("A" if a > h else "D")


def parse_score(s):
    try:
        h, a = s.split("-")
        return int(h), int(a)
    except Exception:
        return None, None


def winner_from_score(s):
    h, a = parse_score(s)
    if h is None:
        return None
    return winner_label(h, a)


def winner_from_probs(row):
    probs = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
    idx = int(pd.Series(probs).idxmax())
    return ["H", "D", "A"][idx]


def run_comparison():
    print("\n==========================================")
    print("        Evaluating Model vs ChatGPT       ")
    print("                (DB-based)                ")
    print("==========================================\n")

    init_db()

    # Ensure results DB is synced with CSV
    cfg = load_config()
    results_rel = cfg.get("data", {}).get("results_csv", "data/raw/epl_combined.csv")
    results_csv = ROOT / results_rel
    upsert_results_from_epl_combined(results_csv)

    with get_conn() as conn:
        results_df = pd.read_sql("SELECT * FROM results", conn)
        preds_df = pd.read_sql("SELECT * FROM predictions", conn)

    if preds_df.empty:
        raise RuntimeError("No predictions found in DB. Run model predictions first.")

    # Latest prediction per match (any model_version)
    preds_df["created_at"] = pd.to_datetime(preds_df["created_at"], errors="coerce")
    preds_df = preds_df.sort_values(
        ["date", "home_team", "away_team", "created_at"]
    ).drop_duplicates(
        subset=["date", "home_team", "away_team"],
        keep="last",
    )

    merged = preds_df.merge(
        results_df,
        on=["date", "home_team", "away_team"],
        how="inner",
        suffixes=("_pred", "_res"),
    )

    if merged.empty:
        raise RuntimeError("No overlapping rows between predictions and results.")

    # Actual labels
    merged["actual_winner"] = merged.apply(
        lambda r: winner_label(int(r["FTHG"]), int(r["FTAG"])),
        axis=1,
    )
    merged["actual_score"] = (
        merged["FTHG"].astype(int).astype(str)
        + "-"
        + merged["FTAG"].astype(int).astype(str)
    )

    # Model metrics (historical only, because we inner-joined with results)
    merged["model_winner_pred"] = merged.apply(winner_from_probs, axis=1)
    merged["correct_winner_model"] = (
        merged["model_winner_pred"] == merged["actual_winner"]
    )

    merged["correct_score_model"] = merged["score_pred"] == merged["actual_score"]

    merged["model_xg_error"] = (
        merged["exp_total_goals"] - (merged["FTHG"] + merged["FTAG"])
    ).abs()

    # ChatGPT metrics: only on rows where chatgpt_pred exists
    has_chat = merged["chatgpt_pred"].notna() & (merged["chatgpt_pred"] != "")
    chat_df = merged[has_chat].copy()

    if not chat_df.empty:
        chat_df["chatgpt_winner_pred"] = chat_df["chatgpt_pred"].apply(
            winner_from_score
        )
        chat_df["correct_winner_chatgpt"] = (
            chat_df["chatgpt_winner_pred"] == chat_df["actual_winner"]
        )
        chat_df["correct_score_chatgpt"] = (
            chat_df["chatgpt_pred"] == chat_df["actual_score"]
        )
    else:
        chat_df = None

    # ----------------- SUMMARY -----------------
    print("===== SUMMARY (MODEL) =====")
    print(f"Matches evaluated (model):      {len(merged)}")
    print(
        f"Model winner accuracy:          {merged['correct_winner_model'].mean():.3f}"
    )
    print(
        f"Model exact score accuracy:     {merged['correct_score_model'].mean():.3f}"
    )
    print(f"Model mean xG error:            {merged['model_xg_error'].mean():.3f}")

    if chat_df is not None and not chat_df.empty:
        print("\n===== SUMMARY (ChatGPT) =====")
        print(f"Matches with ChatGPT prediction: {len(chat_df)}")
        print(
            f"ChatGPT winner accuracy:         {chat_df['correct_winner_chatgpt'].mean():.3f}"
        )
        print(
            f"ChatGPT exact score accuracy:    {chat_df['correct_score_chatgpt'].mean():.3f}"
        )
    else:
        print("\n===== SUMMARY (ChatGPT) =====")
        print("No matches with ChatGPT predictions + results yet.")

    # Save full comparison for inspection
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved detailed comparison → {OUT_PATH}\n")


if __name__ == "__main__":
    run_comparison()
