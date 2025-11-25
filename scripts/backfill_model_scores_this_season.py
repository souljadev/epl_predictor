from pathlib import Path
from datetime import datetime

import pandas as pd
import sqlite3
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../soccer_agent_local
SRC = ROOT / "src"
if str(SRC) not in sys.argv:
    sys.path.insert(0, str(SRC))

from predictor import train_models, predict_fixtures  # noqa: E402

DB_PATH = ROOT / "data" / "soccer_agent.db"
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def get_current_season_bounds():
    """
    Define season as July 1 -> June 30 that contains 'today'.
    e.g. if today is 2025-11-24, season = 2025-07-01 .. 2026-06-30
    """
    today = pd.Timestamp.today().normalize()
    season_year = today.year if today.month >= 7 else today.year - 1
    season_start = pd.Timestamp(season_year, 7, 1)
    season_end = pd.Timestamp(season_year + 1, 6, 30)
    return season_start, season_end


def load_results_from_db():
    """
    Load ALL results currently in the DB.
    This defines which matches we will backfill predictions for.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date, home_team, away_team, FTHG, FTAG FROM results",
        conn,
    )
    conn.close()

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["date"])
    df.drop(columns=["date"], inplace=True)
    return df


def insert_backfill_predictions(preds_df: pd.DataFrame, model_version: str):
    """
    Insert predictions into unified predictions table.

    preds_df must contain:
      Date, HomeTeam, AwayTeam, pH, pD, pA, ExpHomeGoals, ExpAwayGoals, ExpTotalGoals
    """
    if preds_df.empty:
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    now_str = datetime.utcnow().isoformat(timespec="seconds")

    rows = []
    for _, row in preds_df.iterrows():
        date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])

        pH = float(row["pH"])
        pD = float(row["pD"])
        pA = float(row["pA"])
        lamH = float(row["ExpHomeGoals"])
        lamA = float(row["ExpAwayGoals"])
        lamT = float(row.get("ExpTotalGoals", lamH + lamA))

        # Simple score prediction via rounded xG
        score_pred = f"{int(round(lamH))}-{int(round(lamA))}"

        rows.append(
            (
                date_str,
                home,
                away,
                model_version,
                None,  # dixon_coles_probs
                None,  # elo_probs
                None,  # ensemble_probs
                pH,
                pD,
                pA,
                lamH,
                lamA,
                lamT,
                score_pred,
                None,  # chatgpt_pred (leave existing one untouched if present)
                now_str,
            )
        )

    cur.executemany(
        """
        INSERT INTO predictions (
            date,
            home_team,
            away_team,
            model_version,
            dixon_coles_probs,
            elo_probs,
            ensemble_probs,
            home_win_prob,
            draw_prob,
            away_win_prob,
            exp_goals_home,
            exp_goals_away,
            exp_total_goals,
            score_pred,
            chatgpt_pred,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, home_team, away_team, model_version)
        DO UPDATE SET
            dixon_coles_probs=excluded.dixon_coles_probs,
            elo_probs=excluded.elo_probs,
            ensemble_probs=excluded.ensemble_probs,
            home_win_prob=excluded.home_win_prob,
            draw_prob=excluded.draw_prob,
            away_win_prob=excluded.away_win_prob,
            exp_goals_home=excluded.exp_goals_home,
            exp_goals_away=excluded.exp_goals_away,
            exp_total_goals=excluded.exp_total_goals,
            score_pred=excluded.score_pred,
            -- preserve existing ChatGPT score if we already have one
            chatgpt_pred=COALESCE(predictions.chatgpt_pred, excluded.chatgpt_pred),
            created_at=excluded.created_at;
        """,
        rows,
    )

    conn.commit()
    conn.close()


def main():
    print("\n==============================================")
    print("  Backfill model score predictions – THIS SEASON")
    print("==============================================\n")

    cfg = load_config()
    data_cfg = cfg.get("data", {})
    results_csv = ROOT / "data" / "raw" / "fbref_epl_xg.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    # Load full historical results from CSV for training (Dixon–Coles + Elo)
    results_hist = pd.read_csv(results_csv, parse_dates=["Date"])
    results_hist = results_hist.dropna(subset=["FTHG", "FTAG"])

    # Load played matches from DB (this defines what we backfill)
    results_db = load_results_from_db()
    if results_db.empty:
        print("No results found in DB – nothing to backfill.")
        return

    season_start, season_end = get_current_season_bounds()
    today = pd.Timestamp.today().normalize()

    # Filter DB results to current season and already-played matches
    end_bound = min(season_end, today)
    mask = (results_db["Date"] >= season_start) & (results_db["Date"] <= end_bound)
    season_df = results_db[mask].copy()

    if season_df.empty:
        print("No matches in current season found in DB – nothing to backfill.")
        return

    print(f"Season window: {season_start.date()} → {end_bound.date()}")
    print(f"Matches in DB for this season: {len(season_df)}")

    # Unique matchdays (dates with one or more matches)
    matchdays = sorted(season_df["Date"].unique())
    print(f"Unique matchdays to backfill: {len(matchdays)}")

    model_version = "dc_elo_ensemble_backfill"

    total_inserted = 0

    # Model config
    dc_cfg = cfg.get("model", {}).get("dc", {})
    elo_cfg = cfg.get("model", {}).get("elo", {})
    ensemble_cfg = cfg.get("model", {}).get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    for md in matchdays:
        md = pd.to_datetime(md)
        print(f"\n=== Matchday {md.date()} ===")

        # Training data: all historical matches strictly before this matchday
        train_df = results_hist[results_hist["Date"] < md].copy()
        if train_df.empty:
            print("  Skipping – no training data before this date.")
            continue

        # Fixtures on this matchday (from DB results)
        fixtures = season_df[season_df["Date"] == md].copy()
        if fixtures.empty:
            print("  Skipping – no fixtures found in DB for this date.")
            continue

        fixtures_df = fixtures[["Date", "home_team", "away_team"]].copy()
        fixtures_df.rename(
            columns={"home_team": "HomeTeam", "away_team": "AwayTeam"},
            inplace=True,
        )

        # Train models
        dc_model, elo_model = train_models(train_df, dc_cfg, elo_cfg)

        # Predict for this matchday
        preds_df = predict_fixtures(
            fixtures_df,
            dc_model,
            elo_model,
            w_dc=w_dc,
            w_elo=w_elo,
        )

        if preds_df.empty:
            print("  No predictions produced for this matchday.")
            continue

        insert_backfill_predictions(preds_df, model_version=model_version)
        total_inserted += len(preds_df)
        print(f"  Inserted/updated {len(preds_df)} predictions.")

    print("\n==============================================")
    print(f"Backfill complete. Total predictions inserted/updated: {total_inserted}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
