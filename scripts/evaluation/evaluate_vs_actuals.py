"""
Evaluate predictions vs actual results (DB-only).

- Merges predictions with results on (date, home_team, away_team)
- Computes:
  - Winner accuracy (H/D/A)
  - Brier score + log loss (if prob columns exist)
  - Score MAE (if predicted goals exist)
- Outputs:
  - Printed summary by model_version
  - CSV of per-match comparisons for the dashboard

Usage:
  python scripts/evaluation/evaluate_vs_actuals.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
import math
import pandas as pd


DB_PATH = Path("data/soccer_agent.db")
OUT_DIR = Path("data") / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _table_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_prob_cols(cols: list[str]) -> tuple[str | None, str | None, str | None]:
    """
    Try to locate probability columns for (H, D, A).
    Supports variants like:
      home_prob / draw_prob / away_prob
      H Prob / D Prob / A Prob
      p_home / p_draw / p_away
      home_win_prob, etc.
    """
    lc = [c.lower() for c in cols]

    def pick(keys: list[str]) -> str | None:
        for k in keys:
            for i, c in enumerate(lc):
                if c == k:
                    return cols[i]
        # contains-match fallback
        for k in keys:
            for i, c in enumerate(lc):
                if k in c:
                    return cols[i]
        return None

    h = pick(["h_prob", "home_prob", "p_home", "home win prob", "home_win_prob", "homewinprob", "h prob"])
    d = pick(["d_prob", "draw_prob", "p_draw", "draw prob", "d prob"])
    a = pick(["a_prob", "away_prob", "p_away", "away win prob", "away_win_prob", "awaywinprob", "a prob"])
    return h, d, a


def _safe_log(x: float) -> float:
    # clamp to avoid -inf
    eps = 1e-15
    x = min(max(x, eps), 1 - eps)
    return math.log(x)


def _winner_from_result_char(r: str) -> str | None:
    if r in ("H", "D", "A"):
        return r
    return None


def _winner_from_goals(fthg: int, ftag: int) -> str:
    if fthg > ftag:
        return "H"
    if fthg < ftag:
        return "A"
    return "D"


def _argmax_winner(hp: float, dp: float, ap: float) -> str:
    m = max(hp, dp, ap)
    if m == hp:
        return "H"
    if m == ap:
        return "A"
    return "D"


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH.resolve()}")

    conn = sqlite3.connect(DB_PATH)

    # Load results
    results_cols = _table_cols(conn, "results")
    r_date = _find_col(results_cols, ["date", "match_date"]) or "date"
    r_home = _find_col(results_cols, ["home_team"]) or "home_team"
    r_away = _find_col(results_cols, ["away_team"]) or "away_team"
    r_fthg = _find_col(results_cols, ["FTHG", "home_goals"]) or "FTHG"
    r_ftag = _find_col(results_cols, ["FTAG", "away_goals"]) or "FTAG"
    r_res  = _find_col(results_cols, ["Result", "result"]) or "Result"

    results = pd.read_sql_query(
        f"""
        SELECT
          {r_date} AS date,
          {r_home} AS home_team,
          {r_away} AS away_team,
          {r_fthg} AS FTHG,
          {r_ftag} AS FTAG,
          {r_res}  AS Result
        FROM results
        """,
        conn,
    )
    results["date"] = results["date"].astype(str)

    # Load predictions
    pred_cols = _table_cols(conn, "predictions")
    p_date = _find_col(pred_cols, ["date", "match_date"]) or "date"
    p_home = _find_col(pred_cols, ["home_team"]) or "home_team"
    p_away = _find_col(pred_cols, ["away_team"]) or "away_team"
    p_version = _find_col(pred_cols, ["model_version", "version"]) or "model_version"

    # Try to find predicted goals columns
    p_phg = _find_col(pred_cols, ["pred_home_goals", "home_pred_goals", "pred_hg", "phg", "pred_home"])
    p_pag = _find_col(pred_cols, ["pred_away_goals", "away_pred_goals", "pred_ag", "pag", "pred_away"])

    # Probability columns (H/D/A)
    p_hprob, p_dprob, p_aprob = _find_prob_cols(pred_cols)

    select_cols = [
        f"{p_date} AS date",
        f"{p_home} AS home_team",
        f"{p_away} AS away_team",
    ]
    if p_version in pred_cols:
        select_cols.append(f"{p_version} AS model_version")
    else:
        select_cols.append("'unknown' AS model_version")

    if p_phg:
        select_cols.append(f"{p_phg} AS pred_home_goals")
    if p_pag:
        select_cols.append(f"{p_pag} AS pred_away_goals")

    if p_hprob:
        select_cols.append(f"{p_hprob} AS pH")
    if p_dprob:
        select_cols.append(f"{p_dprob} AS pD")
    if p_aprob:
        select_cols.append(f"{p_aprob} AS pA")

    preds = pd.read_sql_query(
        f"SELECT {', '.join(select_cols)} FROM predictions",
        conn,
    )
    preds["date"] = preds["date"].astype(str)

    conn.close()

    # Merge
    merged = preds.merge(
        results,
        how="inner",
        on=["date", "home_team", "away_team"],
        suffixes=("", "_r"),
    )

    if merged.empty:
        print("No overlapping predictions/results found. Check team names + dates.")
        return

    # Actual winner
    merged["actual_winner"] = merged.apply(
        lambda r: _winner_from_result_char(str(r["Result"])) or _winner_from_goals(int(r["FTHG"]), int(r["FTAG"])),
        axis=1,
    )

    # Predicted winner
    if {"pH", "pD", "pA"}.issubset(merged.columns):
        merged["pred_winner"] = merged.apply(lambda r: _argmax_winner(float(r["pH"]), float(r["pD"]), float(r["pA"])), axis=1)
    else:
        merged["pred_winner"] = None

    # Winner correctness
    merged["winner_correct"] = (merged["pred_winner"] == merged["actual_winner"])

    # Score errors (if predicted goals exist)
    if {"pred_home_goals", "pred_away_goals"}.issubset(merged.columns):
        merged["pred_home_goals"] = pd.to_numeric(merged["pred_home_goals"], errors="coerce")
        merged["pred_away_goals"] = pd.to_numeric(merged["pred_away_goals"], errors="coerce")
        merged["mae_home_goals"] = (merged["pred_home_goals"] - merged["FTHG"]).abs()
        merged["mae_away_goals"] = (merged["pred_away_goals"] - merged["FTAG"]).abs()
        merged["mae_score_total"] = (merged["mae_home_goals"] + merged["mae_away_goals"]) / 2.0
    else:
        merged["mae_score_total"] = pd.NA

    # Brier + LogLoss (if probabilities exist)
    if {"pH", "pD", "pA"}.issubset(merged.columns):
        # one-hot actual
        merged["yH"] = (merged["actual_winner"] == "H").astype(int)
        merged["yD"] = (merged["actual_winner"] == "D").astype(int)
        merged["yA"] = (merged["actual_winner"] == "A").astype(int)

        # brier per row = sum_k (p_k - y_k)^2
        merged["brier_row"] = (
            (merged["pH"] - merged["yH"]) ** 2
            + (merged["pD"] - merged["yD"]) ** 2
            + (merged["pA"] - merged["yA"]) ** 2
        )

        # log loss per row = -log(p_true)
        def row_ll(r):
            if r["actual_winner"] == "H":
                return -_safe_log(float(r["pH"]))
            if r["actual_winner"] == "D":
                return -_safe_log(float(r["pD"]))
            return -_safe_log(float(r["pA"]))

        merged["logloss_row"] = merged.apply(row_ll, axis=1)
    else:
        merged["brier_row"] = pd.NA
        merged["logloss_row"] = pd.NA

    # Summary by model_version
    if "model_version" not in merged.columns:
        merged["model_version"] = "unknown"

    def agg(group: pd.DataFrame) -> pd.Series:
        out = {
            "matches": len(group),
            "winner_accuracy": float(group["winner_correct"].mean()) if group["pred_winner"].notna().any() else float("nan"),
        }
        if group["brier_row"].notna().any():
            out["brier"] = float(group["brier_row"].mean())
        else:
            out["brier"] = float("nan")
        if group["logloss_row"].notna().any():
            out["logloss"] = float(group["logloss_row"].mean())
        else:
            out["logloss"] = float("nan")
        if group["mae_score_total"].notna().any():
            out["mae_score"] = float(pd.to_numeric(group["mae_score_total"], errors="coerce").mean())
        else:
            out["mae_score"] = float("nan")
        return pd.Series(out)

    summary = merged.groupby("model_version", dropna=False).apply(agg).reset_index()

    print("\n==============================")
    print("EVALUATION SUMMARY (by model_version)")
    print("==============================")
    print(summary.to_string(index=False))

    # Write per-match comparison CSV
    out_cols = [
        "date", "home_team", "away_team",
        "FTHG", "FTAG", "Result",
        "model_version",
        "pred_winner", "actual_winner", "winner_correct",
    ]
    for c in ["pH", "pD", "pA", "pred_home_goals", "pred_away_goals", "brier_row", "logloss_row", "mae_score_total"]:
        if c in merged.columns:
            out_cols.append(c)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"match_level_eval_{stamp}.csv"
    merged[out_cols].to_csv(out_path, index=False)

    print("\nWrote match-level evaluation to:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
