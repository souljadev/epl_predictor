import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def _compute_metrics(eval_df: pd.DataFrame):
    mapping = {"H": 0, "D": 1, "A": 2}
    y_true = eval_df["Result"].map(mapping).values
    probs = eval_df[["pH", "pD", "pA"]].values
    eps = 1e-12
    log_probs_true = np.log(probs[np.arange(len(y_true)), y_true] + eps)
    log_loss = float(-np.mean(log_probs_true))

    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    mae_home = float(np.mean(np.abs(eval_df["FTHG"] - eval_df["ExpHomeGoals"])))
    mae_away = float(np.mean(np.abs(eval_df["FTAG"] - eval_df["ExpAwayGoals"])))
    mae_total = float(
        np.mean(
            np.abs(
                (eval_df["FTHG"] + eval_df["FTAG"])
                - (eval_df["ExpHomeGoals"] + eval_df["ExpAwayGoals"])
            )
        )
    )

    low_actual = float(np.mean((eval_df["FTHG"] + eval_df["FTAG"]) <= 2))
    low_pred = float(np.mean((eval_df["ExpHomeGoals"] + eval_df["ExpAwayGoals"]) <= 2.5))

    home_win_actual = float(np.mean(eval_df["Result"] == "H"))
    home_win_pred = float(np.mean(eval_df["pH"]))

    return {
        "log_loss": log_loss,
        "brier": brier,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
        "low_actual": low_actual,
        "low_pred": low_pred,
        "home_win_actual": home_win_actual,
        "home_win_pred": home_win_pred,
    }

def _clip(val, lo, hi):
    return max(lo, min(hi, val))

def tune_config(config_path: str, eval_csv: str, out_path: str | None = None):
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    eval_df = pd.read_csv(eval_csv)

    metrics = _compute_metrics(eval_df)

    tcfg = cfg.get("tuning", {})
    low_tol = tcfg.get("low_score_tolerance", 0.05)
    home_tol = tcfg.get("home_win_tolerance", 0.03)
    k_step = tcfg.get("k_step", 2.0)
    rho_step = tcfg.get("rho_step", 0.02)
    home_adv_step = tcfg.get("home_adv_step", 5.0)
    w_step = tcfg.get("w_step", 0.05)

    elo_cfg = cfg.setdefault("model", {}).setdefault("elo", {})
    k = float(elo_cfg.get("k_factor", 18.0))
    home_adv = float(elo_cfg.get("home_advantage", 55.0))

    if metrics["log_loss"] > 1.2:
        k += k_step
    elif metrics["log_loss"] < 0.9:
        k -= k_step
    elo_cfg["k_factor"] = _clip(k, 5.0, 40.0)

    if metrics["home_win_actual"] - metrics["home_win_pred"] > home_tol:
        home_adv += home_adv_step
    elif metrics["home_win_pred"] - metrics["home_win_actual"] > home_tol:
        home_adv -= home_adv_step
    elo_cfg["home_advantage"] = _clip(home_adv, 20.0, 100.0)

    dc_cfg = cfg.setdefault("model", {}).setdefault("dc", {})
    rho = float(dc_cfg.get("rho_init", 0.0))
    if metrics["low_actual"] - metrics["low_pred"] > low_tol:
        rho += rho_step
    elif metrics["low_pred"] - metrics["low_actual"] > low_tol:
        rho -= rho_step
    dc_cfg["rho_init"] = _clip(rho, -0.2, 0.2)

    ens_cfg = cfg.setdefault("model", {}).setdefault("ensemble", {})
    w_dc = float(ens_cfg.get("w_dc", 0.6))
    w_elo = float(ens_cfg.get("w_elo", 0.4))
    if metrics["mae_total"] > 1.3:
        w_dc -= w_step; w_elo += w_step
    elif metrics["mae_total"] < 0.9:
        w_dc += w_step; w_elo -= w_step
    total_w = max(1e-6, w_dc + w_elo)
    w_dc = _clip(w_dc / total_w, 0.3, 0.8)
    w_elo = _clip(w_elo / total_w, 0.2, 0.7)
    ens_cfg["w_dc"] = w_dc; ens_cfg["w_elo"] = w_elo

    target_path = Path(out_path) if out_path else cfg_path
    target_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return metrics, cfg

def main():
    parser = argparse.ArgumentParser(description="Structural auto-tuner for EPL soccer agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--out_config", default=None)
    args = parser.parse_args()

    metrics, cfg = tune_config(args.config, args.eval_csv, args.out_config)
    print("=== Tuning metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nUpdated parameters:")
    print("  Elo k_factor:", cfg["model"]["elo"]["k_factor"])
    print("  Elo home_advantage:", cfg["model"]["elo"]["home_advantage"])
    print("  DC rho_init:", cfg["model"]["dc"]["rho_init"])
    print("  Ensemble (w_dc, w_elo):", cfg["model"]["ensemble"]["w_dc"], cfg["model"]["ensemble"]["w_elo"])

if __name__ == "__main__":
    main()
