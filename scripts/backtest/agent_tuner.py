import itertools
import json
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from datetime import datetime

import yaml
import pandas as pd

from backtest_expanding import backtest_expanding, ROOT, CONFIG_PATH

HISTORY_FILE = ROOT / "models" / "history" / "agent_history.csv"


def load_base_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def fitness_function(metrics: dict) -> float:
    """
    Higher is better.
    Reward:
      - higher accuracy
      - lower brier
      - lower log_loss
    You can tweak weights as you like.
    """
    if not metrics:
        return -1e9

    acc = metrics["accuracy"]
    brier = metrics["brier"]
    log_loss = metrics["log_loss"]

    # simple example:
    # accuracy weight 1.0, brier weight 1.0, log_loss weight 0.5
    score = acc - brier - 0.5 * log_loss
    return score


def build_search_space():
    """
    Define grid of hyperparameters to try.
    You can expand this as needed.
    """
    w_dc_list = [0.4, 0.6, 0.7]
    w_elo_list = [0.6, 0.4, 0.3]

    rho_init_list = [0.0, 0.05, 0.1]
    max_iter_list = [200, 300]
    tol_list = [1e-5, 1e-6]

    k_factor_list = [16.0, 18.0, 20.0]
    home_adv_list = [45.0, 55.0, 65.0]

    configs = []
    for w_dc, w_elo, rho_init, max_iter, tol, k_factor, home_adv in itertools.product(
        w_dc_list, w_elo_list,
        rho_init_list, max_iter_list, tol_list,
        k_factor_list, home_adv_list
    ):
        if abs(w_dc + w_elo - 1.0) > 1e-6:
            continue

        cfg_mod = {
            "ensemble": {"w_dc": w_dc, "w_elo": w_elo},
            "dc": {"rho_init": rho_init, "max_iter": max_iter, "tol": tol},
            "elo": {"k_factor": k_factor, "home_advantage": home_adv}
        }
        configs.append(cfg_mod)
    return configs


def apply_cfg_mod(base_cfg: dict, cfg_mod: dict) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["model"]["ensemble"]["w_dc"] = cfg_mod["ensemble"]["w_dc"]
    cfg["model"]["ensemble"]["w_elo"] = cfg_mod["ensemble"]["w_elo"]

    cfg["model"]["dc"]["rho_init"] = cfg_mod["dc"]["rho_init"]
    cfg["model"]["dc"]["max_iter"] = cfg_mod["dc"]["max_iter"]
    cfg["model"]["dc"]["tol"] = cfg_mod["dc"]["tol"]

    cfg["model"]["elo"]["k_factor"] = cfg_mod["elo"]["k_factor"]
    cfg["model"]["elo"]["home_advantage"] = cfg_mod["elo"]["home_advantage"]
    return cfg


def run_single_config(args):
    base_cfg, cfg_mod, idx = args
    run_tag = f"run{idx}"

    cfg = apply_cfg_mod(base_cfg, cfg_mod)

    metrics, csv_path = backtest_expanding(config=cfg, run_tag=run_tag)
    if metrics is None:
        return None

    fit = fitness_function(metrics)

    return {
        "run_tag": run_tag,
        "config": cfg_mod,
        "metrics": metrics,
        "fitness": fit,
        "csv_path": str(csv_path),
        "timestamp": datetime.utcnow().isoformat()
    }


def save_history(all_results):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        if r is None:
            continue
        flat = {
            "run_tag": r["run_tag"],
            "fitness": r["fitness"],
            "csv_path": r["csv_path"],
            "timestamp": r["timestamp"],
        }
        for k, v in r["metrics"].items():
            flat[f"metric_{k}"] = v

        cfg = r["config"]
        flat["w_dc"] = cfg["ensemble"]["w_dc"]
        flat["w_elo"] = cfg["ensemble"]["w_elo"]
        flat["rho_init"] = cfg["dc"]["rho_init"]
        flat["dc_max_iter"] = cfg["dc"]["max_iter"]
        flat["dc_tol"] = cfg["dc"]["tol"]
        flat["elo_k_factor"] = cfg["elo"]["k_factor"]
        flat["elo_home_adv"] = cfg["elo"]["home_advantage"]

        rows.append(flat)

    if not rows:
        print("No successful runs to save.")
        return

    df_new = pd.DataFrame(rows)

    if HISTORY_FILE.exists():
        df_old = pd.read_csv(HISTORY_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(HISTORY_FILE, index=False)
    print(f"\nAgent history updated → {HISTORY_FILE}")


def main():
    base_cfg = load_base_config()
    search_space = build_search_space()

    print(f"Total configs to try: {len(search_space)}")
    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} parallel workers.")

    args = [(base_cfg, cfg_mod, idx) for idx, cfg_mod in enumerate(search_space)]

    with Pool(processes=n_workers) as pool:
        all_results = list(pool.map(run_single_config, args))

    save_history(all_results)

    # Select best config
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("No valid results, cannot pick best config.")
        return

    best = max(valid_results, key=lambda r: r["fitness"])

    print("\n===== BEST CONFIG FOUND =====")
    print(f"Run tag:   {best['run_tag']}")
    print(f"Fitness:   {best['fitness']:.4f}")
    print(f"CSV path:  {best['csv_path']}")
    print("Metrics:")
    for k, v in best["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Config:")
    print(json.dumps(best["config"], indent=2))


if __name__ == "__main__":
    main()
