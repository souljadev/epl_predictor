import pandas as pd
import yaml
from pathlib import Path
from ..features.features import build_basic_features
from ..features.utils import train_test_split_by_season, set_seed
from ..models.poisson_dc import DixonColes
from ..models.elo import EloModel

def train_models(config_path: str = "config.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg.get("seed", 1337))
    results_csv = cfg["data"]["results_csv"]
    results = pd.read_csv(results_csv, parse_dates=["Date"])
    feats = build_basic_features(results)
    feats.to_csv(cfg["data"]["features_csv"], index=False)

    train_df, test_df = train_test_split_by_season(feats, cfg["train"]["test_start_season"])

    dc_cfg = cfg["model"]["dc"]
    dc = DixonColes(
        rho_init=dc_cfg.get("rho_init", 0.0),
        max_iter=dc_cfg.get("max_iter", 300),
        tol=dc_cfg.get("tol", 1e-6),
    ).fit(train_df if not train_df.empty else feats)

    elo_cfg = cfg["model"]["elo"]
    elo = EloModel(
        k_factor=elo_cfg.get("k_factor", 18.0),
        home_advantage=elo_cfg.get("home_advantage", 55.0),
    ).fit(feats)

    return dc, elo
