# Local Agentic EPL Soccer Prediction AI

This is a **self-contained, local** project for **English Premier League** predictions.

- Trains a **Dixon–Coles Poisson** model and a **simple Elo** model.
- Blends them into an ensemble for win/draw/loss and goal expectations.
- Includes an **agent layer** that learns from past prediction errors and auto-tunes parameters.

## 1) Setup

```bash
cd soccer_agent_local
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run Predictions for Fixtures

Edit `data/raw/fixtures_today.csv` with matches you want to predict, then:

```bash
python -m src.orchestrator.cli --config config.yaml --fixtures data/raw/fixtures_today.csv
```

This prints a table and saves a CSV under `models/predictions/` like `predictions_YYYYMMDD_HHMMSS.csv`.

Columns include:

- `pH`, `pD`, `pA` — probabilities for **home win**, **draw**, **away win**.
- `ExpHomeGoals`, `ExpAwayGoals`, `ExpTotalGoals` — expected goals.
- `PredWinner` — most likely outcome.

## 3) After Matches Finish: Self-Training Loop

1. Create a CSV of **actual EPL results** for those fixtures, e.g.:

```csv
Date,Season,HomeTeam,AwayTeam,FTHG,FTAG,Result
2025-10-26,2025,Arsenal,Crystal Palace,1,0,H
...
```

2. Run the ingest + retrain script:

```bash
python -m src.agent.ingest_and_retrain   --config config.yaml   --new_results data/raw/new_results_round10.csv   --predictions models/predictions/predictions_YYYYMMDD_HHMMSS.csv
```

This will:

- Append new results into your main EPL results file.
- Merge predictions + actuals and save an evaluation CSV in `models/history/`.
- Run the **structural auto-tuner**.
- Retrain the models with updated parameters.

## 4) Auto-Tuner Details

`src/agent/auto_tuner_structural.py` computes:

- Outcome metrics: log loss, Brier score.
- Score metrics: MAE for home, away, and total goals.
- Low-scoring frequency (<= 2 goals).
- Home-win frequency.

Then it automatically adjusts:

- Elo `k_factor` (how reactive the ratings are).
- Elo `home_advantage` (home-field boost in Elo points).
- Dixon–Coles `rho_init` (low-score correlation).
- Ensemble weights (`w_dc`, `w_elo`) between Poisson and Elo.

Over time, this makes the system **self-improving**, especially for EPL scorelines and outcomes.

## 5) Note

This project is **local-only** and does **not** fetch or display regulated betting lines. It is intended for learning, analytics, and experimentation.
