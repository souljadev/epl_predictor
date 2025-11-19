import pandas as pd
from src.orchestrator.orchestrator import Orchestrator

o = Orchestrator("config.yaml")

teams = [
    "Burnley", "Chelsea", "Bournemouth", "West Ham", "Fulham", "Sunderland",
    "Wolves", "Liverpool", "Nott'm Forest", "Brighton", "Brentford",
    "Newcastle", "Man City"
]

print("=== Missing in attack_home ===")
print([t for t in teams if t not in o.attack_home])

print("=== Missing in defence_home ===")
print([t for t in teams if t not in o.defence_home])

print("=== Missing in attack_away ===")
print([t for t in teams if t not in o.attack_away])

print("=== Missing in defence_away ===")
print([t for t in teams if t not in o.defence_away])

print("=== Teams with NaN values ===")
for t in teams:
    ah = o.attack_home.get(t)
    da = o.defence_away.get(t)
    if pd.isna(ah) or pd.isna(da):
        print(t, ah, da)
