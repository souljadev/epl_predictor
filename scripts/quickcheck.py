import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
print("Script path:", ROOT)
print("Parents:")
for i in range(5):
    print(i, ROOT.parents[i])

print("Current sys.path:")
for p in sys.path:
    print(" -", p)
