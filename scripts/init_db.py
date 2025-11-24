from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]   # .../scripts -> project root
SRC = ROOT / "src"
sys.path.append(str(SRC))

from db import init_db  # noqa: E402


def main():
    print("\n================================")
    print("  Initializing Clean Database  ")
    print("================================\n")

    init_db()

    print("\nDone. Database created with empty tables.\n")


if __name__ == "__main__":
    main()
