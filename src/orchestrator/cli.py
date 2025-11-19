import argparse
import logging
from pathlib import Path

from .orchestrator import Orchestrator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_file(path_str: str, purpose: str) -> Path:
    """Validate that a file exists."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{purpose} file not found: {path}")
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Soccer Prediction Orchestrator CLI"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml",
    )

    parser.add_argument(
        "--fixtures",
        type=str,
        help="Path to fixtures CSV file for prediction mode",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model(s)",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predictions using fixtures file",
    )

    return parser


def main():
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    # --- Validate config ---
    config_path = validate_file(args.config, "Config")

    # --- Load orchestrator ---
    orchestrator = Orchestrator(config_path=config_path)

    # --- Mode Logic ---
    if args.train and args.predict:
        raise ValueError("You cannot pass both --train and --predict at the same time.")

    if args.train:
        logging.info("Starting training job...")
        orchestrator.train()
        logging.info("Training complete.")
        return

    if args.predict:
        if not args.fixtures:
            raise ValueError("--predict requires --fixtures <path>")
        fixtures_path = validate_file(args.fixtures, "Fixtures")
        logging.info(f"Running predictions on: {fixtures_path}")
        orchestrator.run_predictions(fixtures_path)
        logging.info("Prediction job complete.")
        return

    # --- If no mode passed ---
    raise ValueError("You must specify either --train or --predict.")


if __name__ == "__main__":
    main()
